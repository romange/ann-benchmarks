import subprocess
import sys
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy

from redis import Redis

from ..base.module import BaseANN


class Dragonfly(BaseANN):
    def __init__(self, metric, M):
        self.metric = metric
        self.ef_construction = 500
        self.M = M
        self.index_name = "ann"
        self.field_name = "vector"
        self.threads = 4  # server-side logical threads parameter used in set_query_arguments

        # Search concurrency and timeouts (can be overridden via env vars)
        # These govern client-side parallelism and socket behavior.
        self.search_threads = int(os.getenv("DF_SEARCH_THREADS", str(max(4, (os.cpu_count() or 4)))))
        self.search_read_timeout = float(os.getenv("DF_SEARCH_TIMEOUT", "15.0"))
        self.connect_timeout = float(os.getenv("DF_CONNECT_TIMEOUT", "10.0"))

        # Redis/Dragonfly endpoint configuration via environment variables
        # Defaults: host=localhost, port=6379
        self.host = os.getenv("DF_HOST", "localhost")
        try:
            self.port = int(os.getenv("DF_PORT", "6379"))
        except Exception:
            self.port = 6379

        # Internal holders for batch results/latencies
        self._batch_results = None
        self._batch_latencies = None

    def fit(self, X):
        print("Running in local mode")

        # Convert to float32 if needed
        X = X.astype(numpy.float32)

        # Connect to Dragonfly endpoint
        print(f"Connecting to Dragonfly at {self.host}:{self.port} ...")
        self.redis = Redis(host=self.host, port=self.port, decode_responses=False)

        try:
          self.redis.execute_command("FT.DROPINDEX", self.index_name)
        except Exception:
          pass
        self.redis.execute_command("FLUSHALL")

        # Create index
        args = [
            "FT.CREATE",
            self.index_name,
            "SCHEMA",
            self.field_name,
            "VECTOR",
            "HNSW",
            "10",  # number of remaining arguments
            "TYPE",
            "FLOAT32",
            "DIM",
            X.shape[1],
            "DISTANCE_METRIC",
            {"angular": "COSINE", "euclidean": "L2"}[self.metric],
            "M",
            self.M,
            "EF_CONSTRUCTION",
            self.ef_construction,
        ]
        print("Running Redis command:", args)
        self.redis.execute_command(*args, target_nodes="random")

        # Insert vectors
        p = self.redis.pipeline(transaction=False)
        for i, v in enumerate(X):
            p.execute_command("HSET", i, self.field_name, v.tobytes())
            if i % 1000 == 999:
                p.execute()
                if i % 100000 == 99999:
                    print(f"Added {i} arguments")
                p.reset()
        p.execute()

    def set_query_arguments(self, ef):
        self.ef = ef // self.threads

    def query(self, v, n):
        # Convert to float32 to match the index
        v = v.astype(numpy.float32)
        q = [
            "FT.SEARCH",
            self.index_name,
            f"*=>[KNN {n} @{self.field_name} $BLOB EF_RUNTIME {self.ef}]",
            "NOCONTENT",
            "SORTBY",
            "__vector_score",
            "LIMIT",
            "0",
            str(n),
            "PARAMS",
            "2",
            "BLOB",
            v.tobytes(),
            "DIALECT",
            "2",
        ]
        resp = self.redis.execute_command(*q, target_nodes="random")

        # Robustly extract document IDs from RediSearch response
        ids = []
        if isinstance(resp, (list, tuple)) and len(resp) > 1:
            for item in resp[1:]:
                if isinstance(item, (bytes, str)):
                    try:
                        ids.append(int(item))
                    except Exception:
                        continue
                elif isinstance(item, (list, tuple)) and len(item) > 0:
                    doc_id = item[0]
                    if isinstance(doc_id, (bytes, str)):
                        try:
                            ids.append(int(doc_id))
                        except Exception:
                            continue
        return ids

    def batch_query(self, X, n):
        """Run many queries concurrently using a thread pool.

        Each thread holds its own Redis client bound to a single dedicated TCP connection.
        This enables true I/O parallelism across many sockets.
        """
        total = len(X)
        self._batch_results = [None] * total
        self._batch_latencies = [0.0] * total

        tls = threading.local()

        def get_client() -> Redis:
            # Lazily create one Redis client per OS thread.
            cli = getattr(tls, "cli", None)
            if cli is None:
                cli = Redis(
                    host=self.host,
                    port=self.port,
                    decode_responses=False,
                    socket_timeout=self.search_read_timeout,
                    socket_connect_timeout=self.connect_timeout,
                    socket_keepalive=True,
                    health_check_interval=30,
                    retry_on_timeout=True,
                )
                # warm up connection
                try:
                    cli.ping()
                except Exception:
                    pass
                tls.cli = cli
            return cli

        def run_one(idx: int):
            v = X[idx].astype(numpy.float32)
            q = [
                "FT.SEARCH",
                self.index_name,
                f"*=>[KNN {n} @{self.field_name} $BLOB EF_RUNTIME {self.ef}]",
                "NOCONTENT",
                "SORTBY",
                "__vector_score",
                "LIMIT",
                "0",
                str(n),
                "PARAMS",
                "2",
                "BLOB",
                v.tobytes(),
                "DIALECT",
                "2",
            ]
            cli = get_client()
            t0 = time.time()
            try:
                resp = cli.execute_command(*q, target_nodes="random")
                # Robust parsing of RediSearch response to extract doc IDs
                res = []
                if isinstance(resp, (list, tuple)) and len(resp) > 1:
                    for item in resp[1:]:
                        if isinstance(item, (bytes, str)):
                            try:
                                res.append(int(item))
                            except Exception:
                                continue
                        elif isinstance(item, (list, tuple)) and len(item) > 0:
                            doc_id = item[0]
                            if isinstance(doc_id, (bytes, str)):
                                try:
                                    res.append(int(doc_id))
                                except Exception:
                                    continue
            except Exception:
                res = []
            finally:
                self._batch_results[idx] = res
                self._batch_latencies[idx] = max(0.0, time.time() - t0)

        with ThreadPoolExecutor(max_workers=self.search_threads) as ex:
            futures = [ex.submit(run_one, i) for i in range(total)]
            for _ in as_completed(futures):
                pass

        # Close per-thread clients
        # Note: we cannot iterate tls for all threads; rely on GC/socket timeouts after pool shutdown.
        # To be explicit, create a finalizer thread to close if present in this main thread (no-op for workers).
        try:
            cli = getattr(tls, "cli", None)
            if cli is not None:
                cli.close()
        except Exception:
            pass

    def get_batch_results(self):
        return self._batch_results

    def get_batch_latencies(self):
        return self._batch_latencies

    def done(self) -> None:
        try:
            self.redis.close()
        except Exception:
            pass

    def __str__(self):
        return f"Dragonfly(M={self.M}, ef={self.ef}, thread={self.threads}, search_threads={self.search_threads})"
