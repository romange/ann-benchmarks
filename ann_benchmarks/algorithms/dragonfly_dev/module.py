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
        self.pipeline_size = int(os.getenv("DF_PIPELINE_SIZE", "32"))
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
        self.redis = Redis(
            host=self.host,
            port=self.port,
            decode_responses=False,
            socket_timeout=self.search_read_timeout,
            socket_connect_timeout=self.connect_timeout
        )

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

        # Extract document IDs from RediSearch response
        return [int(resp[i]) for i in range(1, len(resp), 2)] if len(resp) > 1 else []

    def batch_query(self, X, n):
        """Run many queries using multi-threaded Redis pipelining for optimal performance.

        Each thread creates its own Redis client and pipeline, distributing the workload
        while using Dragonfly's native batch processing capabilities within each thread.
        """
        total = len(X)
        self._batch_results = [None] * total
        self._batch_latencies = [0.0] * total

        # Split work among threads
        chunk_size = max(1, total // self.search_threads)
        chunks = [X[i:i + chunk_size] for i in range(0, total, chunk_size)]

        print(f"batch_query: {total} queries, {len(chunks)} threads, ~{chunk_size} queries per pipeline")

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

        def process_chunk(chunk_idx: int, chunk_data):
            cli = get_client()
            start_idx = chunk_idx * chunk_size

            print(f"Thread {chunk_idx}: processing {len(chunk_data)} queries with pipeline_size={self.pipeline_size}")

            # Process data in pipeline-sized batches
            for batch_start in range(0, len(chunk_data), self.pipeline_size):
                batch_end = min(batch_start + self.pipeline_size, len(chunk_data))
                batch = chunk_data[batch_start:batch_end]

                pipeline = cli.pipeline(transaction=False)

                # Add queries to pipeline (up to pipeline_size)
                for i, v in enumerate(batch):
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
                    pipeline.execute_command(*q, target_nodes="random")

                # Execute pipeline batch and measure time
                start_time = time.time()
                try:
                    responses = pipeline.execute()
                    total_time = time.time() - start_time
                    avg_latency = total_time / len(responses) if responses else 0.0

                    print(f"Thread {chunk_idx}: pipeline batch executed {len(responses)} queries in {total_time:.3f}s")

                    # Process responses for this batch
                    for i, resp in enumerate(responses):
                        global_idx = start_idx + batch_start + i
                        if global_idx < total:
                            # Extract document IDs from RediSearch response
                            res = [int(resp[j]) for j in range(1, len(resp), 2)] if len(resp) > 1 else []
                            self._batch_results[global_idx] = res
                            self._batch_latencies[global_idx] = avg_latency

                except Exception as e:
                    print(f"Pipeline batch execution failed for chunk {chunk_idx}, batch {batch_start}: {e}")
                    for i in range(len(batch)):
                        global_idx = start_idx + batch_start + i
                        if global_idx < total:
                            self._batch_results[global_idx] = []
                            self._batch_latencies[global_idx] = 0.0

        # Execute chunks in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.search_threads) as ex:
            futures = [ex.submit(process_chunk, i, chunk) for i, chunk in enumerate(chunks)]
            for _ in as_completed(futures):
                pass

        # Close per-thread clients
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
        return f"Dragonfly(M={self.M}, ef={self.ef}, threads={self.threads}, search_threads={self.search_threads})"
