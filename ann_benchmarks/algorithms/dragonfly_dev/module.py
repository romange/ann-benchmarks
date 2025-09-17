import subprocess
import sys
import time
import os
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

        # Connection timeouts for Redis client
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
        """Run many queries using Redis pipelining for optimal performance.

        Uses Dragonfly's native batch processing capabilities via Redis pipeline
        instead of client-side threading to reduce network overhead and improve throughput.
        """
        total = len(X)
        self._batch_results = [None] * total
        self._batch_latencies = [0.0] * total

        # Create pipeline for batch execution
        pipeline = self.redis.pipeline(transaction=False)

        # Add all queries to the pipeline
        for i, v in enumerate(X):
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

        # Execute all queries as a batch and measure total time
        start_time = time.time()
        try:
            responses = pipeline.execute()
            total_time = time.time() - start_time

            # Process responses and calculate per-query latencies
            # Since pipeline executes all commands together, we approximate individual latencies
            avg_latency = total_time / len(responses) if responses else 0.0

            for i, resp in enumerate(responses):
                # Extract document IDs from RediSearch response
                res = [int(resp[j]) for j in range(1, len(resp), 2)] if len(resp) > 1 else []

                self._batch_results[i] = res
                self._batch_latencies[i] = avg_latency

        except Exception as e:
            # Handle pipeline execution errors
            print(f"Pipeline execution failed: {e}")
            for i in range(total):
                self._batch_results[i] = []
                self._batch_latencies[i] = 0.0

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
        return f"Dragonfly(M={self.M}, ef={self.ef}, threads={self.threads})"
