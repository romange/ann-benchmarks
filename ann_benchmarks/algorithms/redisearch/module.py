import subprocess
import sys
import time
import numpy

from redis import Redis

from ..base.module import BaseANN


class Redisearch(BaseANN):
    def __init__(self, metric, M):
        self.metric = metric
        self.ef_construction = 500
        self.M = M
        self.index_name = "ann"
        self.field_name = "vector"

    def fit(self, X):
        print("Running in local mode")

        # Convert to float32 if needed
        X = X.astype(numpy.float32)

        # Connect to Redis
        print("Connecting to Redis...")
        self.redis = Redis(host="localhost", port=6379, decode_responses=False)

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
            {"angular": "cosine", "euclidean": "l2"}[self.metric],
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
                if i % 100000 == 199999:
                    print(f"Added {i} arguments")
                p.reset()
        p.execute()

    def set_query_arguments(self, ef):
        self.ef = ef

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
        return [int(doc) for doc in self.redis.execute_command(*q, target_nodes="random")[1:]]

    def __str__(self):
        return f"Redisearch(M={self.M}, ef={self.ef})"
