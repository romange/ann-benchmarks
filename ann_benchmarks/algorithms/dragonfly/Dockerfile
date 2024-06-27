FROM ann-benchmarks

RUN apt-get update && apt-get install -y curl redis-tools && rm -rf /var/lib/apt/lists/*
COPY ann_benchmarks/algorithms/dragonfly/get_df.sh /tmp/

RUN /tmp/get_df.sh && pip3 install redisearch redis

