FROM ann-benchmarks

RUN apt-get update && apt-get install -y curl redis-tools && rm -rf /var/lib/apt/lists/*
COPY ann_benchmarks/algorithms/dragonfly/get_df.sh /tmp/

RUN /tmp/get_df.sh && pip3 install redisearch redis

# Create startup script that runs Dragonfly in background
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Starting Dragonfly server..."\n\
/usr/local/bin/dragonfly --logtostderr --port=6379 &\n\
sleep 3\n\
echo "Dragonfly server started, running benchmark..."\n\
exec "$@"' > /usr/local/bin/start-dragonfly.sh && \
chmod +x /usr/local/bin/start-dragonfly.sh

ENTRYPOINT ["/usr/local/bin/start-dragonfly.sh", "python3", "-u", "run_algorithm.py"]

