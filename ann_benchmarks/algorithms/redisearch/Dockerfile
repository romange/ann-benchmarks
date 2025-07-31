FROM redislabs/redisearch as redisearch
FROM ann-benchmarks

# Install Redis
RUN apt-get update && apt-get install -y curl lsb-core
RUN curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list
RUN apt-get update
RUN apt-get install -y redis

COPY --from=redisearch /usr/lib/redis/modules/redisearch.so /usr/lib/redis/modules/redisearch.so

RUN pip3 install redisearch redis

# Create startup script that runs Redis with RediSearch module in background
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Starting Redis server with RediSearch module..."\n\
redis-server --port 6379 --loadmodule /usr/lib/redis/modules/redisearch.so &\n\
sleep 3\n\
echo "Redis server started, running benchmark..."\n\
exec "$@"' > /usr/local/bin/start-redis.sh && \
chmod +x /usr/local/bin/start-redis.sh

ENTRYPOINT ["/usr/local/bin/start-redis.sh", "python3", "-u", "run_algorithm.py"]
