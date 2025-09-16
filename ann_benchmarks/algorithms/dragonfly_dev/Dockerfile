FROM ann-benchmarks

# Install only Python dependencies needed for the client
RUN pip3 install redis

# Simple entrypoint that just runs the benchmark without starting any server
ENTRYPOINT ["python3", "-u", "run_algorithm.py"]

