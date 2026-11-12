FROM python:3.11-slim
WORKDIR /opt
COPY vendor/CodeTracer/pyproject.toml /opt/CodeTracer/pyproject.toml
COPY vendor/CodeTracer/src /opt/CodeTracer/src
RUN pip install --no-cache-dir /opt/CodeTracer
COPY downstream/run_one.py /opt/run_one.py
WORKDIR /work
ENTRYPOINT ["python", "/opt/run_one.py", "--vendor", "/opt/CodeTracer"]
