FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pytest
COPY repo/ /workspace/
WORKDIR /workspace
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
