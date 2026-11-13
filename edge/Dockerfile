FROM python:3.11-slim

WORKDIR /app/edge

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/edge

ENV PYTHONPATH=/app/edge
ENV CLOUD_API_URL=http://backend:8000
ENV NODE_ID=edge-node-01
ENV SPLIT_LAYER=layer2
ENV COMPRESSION=int8

EXPOSE 9090

CMD ["python", "edge_node.py"]
