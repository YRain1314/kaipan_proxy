FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py .
ENV PORT=8080
ENV KLINES_LIMIT=500
ENV CACHE_TTL_SECONDS=300
ENV REFRESH_INTERVAL_SECONDS=900
CMD ["python", "server.py"]
