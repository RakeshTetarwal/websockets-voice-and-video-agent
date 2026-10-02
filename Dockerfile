FROM astral/uv:python3.12-bookworm-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY requirements.txt .

RUN uv pip install --default-timeout=100 --no-cache-dir -r requirements.txt

COPY . .

CMD [""]
