FROM python:3.12-slim
ARG ENABLE_PADDLE=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    gcc g++ cmake make \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock* README.md ./
RUN if [ "$ENABLE_PADDLE" = "true" ]; then \
      uv sync --frozen --no-dev --extra paddle 2>/dev/null || uv sync --no-dev --extra paddle; \
    else \
      uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev; \
    fi

COPY . .

RUN mkdir -p /app/data/processed /app/data/uploads

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000 8501

ENTRYPOINT ["/entrypoint.sh"]
CMD ["api"]
