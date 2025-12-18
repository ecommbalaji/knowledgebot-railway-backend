FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY services/knowledgebase_ingestion/ ./services/knowledgebase_ingestion/
COPY shared ./shared/

# Set working directory
WORKDIR /app/services/knowledgebase_ingestion



# Add health check (prefer KB_INGESTION_PORT, else PLATFORM PORT, else default)
# Increased start-period and retries to reduce premature restarts
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
    CMD sh -c 'curl -f http://localhost:${KB_INGESTION_PORT:-8001}/health || exit 1'

# Run the application with uvicorn - prefer KB_INGESTION_PORT, else PORT, else default
CMD sh -c 'uvicorn main:app --host 0.0.0.0 --port ${KB_INGESTION_PORT:-8001}'
