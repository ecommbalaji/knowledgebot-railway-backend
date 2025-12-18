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

# Expose port
EXPOSE 8001

# Run the application with uvicorn - use PORT env var from Railway
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8001}
