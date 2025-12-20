FROM python:3.12-slim

# Add metadata labels
LABEL maintainer="Knowledge Bot Team"
LABEL description="Knowledgebase Ingestion Service for Knowledge Bot"
LABEL version="1.0.2"

# Set environment variables for better logging and Python behavior
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=random
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONPATH="/app"
# Force Python logging to output to stdout/stderr
ENV PYTHONIOENCODING=utf-8

WORKDIR /app

# Log build start
RUN echo "🚀 Starting Knowledgebase Ingestion Docker build..." && \
    echo "📅 Build timestamp: $(date)" && \
    echo "🐳 Using Python: $(python3 --version)"

# Install system dependencies with detailed logging
RUN echo "📦 Installing system dependencies..." && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    curl \
    && \
    echo "✅ System dependencies installed" && \
    rm -rf /var/lib/apt/lists/* && \
    echo "🧹 Cleaned up apt cache"

# Copy and install Python dependencies
COPY requirements.txt .
RUN echo "🐍 Installing Python dependencies..." && \
    pip install --no-cache-dir -r requirements.txt && \
    echo "✅ Python dependencies installed" && \
    pip list | grep -E "(fastapi|uvicorn|google-genai|boto3)"

# Copy application code with logging
COPY services/knowledgebase_ingestion/ ./services/knowledgebase_ingestion/
COPY shared ./shared/
RUN echo "📁 Application code copied" && \
    ls -la services/knowledgebase_ingestion/ && \
    echo "📄 Main application file:" && \
    head -n 10 services/knowledgebase_ingestion/main.py

# Set working directory
WORKDIR /app/services/knowledgebase_ingestion

# Create logs directory
RUN mkdir -p /app/logs && \
    echo "📝 Logs directory created"

# Update health check to use the `/health` endpoint
# Increased start-period and retries to reduce premature restarts
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD sh -c 'curl -f http://localhost:${KB_INGESTION_PORT:-${PORT:-8001}}/health || exit 1'

# Run the application with uvicorn - prefer KB_INGESTION_PORT, else PORT, else default 8001
# Enable info log level (debug can be too verbose) and access logs for better visibility
# Using exec to ensure proper signal handling and log flushing
CMD ["sh", "-c", "echo '🚀 Starting Knowledgebase Ingestion Service...' && echo '🌐 Runtime Port Check: KB_INGESTION_PORT='${KB_INGESTION_PORT}' PORT='${PORT} && exec python -m uvicorn main:app --host 0.0.0.0 --port ${KB_INGESTION_PORT:-${PORT:-8001}} --log-level debug --access-log"]
