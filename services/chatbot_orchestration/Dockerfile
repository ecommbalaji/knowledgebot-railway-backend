FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY services/chatbot_orchestration/ ./services/chatbot_orchestration/
COPY shared ./shared/

# Set working directory
WORKDIR /app/services/chatbot_orchestration

# Set the PYTHONPATH environment variable to include the /app directory
ENV PYTHONPATH="/app"

# Update health check to use the `/health` endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD sh -c 'curl -f http://localhost:${CHATBOT_ORCH_PORT:-${PORT:-8003}}/health || exit 1'

# Run the application with uvicorn - prefer CHATBOT_ORCH_PORT, else PORT, else default 8003
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${CHATBOT_ORCH_PORT:-${PORT:-8003}}"]
