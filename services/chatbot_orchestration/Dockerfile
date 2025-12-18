FROM python:3.11-slim

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
COPY services/chatbot_orchestration/ ./services/chatbot_orchestration/
COPY shared ./shared/

# Set working directory
WORKDIR /app/services/chatbot_orchestration

# Expose port
ARG PORT=8003
EXPOSE ${PORT}

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD sh -c 'curl -f http://localhost:${CHATBOT_ORCH_PORT:-${PORT:-8003}}/health || exit 1'

# Run the application with uvicorn - prefer CHATBOT_ORCH_PORT, else PORT, else default
CMD sh -c 'uvicorn main:app --host 0.0.0.0 --port ${CHATBOT_ORCH_PORT:-${PORT:-8003}}'
