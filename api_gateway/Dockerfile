FROM python:3.12-slim

# Add metadata labels
LABEL maintainer="Knowledge Bot Team"
LABEL description="API Gateway service for Knowledge Bot"

# Set environment variables for better logging
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=random
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Log build start
RUN echo "🚀 Starting API Gateway Docker build..." && \
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
    pip list | grep -E "(fastapi|uvicorn|httpx|pydantic)"

# Copy application code with logging
COPY api_gateway/ ./api_gateway/
COPY shared ./shared/
RUN echo "📁 Application code copied" && \
    ls -la api_gateway/ && \
    echo "📄 Main application file:" && \
    head -n 10 api_gateway/main.py

# Set working directory to api_gateway
WORKDIR /app/api_gateway

# Create logs directory
RUN mkdir -p /app/logs && \
    echo "📝 Logs directory created"

# Expose port (Railway will set PORT environment variable)
ARG PORT=8080
EXPOSE ${PORT}

# Add health check (prefer API_GATEWAY_PORT, else PORT, else default)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD sh -c 'curl -f http://localhost:${API_GATEWAY_PORT:-${PORT:-8080}}/health || exit 1'

# Run the application with explicit module path and prefer API_GATEWAY_PORT, then PORT, then default
CMD sh -c 'python -m uvicorn main:app --host 0.0.0.0 --port ${API_GATEWAY_PORT:-${PORT:-8080}} --log-level info --access-log'
