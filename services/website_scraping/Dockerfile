FROM mcr.microsoft.com/playwright/python:v1.56.0-noble

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
COPY services/website_scraping/ ./services/website_scraping/
COPY shared ./shared/

# Set working directory
WORKDIR /app/services/website_scraping

# Expose port
ARG PORT=8002
EXPOSE ${PORT}

# Add health check using PORT environment variable
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD sh -c 'curl -f http://localhost:${PORT:-8002}/health || exit 1'

# Run the application with uvicorn - use PORT env var from Railway
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8002}
