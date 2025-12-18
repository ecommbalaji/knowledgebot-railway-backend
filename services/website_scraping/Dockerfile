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



# Add health check using PORT environment variable
# More forgiving start-period to account for cold starts and heavy startup tasks
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
    CMD sh -c 'curl -f http://localhost:${WEBSITE_SCRAPING_PORT:-8002}/health || exit 1'

# Run the application with uvicorn - prefer WEBSITE_SCRAPING_PORT, else PORT, else default
CMD sh -c 'uvicorn main:app --host 0.0.0.0 --port ${WEBSITE_SCRAPING_PORT:-8002}'
