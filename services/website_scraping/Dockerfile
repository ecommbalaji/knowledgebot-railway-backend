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
EXPOSE 8002

# Run the application with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
