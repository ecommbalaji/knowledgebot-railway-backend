FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for web scraping
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY services/website_scraping/ ./services/website_scraping/
COPY shared ./shared/

# Set working directory
WORKDIR /app/services/website_scraping

# Set Chromium path
ENV CHROMIUM_PATH=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

# Expose port
EXPOSE 8002

# Run the application
CMD ["python", "main.py"]
