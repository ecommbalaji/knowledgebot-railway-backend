FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN find . -maxdepth 2 -not -path '*/.*'
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api-gateway/ ./api-gateway/
COPY shared ./shared/

# Set working directory to api_gateway
WORKDIR /app/api-gateway

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
