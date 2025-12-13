FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api_gateway/ ./api_gateway/
COPY shared/ ./shared/

# Set working directory to api_gateway
WORKDIR /app/api_gateway

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
