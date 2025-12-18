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
EXPOSE 8003

# Run the application with uvicorn - use PORT env var from Railway
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8003}
