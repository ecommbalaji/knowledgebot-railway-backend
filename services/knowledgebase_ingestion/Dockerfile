FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY services/knowledgebase_ingestion/ ./services/knowledgebase_ingestion/
COPY shared ./shared/

# Set working directory
WORKDIR /app/services/knowledgebase_ingestion

# Expose port
EXPOSE 8001

# Run the application
CMD ["python", "main.py"]
