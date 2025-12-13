FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY services/chatbot_orchestration/ ./services/chatbot_orchestration/
COPY shared/ ./shared/

# Set working directory
WORKDIR /app/services/chatbot_orchestration

# Expose port
EXPOSE 8003

# Run the application
CMD ["python", "main.py"]
