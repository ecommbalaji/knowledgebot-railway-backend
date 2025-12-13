"""Shared configuration settings for all services."""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    gemini_api_key: str
    openai_api_key: str
    
    # Service URLs
    knowledgebase_ingestion_url: str = "http://localhost:8001"
    website_scraping_url: str = "http://localhost:8002"
    chatbot_orchestration_url: str = "http://localhost:8003"
    
    # API Gateway
    api_gateway_port: int = 8000
    api_gateway_host: str = "0.0.0.0"
    
    # Chatbot Configuration
    chatbot_model: str = "gpt-4o"
    chatbot_temperature: float = 0.7
    chatbot_max_tokens: int = 2000
    
    # Human-in-the-Loop
    human_in_the_loop_enabled: bool = False
    human_in_the_loop_webhook_url: Optional[str] = None
    
    # Gemini FileSearch
    gemini_filesearch_project_id: Optional[str] = None
    gemini_filesearch_location: str = "us-central1"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
