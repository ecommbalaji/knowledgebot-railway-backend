"""API Gateway - Central entry point for all requests."""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import logging
from typing import Optional, Dict, Any
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Knowledge Bot API Gateway", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service URLs from environment
KNOWLEDGEBASE_INGESTION_URL = os.getenv(
    "KNOWLEDGEBASE_INGESTION_URL", "http://localhost:8001"
)
WEBSITE_SCRAPING_URL = os.getenv(
    "WEBSITE_SCRAPING_URL", "http://localhost:8002"
)
CHATBOT_ORCHESTRATION_URL = os.getenv(
    "CHATBOT_ORCHESTRATION_URL", "http://localhost:8003"
)


class HealthResponse(BaseModel):
    status: str
    services: Dict[str, str]


@app.get("/health")
async def health_check():
    """Health check endpoint - returns gateway status only."""
    return {"status": "healthy", "service": "api-gateway"}


@app.get("/status")
async def system_status():
    """Check connections to all downstream services."""
    services = {
        "knowledgebase": KNOWLEDGEBASE_INGESTION_URL,
        "website_scraping": WEBSITE_SCRAPING_URL,
        "chatbot": CHATBOT_ORCHESTRATION_URL
    }
    statuses = {"gateway": "online"}
    
    async with httpx.AsyncClient() as client:
        for name, url in services.items():
            try:
                # Try to hit the health endpoint of the downstream service
                resp = await client.get(f"{url}/health", timeout=5.0)
                if resp.status_code == 200:
                    statuses[name] = "online" 
                else:
                    statuses[name] = f"error: {resp.status_code}"
            except Exception as e:
                statuses[name] = f"unreachable: {str(e)}"
    
    return statuses


# Knowledgebase Ingestion Routes
@app.post("/api/v1/knowledgebase/upload")
async def upload_document(request: Request):
    """Route to knowledgebase ingestion service for document upload."""
    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{KNOWLEDGEBASE_INGESTION_URL}/upload",
                content=body,
                headers=headers,
                timeout=60.0
            )
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code
            )
        except Exception as e:
            logger.error(f"Knowledgebase ingestion error: {e}")
            raise HTTPException(status_code=502, detail=f"Service error: {str(e)}")


@app.get("/api/v1/knowledgebase/files")
async def list_files():
    """Route to list uploaded files."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{KNOWLEDGEBASE_INGESTION_URL}/files",
                timeout=30.0
            )
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code
            )
        except Exception as e:
            logger.error(f"Knowledgebase ingestion error: {e}")
            raise HTTPException(status_code=502, detail=f"Service error: {str(e)}")


# Website Scraping Routes
@app.post("/api/v1/scrape")
async def scrape_website(data: Dict[str, Any]):
    """Route to website scraping service."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{WEBSITE_SCRAPING_URL}/scrape",
                json=data,
                timeout=120.0
            )
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code
            )
        except Exception as e:
            logger.error(f"Website scraping error: {e}")
            raise HTTPException(status_code=502, detail=f"Service error: {str(e)}")


# Chatbot Routes
@app.post("/api/v1/chat")
async def chat(data: Dict[str, Any]):
    """Route to chatbot orchestration service."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{CHATBOT_ORCHESTRATION_URL}/chat",
                json=data,
                timeout=60.0
            )
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code
            )
        except Exception as e:
            logger.error(f"Chatbot orchestration error: {e}")
            raise HTTPException(status_code=502, detail=f"Service error: {str(e)}")


@app.get("/api/v1/chat/sessions")
async def list_sessions():
    """Route to list active chat sessions."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{CHATBOT_ORCHESTRATION_URL}/sessions",
                timeout=30.0
            )
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code
            )
        except Exception as e:
            logger.error(f"Chatbot orchestration error: {e}")
            raise HTTPException(status_code=502, detail=f"Service error: {str(e)}")


@app.delete("/api/v1/chat/sessions/{session_id}")
async def delete_session(session_id: str):
    """Route to delete a chat session."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.delete(
                f"{CHATBOT_ORCHESTRATION_URL}/sessions/{session_id}",
                timeout=30.0
            )
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code
            )
        except Exception as e:
            logger.error(f"Chatbot orchestration error: {e}")
            raise HTTPException(status_code=502, detail=f"Service error: {str(e)}")


# Human-in-the-Loop Routes
@app.post("/api/v1/chat/{session_id}/review")
async def review_response(session_id: str, data: Dict[str, Any]):
    """Route for human-in-the-loop review."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{CHATBOT_ORCHESTRATION_URL}/sessions/{session_id}/review",
                json=data,
                timeout=30.0
            )
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code
            )
        except Exception as e:
            logger.error(f"Human-in-the-loop error: {e}")
            raise HTTPException(status_code=502, detail=f"Service error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Railway sets PORT, fallback to API_GATEWAY_PORT or 8000
    port = int(os.getenv("PORT", os.getenv("API_GATEWAY_PORT", "8000")))
    host = os.getenv("API_GATEWAY_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
