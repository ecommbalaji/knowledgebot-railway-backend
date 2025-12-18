"""API Gateway - Central entry point for all requests."""
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import logging
from typing import Optional, Dict, Any, List
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

# Pydantic Models for API Gateway
# Based on the README and downstream service expectations

# Knowledgebase Models
class FileUploadResponse(BaseModel):
    message: str
    file_id: str
    filename: str

class FileMetadata(BaseModel):
    id: str
    filename: str
    display_name: str
    mime_type: str
    size_bytes: int
    create_time: str

class ListFilesResponse(BaseModel):
    files: List[FileMetadata]

# Website Scraping Models
class ScrapeRequest(BaseModel):
    url: str
    max_depth: int = 1
    max_pages: int = 10

class ScrapeResponse(BaseModel):
    message: str
    total_pages_scraped: int
    total_files_uploaded: int

# Chatbot Models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    use_rag: bool = True
    max_results: int = 5

class ChatResponse(BaseModel):
    session_id: str
    response: str

class SessionSummary(BaseModel):
    session_id: str
    start_time: str
    message_count: int

class ListSessionsResponse(BaseModel):
    sessions: List[SessionSummary]

class DeleteSessionResponse(BaseModel):
    message: str
    session_id: str

class ReviewRequest(BaseModel):
    approved: bool
    feedback: Optional[str] = None
    corrected_answer: Optional[str] = None

class ReviewResponse(BaseModel):
    message: str
    review_status: str


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


# API Gateway Routing Endpoints

@app.post("/api/v1/chat")
async def chat_endpoint(request: Request):
    """Route chat requests to chatbot orchestration service."""
    try:
        # Get the request body
        body = await request.body()
        headers = dict(request.headers)

        # Remove hop-by-hop headers that shouldn't be forwarded
        hop_by_hop_headers = [
            'connection', 'keep-alive', 'proxy-authenticate',
            'proxy-authorization', 'te', 'trailers', 'transfer-encoding', 'upgrade'
        ]
        headers = {k: v for k, v in headers.items() if k.lower() not in hop_by_hop_headers}

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{CHATBOT_ORCHESTRATION_URL}/chat",
                content=body,
                headers=headers,
                timeout=30.0
            )
            return JSONResponse(
                status_code=resp.status_code,
                content=resp.json() if resp.headers.get('content-type', '').startswith('application/json') else resp.text
            )
    except Exception as e:
        logger.error(f"Error routing chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Chat service error: {str(e)}")


@app.post("/api/v1/knowledgebase/upload")
async def knowledgebase_upload_endpoint(request: Request):
    """Route knowledgebase upload requests to knowledgebase ingestion service."""
    try:
        # Get the request body (multipart form data)
        body = await request.body()
        headers = dict(request.headers)

        # Remove hop-by-hop headers
        hop_by_hop_headers = [
            'connection', 'keep-alive', 'proxy-authenticate',
            'proxy-authorization', 'te', 'trailers', 'transfer-encoding', 'upgrade'
        ]
        headers = {k: v for k, v in headers.items() if k.lower() not in hop_by_hop_headers}

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{KNOWLEDGEBASE_INGESTION_URL}/upload",
                content=body,
                headers=headers,
                timeout=60.0  # Longer timeout for file uploads
            )
            return JSONResponse(
                status_code=resp.status_code,
                content=resp.json() if resp.headers.get('content-type', '').startswith('application/json') else resp.text
            )
    except Exception as e:
        logger.error(f"Error routing knowledgebase upload request: {e}")
        raise HTTPException(status_code=500, detail=f"Knowledgebase service error: {str(e)}")


@app.get("/api/v1/knowledgebase/files")
async def knowledgebase_files_endpoint(request: Request):
    """Route knowledgebase files list requests to knowledgebase ingestion service."""
    try:
        # Get query parameters
        query_params = str(request.url.query)
        url = f"{KNOWLEDGEBASE_INGESTION_URL}/files"
        if query_params:
            url += f"?{query_params}"

        headers = dict(request.headers)
        # Remove hop-by-hop headers
        hop_by_hop_headers = [
            'connection', 'keep-alive', 'proxy-authenticate',
            'proxy-authorization', 'te', 'trailers', 'transfer-encoding', 'upgrade'
        ]
        headers = {k: v for k, v in headers.items() if k.lower() not in hop_by_hop_headers}

        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=headers, timeout=30.0)
            return JSONResponse(
                status_code=resp.status_code,
                content=resp.json() if resp.headers.get('content-type', '').startswith('application/json') else resp.text
            )
    except Exception as e:
        logger.error(f"Error routing knowledgebase files request: {e}")
        raise HTTPException(status_code=500, detail=f"Knowledgebase service error: {str(e)}")


@app.post("/api/v1/scrape")
async def scrape_endpoint(request: Request):
    """Route scraping requests to website scraping service."""
    try:
        # Get the request body
        body = await request.body()
        headers = dict(request.headers)

        # Remove hop-by-hop headers
        hop_by_hop_headers = [
            'connection', 'keep-alive', 'proxy-authenticate',
            'proxy-authorization', 'te', 'trailers', 'transfer-encoding', 'upgrade'
        ]
        headers = {k: v for k, v in headers.items() if k.lower() not in hop_by_hop_headers}

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{WEBSITE_SCRAPING_URL}/scrape",
                content=body,
                headers=headers,
                timeout=60.0  # Longer timeout for scraping
            )
            return JSONResponse(
                status_code=resp.status_code,
                content=resp.json() if resp.headers.get('content-type', '').startswith('application/json') else resp.text
            )
    except Exception as e:
        logger.error(f"Error routing scrape request: {e}")
        raise HTTPException(status_code=500, detail=f"Scraping service error: {str(e)}")