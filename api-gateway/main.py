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


# Knowledgebase Ingestion Routes
@app.post("/api/v1/knowledgebase/upload", response_model=FileUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    display_name: Optional[str] = Form(None)
):
    """Route to knowledgebase ingestion service for document upload."""
    
    content = await file.read()
    files = {'file': (file.filename, content, file.content_type)}
    data = {'display_name': display_name} if display_name else {}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{KNOWLEDGEBASE_INGESTION_URL}/upload",
                files=files,
                data=data,
                timeout=300.0
            )
            response.raise_for_status()
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from knowledgebase service: {e.response.text}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=e.response.json()
            )
        except httpx.TimeoutException as e:
            logger.error(f"Timeout connecting to knowledgebase service: {e}")
            raise HTTPException(
                status_code=504,
                detail="Gateway Timeout: The upload took too long to process."
            )
        except Exception as e:
            logger.error(f"Knowledgebase ingestion error: {repr(e)}")
            # Use repr() to get the exception type and message, avoiding empty strings
            raise HTTPException(
                status_code=502,
                detail=f"Service error: {repr(e)}"
            )


@app.get("/api/v1/knowledgebase/files", response_model=ListFilesResponse)
async def list_files():
    """Route to list uploaded files."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{KNOWLEDGEBASE_INGESTION_URL}/files",
                timeout=30.0
            )
            response.raise_for_status()
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from knowledgebase service: {e.response.text}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=e.response.json()
            )
        except Exception as e:
            logger.error(f"Knowledgebase ingestion error: {e}")
            raise HTTPException(status_code=502, detail=f"Service error: {str(e)}")


# Website Scraping Routes
@app.post("/api/v1/scrape", response_model=ScrapeResponse)
async def scrape_website(data: ScrapeRequest):
    """Route to website scraping service."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{WEBSITE_SCRAPING_URL}/scrape",
                json=data.dict(),
                timeout=120.0
            )
            response.raise_for_status()
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from scraping service: {e.response.text}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=e.response.json()
            )
        except Exception as e:
            logger.error(f"Website scraping error: {e}")
            raise HTTPException(status_code=502, detail=f"Service error: {str(e)}")


# Chatbot Routes
@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(data: ChatRequest):
    """Route to chatbot orchestration service."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{CHATBOT_ORCHESTRATION_URL}/chat",
                json=data.dict(),
                timeout=60.0
            )
            response.raise_for_status()
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from chatbot service: {e.response.text}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=e.response.json()
            )
        except Exception as e:
            logger.error(f"Chatbot orchestration error: {e}")
            raise HTTPException(status_code=502, detail=f"Service error: {str(e)}")


@app.get("/api/v1/chat/sessions", response_model=ListSessionsResponse)
async def list_sessions():
    """Route to list active chat sessions."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{CHATBOT_ORCHESTRATION_URL}/sessions",
                timeout=30.0
            )
            response.raise_for_status()
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from chatbot service: {e.response.text}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=e.response.json()
            )
        except Exception as e:
            logger.error(f"Chatbot orchestration error: {e}")
            raise HTTPException(status_code=502, detail=f"Service error: {str(e)}")


@app.delete("/api/v1/chat/sessions/{session_id}", response_model=DeleteSessionResponse)
async def delete_session(session_id: str):
    """Route to delete a chat session."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.delete(
                f"{CHATBOT_ORCHESTRATION_URL}/sessions/{session_id}",
                timeout=30.0
            )
            response.raise_for_status()
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from chatbot service: {e.response.text}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=e.response.json()
            )
        except Exception as e:
            logger.error(f"Chatbot orchestration error: {e}")
            raise HTTPException(status_code=502, detail=f"Service error: {str(e)}")


# Human-in-the-Loop Routes
@app.post("/api/v1/chat/{session_id}/review", response_model=ReviewResponse)
async def review_response(session_id: str, data: ReviewRequest):
    """Route for human-in-the-loop review."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{CHATBOT_ORCHESTRATION_URL}/sessions/{session_id}/review",
                json=data.dict(),
                timeout=30.0
            )
            response.raise_for_status()
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from chatbot service: {e.response.text}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=e.response.json()
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
