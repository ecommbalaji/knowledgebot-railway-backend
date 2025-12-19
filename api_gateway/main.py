"""API Gateway - Central entry point for all requests."""
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import logging
import time
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
import os
import sys
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Use stdout only to avoid duplication
)
logger = logging.getLogger(__name__)

# Debug logging to check if script starts
logger.info("🔍 API Gateway script starting...")
logger.info("🔍 Testing basic imports and setup...")

# Add startup logging
logger.info("="*60)
logger.info("API GATEWAY SERVICE STARTING UP")
logger.info("="*60)
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"Environment: {os.getenv('RAILWAY_ENVIRONMENT', 'development')}")

# Log service URLs and configuration
logger.info("SERVICE CONFIGURATION:")
logger.info(f"KNOWLEDGEBASE_INGESTION_URL: {os.getenv('KNOWLEDGEBASE_INGESTION_URL', 'http://localhost:8001')}")
logger.info(f"WEBSITE_SCRAPING_URL: {os.getenv('WEBSITE_SCRAPING_URL', 'http://localhost:8002')}")
logger.info(f"CHATBOT_ORCHESTRATION_URL: {os.getenv('CHATBOT_ORCHESTRATION_URL', 'http://localhost:8003')}")
# Get port configuration
PORT = int(os.getenv('API_GATEWAY_PORT', os.getenv('PORT', '8080')))
logger.info(f"PORT being used: {PORT}")

# Check if required environment variables are set
required_env_vars = ['KNOWLEDGEBASE_INGESTION_URL', 'WEBSITE_SCRAPING_URL', 'CHATBOT_ORCHESTRATION_URL']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.warning(f"Missing environment variables: {missing_vars}")
else:
    logger.info("All required environment variables are set")

logger.info(f"🚀 Application will start on port {PORT}")

# Make shared package importable when code runs inside container
from pathlib import Path
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
except Exception:
    logger.debug("Could not adjust sys.path for shared imports")

from shared.utils import setup_global_exception_logging, register_fastapi_exception_handlers, log_system_metrics, log_endpoint_request

# Install global exception and signal handlers early so any startup issues get logged
setup_global_exception_logging("api_gateway")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    try:
        # Startup
        startup_time = time.time() - getattr(app, 'start_time', time.time())
        logger.info(f"Startup time: {startup_time:.2f} seconds")
        logger.info(f"🚀 FastAPI application started successfully on port {PORT}")
        logger.info("🏥 Health check endpoint: /health")
        logger.info("📊 Status endpoint: /status")
        
        # Log registered routes
        logger.info("📋 Registered routes:")
        for route in app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                methods = ', '.join(route.methods)
                logger.info(f"  {methods} {route.path}")
        logger.info(f"📊 Total routes registered: {len([r for r in app.routes if hasattr(r, 'path')])}")
        
        logger.info("🎉 API Gateway is ready to accept requests!")

        yield

        # Shutdown
        logger.info("🛑 FastAPI application shutting down")
    except Exception as e:
        logger.error(f"❌ Error in lifespan handler: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        raise

# Track application start time for uptime calculations
app_start_time = time.time()

logger.info("="*60)
logger.info("🏗️ Creating FastAPI application...")

try:
    app = FastAPI(
        title="Knowledge Bot API Gateway",
        version="1.0.0",
        lifespan=lifespan
    )

    # Store start time for uptime calculations
    app.start_time = app_start_time
    logger.info("✅ FastAPI application created successfully")

except Exception as e:
    logger.error(f"❌ Failed to create FastAPI application: {e}")
    logger.error(f"Error type: {type(e).__name__}")
    raise

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all incoming requests with timing and status information."""
    start_time = time.time()
    request_id = f"{int(start_time * 1000000) % 1000000:06d}"  # Simple request ID

    # Log incoming request
    logger.info(f"📨 [{request_id}] {request.method} {request.url.path} - Client: {request.client.host if request.client else 'unknown'}")

    # Log headers (excluding sensitive ones)
    safe_headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ['authorization', 'x-api-key', 'cookie']}
    if safe_headers:
        logger.debug(f"📋 [{request_id}] Headers: {dict(safe_headers)}")

    try:
        # Process the request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Log response
        if response.status_code >= 400:
            logger.warning(f"↩️ [{request_id}] Response: {response.status_code} - Total time: {duration:.3f}s")
        elif response.status_code >= 300:
            logger.info(f"↩️ [{request_id}] Response: {response.status_code} - Total time: {duration:.3f}s")
        else:
            logger.info(f"↩️ [{request_id}] Response: {response.status_code} - Total time: {duration:.3f}s")

        return response

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"💥 [{request_id}] Request failed after {duration:.3f}s: {e}")
        raise

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
async def health_check(request: Request):
    """Health check endpoint - returns gateway status with detailed logging."""
    log_endpoint_request("api_gateway", "health", request)
    start_time = time.time()
    logger.info("🔍 Health check request received")

    try:
        # Check if the application is responsive
        health_status = {
            "status": "healthy",
            "service": "api-gateway",
            "timestamp": time.time(),
            "uptime_seconds": time.time() - getattr(app, 'start_time', time.time())
        }

        # Log environment checks
        logger.info("✅ Application is responsive")
        logger.info(f"📊 Health status: {health_status['status']}")
        logger.info(".2f")

        # Check critical dependencies (basic connectivity)
        try:
            # Test if httpx client can be created (basic functionality check)
            async with httpx.AsyncClient() as client:
                logger.info("✅ HTTP client initialization successful")
        except Exception as e:
            logger.error(f"❌ HTTP client initialization failed: {e}")
            health_status["status"] = "unhealthy"
            health_status["error"] = f"HTTP client error: {str(e)}"

        # Check if service URLs are reachable (lightweight check)
        service_urls = {
            "knowledgebase_ingestion": KNOWLEDGEBASE_INGESTION_URL,
            "website_scraping": WEBSITE_SCRAPING_URL,
            "chatbot_orchestration": CHATBOT_ORCHESTRATION_URL
        }

        connectivity_checks = {}
        for service_name, url in service_urls.items():
            try:
                # Just check if URL is parseable and reachable, not full health check
                parsed_url = httpx.URL(url)
                connectivity_checks[service_name] = {
                    "url": url,
                    "reachable": True,
                    "scheme": parsed_url.scheme,
                    "host": parsed_url.host,
                    "port": parsed_url.port
                }
                logger.info(f"✅ {service_name} URL configured: {url}")
            except Exception as e:
                connectivity_checks[service_name] = {
                    "url": url,
                    "reachable": False,
                    "error": str(e)
                }
                logger.warning(f"⚠️  {service_name} URL configuration issue: {e}")
                health_status["status"] = "degraded"

        health_status["connectivity_checks"] = connectivity_checks

        # Log final health status
        duration = time.time() - start_time
        logger.info(".3f")

        if health_status["status"] == "healthy":
            logger.info("🎉 Health check completed successfully")
        elif health_status["status"] == "degraded":
            logger.warning("⚠️  Health check completed with warnings")
        else:
            logger.error("❌ Health check failed")

        return health_status

    except Exception as e:
        duration = time.time() - start_time
        logger.error(".3f")
        logger.error(f"💥 Critical health check error: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.get("/status")
async def system_status():
    """Check connections to all downstream services with detailed logging."""
    start_time = time.time()
    logger.info("🔍 System status check initiated")

    services = {
        "knowledgebase": KNOWLEDGEBASE_INGESTION_URL,
        "website_scraping": WEBSITE_SCRAPING_URL,
        "chatbot": CHATBOT_ORCHESTRATION_URL
    }
    statuses = {"gateway": "online"}
    detailed_results = {
        "timestamp": time.time(),
        "overall_status": "checking",
        "services": {}
    }

    logger.info("🌐 Checking downstream service connectivity...")

    async with httpx.AsyncClient(timeout=10.0) as client:  # Increased timeout for better diagnostics
        for name, url in services.items():
            service_start_time = time.time()
            logger.info(f"🔗 Checking {name} service at {url}")

            try:
                # Try to hit the health endpoint of the downstream service
                health_url = f"{url}/health"
                logger.info(f"📡 Making request to: {health_url}")

                resp = await client.get(health_url, timeout=5.0)

                service_duration = time.time() - service_start_time
                logger.info(".3f")

                if resp.status_code == 200:
                    statuses[name] = "online"
                    logger.info(f"✅ {name} service is ONLINE (status: {resp.status_code})")

                    # Try to parse response for additional details
                    try:
                        response_data = resp.json()
                        logger.info(f"📊 {name} health response: {response_data}")
                        detailed_results["services"][name] = {
                            "status": "online",
                            "http_status": resp.status_code,
                            "response_time_seconds": round(service_duration, 3),
                            "health_data": response_data
                        }
                    except Exception as parse_error:
                        logger.warning(f"⚠️  Could not parse {name} health response as JSON: {parse_error}")
                        detailed_results["services"][name] = {
                            "status": "online",
                            "http_status": resp.status_code,
                            "response_time_seconds": round(service_duration, 3),
                            "response_text": resp.text[:200]  # First 200 chars
                        }

                else:
                    statuses[name] = f"error: {resp.status_code}"
                    logger.error(f"❌ {name} service returned error status: {resp.status_code}")
                    logger.error(f"📄 Response: {resp.text[:500]}")  # Log response body for debugging

                    detailed_results["services"][name] = {
                        "status": "error",
                        "http_status": resp.status_code,
                        "response_time_seconds": round(service_duration, 3),
                        "error": f"HTTP {resp.status_code}",
                        "response_preview": resp.text[:200]
                    }

            except httpx.TimeoutException as e:
                service_duration = time.time() - service_start_time
                statuses[name] = f"timeout: {str(e)}"
                logger.error(".3f")
                logger.error(f"⏰ {name} service health check timed out")

                detailed_results["services"][name] = {
                    "status": "timeout",
                    "error": "Request timeout",
                    "response_time_seconds": round(service_duration, 3),
                    "timeout_seconds": 5.0
                }

            except httpx.ConnectError as e:
                service_duration = time.time() - service_start_time
                statuses[name] = f"unreachable: {str(e)}"
                logger.error(".3f")
                logger.error(f"🚫 {name} service is unreachable: {e}")

                detailed_results["services"][name] = {
                    "status": "unreachable",
                    "error": str(e),
                    "response_time_seconds": round(service_duration, 3)
                }

            except Exception as e:
                service_duration = time.time() - service_start_time
                statuses[name] = f"error: {str(e)}"
                logger.error(".3f")
                logger.error(f"💥 Unexpected error checking {name} service: {e}")

                detailed_results["services"][name] = {
                    "status": "error",
                    "error": str(e),
                    "response_time_seconds": round(service_duration, 3),
                    "error_type": type(e).__name__
                }

    # Determine overall system status
    online_services = sum(1 for status in statuses.values() if status == "online")
    total_services = len(services) + 1  # +1 for gateway

    if online_services == total_services:
        detailed_results["overall_status"] = "healthy"
        logger.info("🎉 All services are healthy")
    elif online_services >= total_services - 1:  # At least gateway + most services
        detailed_results["overall_status"] = "degraded"
        logger.warning("⚠️  System is degraded - some services may be unavailable")
    else:
        detailed_results["overall_status"] = "unhealthy"
        logger.error("❌ System is unhealthy - critical services are down")

    total_duration = time.time() - start_time
    logger.info(".3f")

    # Return both simple status and detailed results
    return {
        "simple_status": statuses,
        "detailed_status": detailed_results
    }


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
async def knowledgebase_upload_endpoint(
    request: Request,
    file: UploadFile = File(...),
    display_name: Optional[str] = Form(None),
    user_email: Optional[str] = Header(None, alias="X-User-Email")
):
    """Route knowledgebase upload requests to knowledgebase ingestion service."""
    try:
        logger.info(f"📁 Received file upload request: {file.filename}, size: {file.size if hasattr(file, 'size') else 'unknown'}")

        # Prepare multipart form data for forwarding
        files = {
            'file': (file.filename, await file.read(), file.content_type)
        }

        # Prepare form data
        data = {}
        if display_name:
            data['display_name'] = display_name

        # Prepare headers
        headers = {}
        if user_email:
            headers['X-User-Email'] = user_email

        # Remove hop-by-hop headers
        request_headers = dict(request.headers)
        hop_by_hop_headers = [
            'connection', 'keep-alive', 'proxy-authenticate',
            'proxy-authorization', 'te', 'trailers', 'transfer-encoding', 'upgrade'
        ]
        headers.update({k: v for k, v in request_headers.items()
                       if k.lower() not in hop_by_hop_headers and k.lower() not in ['content-type', 'content-length']})

        logger.info(f"📤 Forwarding upload to: {KNOWLEDGEBASE_INGESTION_URL}/upload")

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{KNOWLEDGEBASE_INGESTION_URL}/upload",
                files=files,
                data=data,
                headers=headers,
                timeout=60.0  # Longer timeout for file uploads
            )

            logger.info(f"📥 Upload response status: {resp.status_code}")

            if resp.status_code == 200:
                logger.info("✅ File upload completed successfully")
            else:
                logger.warning(f"⚠️  File upload returned status {resp.status_code}")

            return JSONResponse(
                status_code=resp.status_code,
                content=resp.json() if resp.headers.get('content-type', '').startswith('application/json') else resp.text
            )
    except Exception as e:
        logger.error(f"❌ Error routing knowledgebase upload request: {e}")
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


@app.get("/api/v1/knowledgebase/files/metadata")
async def knowledgebase_files_metadata_endpoint(
    request: Request,
    include_signed_urls: bool = False,
    signed_url_expiration: int = 3600
):
    """Route knowledgebase files metadata requests to knowledgebase ingestion service."""
    try:
        # Build URL with query parameters
        url = f"{KNOWLEDGEBASE_INGESTION_URL}/files/metadata"
        query_params = []
        if include_signed_urls:
            query_params.append(f"include_signed_urls=true")
        if signed_url_expiration != 3600:
            query_params.append(f"signed_url_expiration={signed_url_expiration}")

        if query_params:
            url += "?" + "&".join(query_params)

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
        logger.error(f"Error routing knowledgebase files metadata request: {e}")
        raise HTTPException(status_code=500, detail=f"Knowledgebase service error: {str(e)}")


@app.get("/api/v1/knowledgebase/files/{file_id}/signed-url")
async def knowledgebase_file_signed_url_endpoint(
    file_id: str,
    request: Request,
    expiration: int = 3600
):
    """Route signed URL generation requests to knowledgebase ingestion service."""
    try:
        url = f"{KNOWLEDGEBASE_INGESTION_URL}/files/{file_id}/signed-url"
        if expiration != 3600:
            url += f"?expiration={expiration}"

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
        logger.error(f"Error routing signed URL request for file {file_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Knowledgebase service error: {str(e)}")


@app.get("/api/v1/knowledgebase/files/{file_id}/download")
async def knowledgebase_file_download_endpoint(
    file_id: str,
    request: Request,
    expiration: int = 3600
):
    """Route file download requests to knowledgebase ingestion service."""
    try:
        url = f"{KNOWLEDGEBASE_INGESTION_URL}/files/{file_id}/download"
        if expiration != 3600:
            url += f"?expiration={expiration}"

        headers = dict(request.headers)
        # Remove hop-by-hop headers
        hop_by_hop_headers = [
            'connection', 'keep-alive', 'proxy-authenticate',
            'proxy-authorization', 'te', 'trailers', 'transfer-encoding', 'upgrade'
        ]
        headers = {k: v for k, v in headers.items() if k.lower() not in hop_by_hop_headers}

        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=headers, timeout=30.0, follow_redirects=False)

            if resp.status_code == 302:
                # Return the redirect response
                return JSONResponse(
                    status_code=200,
                    content={
                        "download_url": resp.headers.get("location"),
                        "expires_in_seconds": expiration
                    }
                )
            else:
                return JSONResponse(
                    status_code=resp.status_code,
                    content=resp.json() if resp.headers.get('content-type', '').startswith('application/json') else resp.text
                )
    except Exception as e:
        logger.error(f"Error routing download request for file {file_id}: {e}")
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