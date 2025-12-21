"""Knowledgebase Ingestion Service - Handles document uploads, R2 storage, and Gemini FileSearch."""
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List, Dict, Any
from google import genai
import os
import logging
from dotenv import load_dotenv
import json
import asyncio
import hashlib
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Add shared directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.config import settings
from shared import db
from shared.r2_storage import R2Storage

load_dotenv()

# Configure logging for containerized environments (Railway, Docker)
# Force output to stdout/stderr for proper log aggregation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Explicitly use stdout
    ],
    force=True  # Override any existing configuration
)
logger = logging.getLogger(__name__)

# Log startup diagnostics
logger.info("="*60)
logger.info("KNOWLEDGEBASE INGESTION SERVICE STARTING UP")
logger.info("="*60)
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"Environment: {os.getenv('RAILWAY_ENVIRONMENT', 'development')}")

# Ensure shared utilities are importable and enable global exception logging
import sys
from pathlib import Path
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
except Exception:
    logger.debug("Could not adjust sys.path for shared imports")

from shared.utils import setup_global_exception_logging, register_fastapi_exception_handlers, dependency_unavailable_error, log_system_metrics, log_endpoint_request
setup_global_exception_logging("knowledgebase_ingestion")

# Validate required environment variables for this service
if not settings.gemini_api_key:
    raise dependency_unavailable_error("gemini_api_key", "Knowledgebase ingestion service requires GEMINI_API_KEY")

# Lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    try:
        # Startup - Initialize database connections
        if settings.railway_postgres_url:
            try:
                await db.init_railway_db(settings.railway_postgres_url)
                logger.info("Railway PostgreSQL database initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Railway PostgreSQL: {e}")

        logger.info("🚀 Knowledgebase ingestion service started successfully")
        logger.info("🏥 Health check endpoint: /health")
        logger.info("📤 Upload endpoint: POST /upload")
        logger.info("📄 Files endpoint: GET /files")

        yield

        # Shutdown - Close database connections
        if db.railway_db:
            await db.railway_db.disconnect()
        logger.info("🛑 Knowledgebase ingestion service shutdown complete")
    except Exception as e:
        logger.error(f"❌ Error in lifespan handler: {e}")
        raise

app = FastAPI(
    title="Knowledgebase Ingestion Service",
    version="1.0.0",
    lifespan=lifespan
)

# Register FastAPI-level exception handlers to ensure stack traces are logged
register_fastapi_exception_handlers(app, "knowledgebase_ingestion")

# Request logging middleware
import time

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
            logger.warning(f"↩️ [{request_id}] Response: {response.status_code} - Path: {request.url.path} - Duration: {duration:.3f}s")
        else:
            logger.info(f"↩️ [{request_id}] Response: {response.status_code} - Duration: {duration:.3f}s")

        return response

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"💥 [{request_id}] Request failed after {duration:.3f}s: {e}", exc_info=True)
        raise

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or settings.gemini_api_key
genai_client = None
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY environment variable not set - Gemini-dependent endpoints will be unavailable")
else:
    try:
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("✅ Gemini client initialized successfully")
    except Exception as e:
        genai_client = None
        logger.error(f"❌ Failed to initialize Gemini client: {e}")

# Initialize R2 Storage (optional)
r2_storage: Optional[R2Storage] = None
r2_config_value = settings.cloudflare_r2_url
logger.info(f"R2 config from settings: {'SET' if r2_config_value else 'NOT SET'}")

if r2_config_value:
    logger.info(f"R2 connection string: {r2_config_value[:50]}...")  # Log first 50 chars for debugging
    try:
        r2_storage = R2Storage(r2_config_value)
        logger.info("✅ R2 storage initialized successfully")
        logger.info(f"R2 bucket: {r2_storage.bucket_name}")
        logger.info(f"R2 public URL: {r2_storage.public_url or 'Not configured'}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize R2 storage: {e}")
        logger.error(f"Error type: {type(e).__name__}")
else:
    logger.info("ℹ️  R2 storage not configured (cloudflare_r2_url not set)")

class FileInfo(BaseModel):
    name: str
    display_name: str
    mime_type: str
    create_time: Optional[str] = None
    update_time: Optional[str] = None
    expiration_time: Optional[str] = None
    size_bytes: Optional[str] = None
    sha256_hash: Optional[str] = None
    uri: Optional[str] = None
    state: Optional[str] = None
    r2_url: Optional[str] = None
    r2_key: Optional[str] = None
    db_record_id: Optional[str] = None


class UploadResponse(BaseModel):
    success: bool
    file: Optional[FileInfo] = None
    message: str


class FilesResponse(BaseModel):
    files: List[FileInfo]


@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint."""
    log_endpoint_request("knowledgebase_ingestion", "health", request)
    return {"status": "healthy", "service": "knowledgebase_ingestion"}

def calculate_sha256(file_path: str) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


async def get_or_create_user(email: str) -> str:
    """Get or create a user in the database and return user ID."""
    if not db.railway_db:
        return None

    try:
        # Try to get existing user
        user = await db.railway_db.fetchrow(
            "SELECT id FROM users WHERE email = $1",
            email
        )

        if user:
            return str(user['id'])

        # Create new user
        user_id = await db.railway_db.fetchval(
            """
            INSERT INTO users (email, name, is_active)
            VALUES ($1, $2, $3)
            RETURNING id
            """,
            email,
            email.split('@')[0],  # Use email prefix as name
            True
        )

    except Exception as e:
        logger.error(f"Error getting/creating user with email {email}: {e}")
        # Try to insert system user if not exists or return default system user ID
        if settings.default_user_email and email != settings.default_user_email:
             logger.warning(f"Falling back to default user: {settings.default_user_email}")
             return await get_or_create_user(settings.default_user_email)
        return None


async def _record_api_usage(
    user_id: Optional[str],
    provider: str,
    endpoint: str,
    method: str = "POST",
    status_code: int = 200,
    req_size: int = 0,
    res_size: int = 0,
    duration_ms: int = 0,
    metadata: Dict[str, Any] = None
):
    """Record API usage to the database."""
    if not db.railway_db:
        return

    try:
        await db.railway_db.execute(
            """
            INSERT INTO api_usage (
                api_provider, api_endpoint, http_method,
                request_size_bytes, response_size_bytes, status_code,
                user_id, duration_ms, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            provider, endpoint, method,
            req_size, res_size, status_code,
            user_id, duration_ms, json.dumps(metadata or {})
        )
    except Exception as e:
        logger.warning(f"Failed to record API usage: {e}")

async def _stream_to_temp_file(file: UploadFile, original_filename: str) -> tuple[str, int]:
    """Stream an uploaded file to a temporary location."""
    import tempfile
    file_ext = os.path.splitext(original_filename)[1] or ".bin"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_path = tmp_file.name
        file_size = 0
        logger.info(f"💾 [STREAM] Starting stream of {original_filename} to {tmp_path}")
        
        while chunk := await file.read(1024 * 1024):
            tmp_file.write(chunk)
            file_size += len(chunk)
        
        logger.info(f"✅ [STREAM] Stream complete. Total size: {file_size} bytes")
        return tmp_path, file_size


async def _persist_to_r2(tmp_path: str, original_filename: str, file_display_name: str, 
                        content_type: str, email: str):
    """Upload a file to Cloudflare R2 if configured."""
    r2_result, r2_key, r2_url = None, None, None
    
    if r2_storage:
        logger.info(f"☁️ [R2] Initiating upload for {original_filename}")
        try:
            r2_result = await r2_storage.upload_file(
                file_path=tmp_path,
                content_type=content_type,
                metadata={
                    'original_filename': original_filename,
                    'display_name': file_display_name,
                    'uploaded_by': email
                }
            )
            r2_key = r2_result.get('key')
            r2_url = r2_result.get('url')
            logger.info(f"✅ [R2] Upload successful. Key: {r2_key}")
        except Exception as e:
            logger.warning(f"⚠️ [R2] Upload failed, but continuing: {e}")
    else:
        logger.info("ℹ️ [R2] Skipping - Storage not configured")
        
    return r2_result, r2_key, r2_url


async def _process_with_gemini(tmp_path: str, file_display_name: str, content_type: str):
    """Upload to Gemini FileSearch and poll for processing completion."""
    logger.info(f"🤖 [GEMINI] Uploading {file_display_name} to FileSearch...")
    uploaded_file = genai_client.files.upload(
        file=tmp_path,
        config=dict(
            display_name=file_display_name,
            mime_type=content_type
        )
    )
    
    logger.info(f"✅ [GEMINI] Upload complete. File ID: {uploaded_file.name}, State: {uploaded_file.state.name}")
    logger.info(f"🔗 [GEMINI] URI: {getattr(uploaded_file, 'uri', 'N/A')}")
    
    final_state = uploaded_file.state.name
    gemini_processed_at = None
    
    try:
        for i in range(15):  # Poll for up to 30 seconds
            current_file = genai_client.files.get(name=uploaded_file.name)
            final_state = current_file.state.name
            logger.info(f"🔄 [GEMINI] Polling state (Attempt {i+1}/15): {final_state}")
            
            if final_state == "ACTIVE":
                from datetime import datetime
                gemini_processed_at = datetime.utcnow()
                logger.info("⚡ [GEMINI] Processing complete - File is now ACTIVE")
                break
            elif final_state == "FAILED":
                logger.error(f"❌ [GEMINI] Processing FAILED for {uploaded_file.name}")
                break
                
            await asyncio.sleep(2)
    except Exception as e:
        logger.warning(f"⚠️ [GEMINI] Error during polling: {e}")
        
    return uploaded_file, final_state, gemini_processed_at


async def _record_metadata(user_id: str, original_filename: str, file_display_name: str, 
                         file_ext: str, r2_url: str, r2_key: str, uploaded_file: Any, 
                         file_size: int, sha256_hash: str, r2_result: Any, 
                         final_state: str, gemini_processed_at: Any):
    """Persist file metadata and metrics to the PostgreSQL database."""
    db_record_id = None
    if not db.railway_db:
        logger.warning("⚠️ [DB] Database unavailable - Skipping metadata record")
        return None

    try:
        logger.info(f"🗄️ [DB] Saving metadata for {original_filename}")
        db_record_id = await db.railway_db.fetchval(
            """
            INSERT INTO file_uploads (
                user_id, original_filename, display_name, file_extension,
                cloudflare_r2_url, cloudflare_r2_key, gemini_file_name, gemini_file_uri,
                mime_type, size_bytes, sha256_hash,
                r2_upload_status, gemini_upload_status, gemini_state,
                gemini_processed_at, expires_at, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
            RETURNING id
            """,
            user_id,
            original_filename,
            file_display_name,
            file_ext.lstrip('.'),
            r2_url,
            r2_key,
            uploaded_file.name,
            getattr(uploaded_file, 'uri', None),
            uploaded_file.mime_type or "application/octet-stream",
            file_size,
            sha256_hash,
            'completed' if r2_result else 'skipped',
            final_state.lower(),
            final_state,
            gemini_processed_at,
            uploaded_file.expiration_time if hasattr(uploaded_file, 'expiration_time') else None,
            json.dumps({'r2_uploaded': r2_result is not None, 'gemini_file_id': uploaded_file.name})
        )
        logger.info(f"✅ [DB] Record created with ID: {db_record_id}")
        
        # Log metric
        await db.railway_db.execute(
            """
            INSERT INTO metrics (metric_type, metric_name, value, unit, user_id, file_upload_id, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            'file_upload', 'file_size_bytes', file_size, 'bytes', 
            user_id, db_record_id, json.dumps({'filename': original_filename})
        )
        logger.info(f"✅ [DB] Metric 'file_size_bytes' recorded for file {db_record_id}")
    except Exception as e:
        logger.error(f"❌ [DB] Error recording metadata (metrics might be missing): {e}")
        import traceback
        logger.error(traceback.format_exc())
        
    return db_record_id


import time
import logging
from fastapi import HTTPException, status

# Use a specific logger for this module
logger = logging.getLogger(__name__)

@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    display_name: Optional[str] = Form(None),
    user_email: Optional[str] = Header(None, alias="X-User-Email")
):
    """
    Modularized Document Ingestion Pipeline with structured logging and performance tracking.
    """
    start_time = time.perf_counter()
    
    # Configuration Check
    if not genai_client:
        logger.critical("Upload failed: Gemini client not configured in environment.")
        from shared.utils import dependency_unavailable_error
        raise dependency_unavailable_error("gemini", "client not configured")

    # Initial State
    tmp_path = None
    original_filename = file.filename or "unknown_file"
    file_display_name = display_name or original_filename
    email = user_email or settings.default_user_email
    
    # Create a contextual prefix for logs to track this specific request
    log_context = {"upload_file_name": original_filename, "user_email": email}
    logger.info(f"Initiating upload pipeline for {original_filename}", extra=log_context)

    try:
        # Step 0: User Setup
        user_id = None
        if db.railway_db:
            user_id = await get_or_create_user(email)
            log_context["user_id"] = user_id
            logger.debug(f"User resolved to ID: {user_id}", extra=log_context)
        else:
            logger.warning("RAILWAY_DB is None. Checking config...", extra=log_context)
            url_configured = bool(settings.railway_postgres_url)
            logger.warning(f"Is RAILWAY_POSTGRES_URL configured in settings? {url_configured}", extra=log_context)
            if url_configured:
                masked_url = settings.railway_postgres_url.split('@')[-1] if '@' in settings.railway_postgres_url else "..."
                logger.warning(f"Configured URL host: {masked_url}", extra=log_context)

        # Step 1: Stream to disk
        logger.info(f"Streaming {original_filename} to local temp storage...", extra=log_context)
        tmp_path, file_size = await _stream_to_temp_file(file, original_filename)
        sha256_hash = calculate_sha256(tmp_path)
        log_context["sha256"] = sha256_hash
        logger.info(f"File streamed. Size: {file_size} bytes, Hash: {sha256_hash}", extra=log_context)

        # Step 2: Persist to R2 (Cloud Storage)
        logger.info("Persisting file to Cloudflare R2...", extra=log_context)
        r2_result, r2_key, r2_url = await _persist_to_r2(
            tmp_path, original_filename, file_display_name, 
            file.content_type or "application/octet-stream", email
        )
        logger.info(f"R2 Persistence complete. Key: {r2_key}", extra=log_context)
        
        # Track R2 Usage
        if r2_storage:
             await _record_api_usage(
                user_id, "cloudflare_r2", "put_object", "PUT", 200, 
                file_size, 0, 0, {"key": r2_key}
             )

        # Step 3: Process with Gemini (AI Processing)
        logger.info("Sending file to Gemini FileSearch API...", extra=log_context)
        uploaded_file, final_state, gemini_processed_at = await _process_with_gemini(
            tmp_path, file_display_name, file.content_type or "application/octet-stream"
        )
        logger.info(f"Gemini processing finished. State: {final_state}", extra=log_context)
        
        # Track Gemini Usage
        await _record_api_usage(
            user_id, "gemini", "files.upload", "POST", 200,
            file_size, 0, 0, {"file_name": uploaded_file.name}
        )

        # Step 4: Record Metadata (PostgreSQL)
        logger.info("Recording final metadata to PostgreSQL...", extra=log_context)
        file_ext = os.path.splitext(original_filename)[1] or ".bin"
        db_record_id = await _record_metadata(
            user_id, original_filename, file_display_name, file_ext,
            r2_url, r2_key, uploaded_file, file_size, sha256_hash,
            r2_result, final_state, gemini_processed_at
        )

        # Step 5: Finalize Response
        duration = time.perf_counter() - start_time
        logger.info(
            f"Pipeline successful for {original_filename} in {duration:.2f}s",
            extra={**log_context, "duration_sec": duration}
        )

        # Track Internal API Usage
        await _record_api_usage(
            user_id, "internal", "/upload", "POST", 200,
            file_size, 0, int(duration * 1000), {"filename": original_filename}
        )

        return UploadResponse(
            success=True,
            file=FileInfo(
                name=uploaded_file.name,
                display_name=uploaded_file.display_name,
                mime_type=uploaded_file.mime_type,
                create_time=uploaded_file.create_time.isoformat() if uploaded_file.create_time else None,
                update_time=uploaded_file.update_time.isoformat() if uploaded_file.update_time else None,
                expiration_time=uploaded_file.expiration_time.isoformat() if uploaded_file.expiration_time else None,
                size_bytes=str(uploaded_file.size_bytes or file_size),
                sha256_hash=sha256_hash,
                uri=getattr(uploaded_file, 'uri', None),
                state=final_state,
                r2_url=r2_url,
                r2_key=r2_key,
                db_record_id=str(db_record_id) if db_record_id else None
            ),
            message=f"File processed successfully: {file_display_name}"
        )

    except ValueError as ve:
        # Catch validation/input errors separately
        logger.warning(f"Validation error during upload: {ve}", extra=log_context)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))

    except Exception as e:
        # Log the full stack trace for unexpected system errors
        logger.error(
            f"Critical failure in ingestion pipeline: {type(e).__name__} - {e}", 
            exc_info=True, 
            extra=log_context
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="An internal error occurred during document processing."
        )
        
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.debug(f"Temporary file deleted: {tmp_path}", extra=log_context)
            except Exception as cleanup_err:
                logger.warning(f"Failed to delete temp file {tmp_path}: {cleanup_err}", extra=log_context)

@app.get("/files", response_model=FilesResponse)
async def list_files():
    """List all uploaded files in Gemini FileSearch."""
    if not genai_client:
        from shared.utils import dependency_unavailable_error
        raise dependency_unavailable_error("gemini", "client not configured")

    try:
        files = genai_client.files.list()

        file_list = []
        for file in files:
            file_info = FileInfo(
                name=file.name,
                display_name=file.display_name,
                mime_type=file.mime_type,
                create_time=file.create_time.isoformat() if file.create_time else None,
                update_time=file.update_time.isoformat() if file.update_time else None,
                expiration_time=file.expiration_time.isoformat() if file.expiration_time else None,
                size_bytes=str(file.size_bytes) if file.size_bytes else None,
                sha256_hash=file.sha256_hash if hasattr(file, 'sha256_hash') else None,
                uri=file.uri if hasattr(file, 'uri') else None,
                state=file.state.name if hasattr(file, 'state') else None,
            )
            file_list.append(file_info)

        return FilesResponse(files=file_list)
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")

@app.get("/files/metadata")
async def list_files_metadata(
    include_signed_urls: bool = False,
    signed_url_expiration: int = 3600,
    user_email: Optional[str] = Header(None, alias="X-User-Email")
):
    """
    List all uploaded files with metadata from database.

    Args:
        include_signed_urls: Whether to include signed URLs for private R2 files
        signed_url_expiration: Expiration time for signed URLs in seconds (default 1 hour)
        user_email: User email for access tracking
    """
    if not railway_db:
        from shared.utils import dependency_unavailable_error
        raise dependency_unavailable_error("database", "database not configured")

    try:
        files = await db.railway_db.fetch(
            """
            SELECT id, original_filename, display_name, file_extension, mime_type,
                   size_bytes, created_at, cloudflare_r2_url, cloudflare_r2_key, gemini_file_name
            FROM file_uploads
            ORDER BY created_at DESC
            """
        )

        file_list = []
        for f in files:
            file_info = {
                "id": str(f["id"]),
                "original_filename": f["original_filename"],
                "display_name": f["display_name"],
                "file_extension": f["file_extension"],
                "mime_type": f["mime_type"],
                "size_bytes": f["size_bytes"],
                "created_at": f["created_at"].isoformat() if f["created_at"] else None,
                "cloudflare_r2_url": f["cloudflare_r2_url"],
                "cloudflare_r2_key": f["cloudflare_r2_key"],
                "gemini_file_name": f["gemini_file_name"],
                "r2_access_type": "public" if f["cloudflare_r2_url"] else "private"
            }

            # Add signed URL for private R2 files if requested
            if include_signed_urls and f["cloudflare_r2_key"] and not f["cloudflare_r2_url"]:
                try:
                    signed_url = r2_storage.generate_signed_url(
                        f["cloudflare_r2_key"],
                        expiration=signed_url_expiration
                    )
                    file_info["signed_url"] = signed_url
                    file_info["signed_url_expires_in"] = signed_url_expiration
                except Exception as e:
                    logger.warning(f"Failed to generate signed URL for file {f['id']}: {e}")
                    file_info["signed_url_error"] = str(e)

            file_list.append(file_info)

        return {
            "files": file_list,
            "total_count": len(file_list),
            "signed_urls_included": include_signed_urls,
            "signed_url_expiration": signed_url_expiration if include_signed_urls else None
        }
    except Exception as e:
        logger.error(f"Error listing files metadata: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")

@app.get("/status")
async def get_service_status():
    """Get service status and configuration information."""
    r2_info = None
    if r2_storage:
        r2_info = {
            "bucket": r2_storage.bucket_name,
            "public_url": r2_storage.public_url,
            "is_private": r2_storage.public_url is None,
            "endpoint_url": r2_storage.s3_client.meta.endpoint_url
        }

    return {
        "service": "knowledgebase_ingestion",
        "status": "healthy",
        "r2_configured": r2_storage is not None,
        "r2_info": r2_info,
        "gemini_configured": genai_client is not None,
        "database_configured": db.railway_db is not None,
        "version": "1.0.0"
    }

@app.get("/files/{file_id}/signed-url")
async def get_file_signed_url(
    file_id: str,
    expiration: int = 3600,  # Default 1 hour
    user_email: Optional[str] = Header(None, alias="X-User-Email")
):
    """
    Generate a signed URL for accessing a private R2 file.

    Args:
        file_id: The file ID (UUID) or R2 key
        expiration: URL expiration time in seconds (default 1 hour, max 24 hours)
        user_email: User email for access tracking

    Returns:
        Signed URL for file access
    """
    if not r2_storage:
        from shared.utils import dependency_unavailable_error
        raise dependency_unavailable_error("r2", "R2 storage not configured")

    # Validate expiration time (max 24 hours for security)
    if expiration > 86400:  # 24 hours
        raise HTTPException(status_code=400, detail="Expiration time cannot exceed 24 hours")

    if expiration < 60:  # 1 minute minimum
        raise HTTPException(status_code=400, detail="Expiration time must be at least 60 seconds")

    try:
        # First try to find by file ID (UUID from database)
        r2_key = None
        if db.railway_db:
            # Try to find the file by ID
            file_record = await db.railway_db.fetchrow(
                "SELECT cloudflare_r2_key, original_filename FROM file_uploads WHERE id = $1",
                file_id
            )
            if file_record and file_record['cloudflare_r2_key']:
                r2_key = file_record['cloudflare_r2_key']
                logger.info(f"Found R2 key for file ID {file_id}: {r2_key}")
            else:
                # If not found by ID, assume the file_id is actually the R2 key
                r2_key = file_id
                logger.info(f"Using provided file_id as R2 key: {r2_key}")
        else:
            # No database, assume file_id is the R2 key
            r2_key = file_id

        # Generate signed URL
        signed_url = r2_storage.generate_signed_url(r2_key, expiration=expiration)

        logger.info(f"Generated signed URL for file {r2_key}, expires in {expiration} seconds")

        return {
            "signed_url": signed_url,
            "expires_in_seconds": expiration,
            "file_key": r2_key,
            "is_private": True
        }

    except Exception as e:
        logger.error(f"Failed to generate signed URL for file {file_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate signed URL: {str(e)}")

@app.get("/files/{file_id}/download")
async def download_file(
    file_id: str,
    expiration: int = 3600,
    user_email: Optional[str] = Header(None, alias="X-User-Email")
):
    """
    Download a file from private R2 storage using a signed URL redirect.

    This endpoint redirects to a signed URL for immediate download.
    """
    try:
        # Get signed URL
        result = await get_file_signed_url(file_id, expiration, user_email)

        # Redirect to signed URL for download
        from fastapi.responses import RedirectResponse
        return RedirectResponse(
            url=result["signed_url"],
            status_code=302,
            headers={"Cache-Control": "private, no-cache"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download file {file_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download file: {str(e)}")


@app.delete("/files/{file_name}")
async def delete_file(file_name: str):
    """Delete a file from Gemini FileSearch, R2 storage, and database."""
    if not genai_client:
        from shared.utils import dependency_unavailable_error
        raise dependency_unavailable_error("gemini", "client not configured")

    try:
        # First, delete from Gemini FileSearch
        try:
            genai_client.files.delete(name=file_name)
            logger.info(f"Deleted file from Gemini FileSearch: {file_name}")
        except Exception as e:
            logger.warning(f"Failed to delete from Gemini FileSearch: {e}")

        # Delete from Cloudflare R2 if configured and we have the key
        if r2_storage and db.railway_db:
            try:
                # Get R2 key from database
                file_record = await db.railway_db.fetchrow(
                    "SELECT cloudflare_r2_key FROM file_uploads WHERE gemini_file_name = $1",
                    file_name
                )

                if file_record and file_record['cloudflare_r2_key']:
                    await r2_storage.delete_file(file_record['cloudflare_r2_key'])
                    logger.info(f"Deleted file from R2 storage: {file_record['cloudflare_r2_key']}")
            except Exception as e:
                logger.warning(f"Failed to delete from R2 storage: {e}")

        # Delete from database (both file_uploads and potentially scraped_websites)
        if db.railway_db:
            try:
                # Delete from file_uploads
                deleted_uploads = await db.railway_db.execute(
                    "DELETE FROM file_uploads WHERE gemini_file_name = $1",
                    file_name
                )

                # Also check scraped_websites in case it's a scraped website
                deleted_scraped = await db.railway_db.execute(
                    "DELETE FROM scraped_websites WHERE gemini_file_name = $1",
                    file_name
                )

                logger.info(f"Deleted from database: {deleted_uploads} file uploads, {deleted_scraped} scraped websites")
            except Exception as e:
                logger.warning(f"Failed to delete from database: {e}")

        return {"success": True, "message": f"File {file_name} deleted from all storage layers"}

    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Port selection order: Service-specific -> Railway PORT -> Default 8001
    port = int(os.getenv("KB_INGESTION_PORT", os.getenv("PORT", "8001")))
    logger.info(f"🚀 Starting knowledgebase_ingestion service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
