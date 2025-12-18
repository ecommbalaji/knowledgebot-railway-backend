"""Knowledgebase Ingestion Service - Handles document uploads, R2 storage, and Gemini FileSearch."""
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from google import genai
import os
import logging
from dotenv import load_dotenv
import json
import asyncio
import hashlib
import sys
from pathlib import Path

# Add shared directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.config import settings
from shared.db import init_railway_db, railway_db
from shared.r2_storage import R2Storage

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Knowledgebase Ingestion Service", version="1.0.0")

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
    logger.warning("GEMINI_API_KEY environment variable not set - API endpoints will fail")
else:
    genai_client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize R2 Storage (optional)
r2_storage: Optional[R2Storage] = None
if settings.cloudflare_r2_url:
    try:
        r2_storage = R2Storage(settings.cloudflare_r2_url)
        logger.info("R2 storage initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize R2 storage: {e}")

# Initialize PostgreSQL database (optional)
@app.on_event("startup")
async def startup_event():
    """Initialize database connections on startup."""
    if settings.railway_postgres_url:
        try:
            await init_railway_db(settings.railway_postgres_url)
            logger.info("Railway PostgreSQL database initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Railway PostgreSQL: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Close database connections on shutdown."""
    if railway_db:
        await railway_db.disconnect()


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
async def health_check():
    """Health check endpoint."""
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
    if not railway_db:
        return None

    try:
        # Try to get existing user
        user = await railway_db.fetchrow(
            "SELECT id FROM users WHERE email = $1",
            email
        )

        if user:
            return str(user['id'])

        # Create new user
        user_id = await railway_db.fetchval(
            """
            INSERT INTO users (email, name, is_active)
            VALUES ($1, $2, $3)
            RETURNING id
            """,
            email,
            email.split('@')[0],  # Use email prefix as name
            True
        )

        return str(user_id) if user_id else None
    except Exception as e:
        logger.error(f"Error getting/creating user: {e}")
        return None


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    display_name: Optional[str] = None,
    user_email: Optional[str] = Header(None, alias="X-User-Email")
):
    """
    Upload a document: R2 -> Gemini -> PostgreSQL metadata.

    Args:
        file: The file to upload
        display_name: Optional display name for the file
        user_email: Optional user email from header (for tracking)

    Returns:
        UploadResponse with file information
    """
    if not genai_client:
        raise HTTPException(status_code=503, detail="Gemini API client not configured")

    r2_result = None
    db_record_id = None
    user_id = None

    try:
        # Use display name or filename
        file_display_name = display_name or file.filename or "uploaded_file"
        original_filename = file.filename or "uploaded_file"

        # Get or create user
        email = user_email or settings.default_user_email
        if railway_db:
            user_id = await get_or_create_user(email)

        # Gemini API requires file path, so write to temp file
        import tempfile
        file_ext = os.path.splitext(original_filename)[1] or ".bin"

        # Stream file to disk to avoid OOM
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            # Read in 1MB chunks
            file_size = 0
            while chunk := await file.read(1024 * 1024):
                tmp_file.write(chunk)
                file_size += len(chunk)
            tmp_path = tmp_file.name

        try:
            # Step 1: Upload to Cloudflare R2 (if configured)
            r2_key = None
            r2_url = None
            if r2_storage:
                try:
                    logger.info(f"Uploading file to R2: {original_filename}")
                    r2_result = await r2_storage.upload_file(
                        file_path=tmp_path,
                        content_type=file.content_type or "application/octet-stream",
                        metadata={
                            'original_filename': original_filename,
                            'display_name': file_display_name,
                            'uploaded_by': email
                        }
                    )
                    r2_key = r2_result['key']
                    r2_url = r2_result['url']
                    logger.info(f"File uploaded to R2: {r2_key}")
                except Exception as e:
                    logger.error(f"R2 upload failed: {e}")
                    # Continue with Gemini upload even if R2 fails

            # Step 2: Calculate file hash
            sha256_hash = calculate_sha256(tmp_path)

            # Step 3: Upload to Gemini FileSearch
            uploaded_file = genai_client.files.upload(
                path=tmp_path,
                config=dict(
                    display_name=file_display_name,
                    mime_type=file.content_type or "application/octet-stream"
                )
            )

            logger.info(f"Uploaded file to Gemini: {uploaded_file.name}, initial state: {uploaded_file.state.name}")

            # Poll for ACTIVE state
            final_state = uploaded_file.state.name
            gemini_processed_at = None
            try:
                for _ in range(15):  # Wait up to 30 seconds
                    current_file = genai_client.files.get(name=uploaded_file.name)
                    final_state = current_file.state.name
                    logger.info(f"Polling file {uploaded_file.name} state: {final_state}")

                    if final_state == "ACTIVE":
                        from datetime import datetime
                        gemini_processed_at = datetime.utcnow()
                        break
                    elif final_state == "FAILED":
                        logger.error(f"File {uploaded_file.name} failed processing")
                        break

                    await asyncio.sleep(2)
            except Exception as e:
                logger.warning(f"Error polling file state: {e}")

            # Step 4: Save metadata to PostgreSQL
            if railway_db:
                try:
                    db_record_id = await railway_db.fetchval(
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
                        file.content_type or "application/octet-stream",
                        file_size,
                        sha256_hash,
                        'completed' if r2_result else 'skipped',
                        final_state.lower(),
                        final_state,
                        gemini_processed_at,
                        uploaded_file.expiration_time if hasattr(uploaded_file, 'expiration_time') and uploaded_file.expiration_time else None,
                        json.dumps({
                            'r2_uploaded': r2_result is not None,
                            'gemini_file_id': uploaded_file.name
                        })
                    )
                    logger.info(f"File metadata saved to database: {db_record_id}")

                    # Record metric
                    await railway_db.execute(
                        """
                        INSERT INTO metrics (metric_type, metric_name, value, unit, user_id, file_upload_id, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """,
                        'file_upload',
                        'file_size_bytes',
                        file_size,
                        'bytes',
                        user_id,
                        db_record_id,
                        json.dumps({'filename': original_filename})
                    )
                except Exception as e:
                    logger.error(f"Failed to save metadata to database: {e}")
                    # Continue even if DB save fails

            file_info = FileInfo(
                name=uploaded_file.name,
                display_name=uploaded_file.display_name,
                mime_type=uploaded_file.mime_type,
                create_time=uploaded_file.create_time.isoformat() if uploaded_file.create_time else None,
                update_time=uploaded_file.update_time.isoformat() if uploaded_file.update_time else None,
                expiration_time=uploaded_file.expiration_time.isoformat() if uploaded_file.expiration_time else None,
                size_bytes=str(uploaded_file.size_bytes) if uploaded_file.size_bytes else str(file_size),
                sha256_hash=sha256_hash,
                uri=getattr(uploaded_file, 'uri', None),
                state=final_state,
                r2_url=r2_url,
                r2_key=r2_key,
                db_record_id=str(db_record_id) if db_record_id else None
            )

            return UploadResponse(
                success=True,
                file=file_info,
                message=f"File uploaded successfully: {file_display_name}"
            )
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        logger.error(f"Error uploading file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/files", response_model=FilesResponse)
async def list_files():
    """List all uploaded files in Gemini FileSearch."""
    if not genai_client:
        raise HTTPException(status_code=503, detail="Gemini API client not configured")

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


@app.delete("/files/{file_name}")
async def delete_file(file_name: str):
    """Delete a file from Gemini FileSearch, R2 storage, and database."""
    if not genai_client:
        raise HTTPException(status_code=503, detail="Gemini API client not configured")

    try:
        # First, delete from Gemini FileSearch
        try:
            genai_client.files.delete(name=file_name)
            logger.info(f"Deleted file from Gemini FileSearch: {file_name}")
        except Exception as e:
            logger.warning(f"Failed to delete from Gemini FileSearch: {e}")

        # Delete from Cloudflare R2 if configured and we have the key
        if r2_storage and railway_db:
            try:
                # Get R2 key from database
                file_record = await railway_db.fetchrow(
                    "SELECT cloudflare_r2_key FROM file_uploads WHERE gemini_file_name = $1",
                    file_name
                )

                if file_record and file_record['cloudflare_r2_key']:
                    await r2_storage.delete_file(file_record['cloudflare_r2_key'])
                    logger.info(f"Deleted file from R2 storage: {file_record['cloudflare_r2_key']}")
            except Exception as e:
                logger.warning(f"Failed to delete from R2 storage: {e}")

        # Delete from database (both file_uploads and potentially scraped_websites)
        if railway_db:
            try:
                # Delete from file_uploads
                deleted_uploads = await railway_db.execute(
                    "DELETE FROM file_uploads WHERE gemini_file_name = $1",
                    file_name
                )

                # Also check scraped_websites in case it's a scraped website
                deleted_scraped = await railway_db.execute(
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
    # Railway sets PORT, fallback to 8001
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)