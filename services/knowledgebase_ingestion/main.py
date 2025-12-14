"""Knowledgebase Ingestion Service - Handles document uploads and Gemini FileSearch."""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
import os
import logging
from dotenv import load_dotenv
import json

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

# Initialize Gemini (optional for healthcheck)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY environment variable not set - API endpoints will fail")
else:
    genai.configure(api_key=GEMINI_API_KEY)


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


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    display_name: Optional[str] = None
):
    """
    Upload a document to Gemini FileSearch.
    
    Args:
        file: The file to upload
        display_name: Optional display name for the file
    
    Returns:
        UploadResponse with file information
    """
    try:
        # Read file content
        content = await file.read()
        
        # Use display name or filename
        file_display_name = display_name or file.filename or "uploaded_file"
        
        # Gemini API requires file path, so write to temp file
        import tempfile
        file_ext = os.path.splitext(file.filename or "")[1] or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Upload to Gemini FileSearch
            uploaded_file = genai.upload_file(
                path=tmp_path,
                display_name=file_display_name,
                mime_type=file.content_type or "application/octet-stream"
            )
            
            # Wait for file processing (check state)
            logger.info(f"Uploaded file: {uploaded_file.name}, state: {uploaded_file.state}")
            
            file_info = FileInfo(
                name=uploaded_file.name,
                display_name=uploaded_file.display_name,
                mime_type=uploaded_file.mime_type,
                create_time=uploaded_file.create_time.isoformat() if uploaded_file.create_time else None,
                update_time=uploaded_file.update_time.isoformat() if uploaded_file.update_time else None,
                expiration_time=uploaded_file.expiration_time.isoformat() if uploaded_file.expiration_time else None,
                size_bytes=str(uploaded_file.size_bytes) if uploaded_file.size_bytes else None,
                sha256_hash=getattr(uploaded_file, 'sha256_hash', None),
                uri=getattr(uploaded_file, 'uri', None),
                state=uploaded_file.state.name if hasattr(uploaded_file, 'state') else None,
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
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/files", response_model=FilesResponse)
async def list_files():
    """List all uploaded files in Gemini FileSearch."""
    try:
        files = genai.list_files()
        
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
    """Delete a file from Gemini FileSearch."""
    try:
        genai.delete_file(file_name)
        return {"success": True, "message": f"File {file_name} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Railway sets PORT, fallback to 8001
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
