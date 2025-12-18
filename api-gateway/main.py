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


@app.delete("/api/v1/knowledgebase/files/{file_name}")
async def delete_file(file_name: str):
    """Route to delete uploaded files and scraped websites from all storage layers."""
    async with httpx.AsyncClient() as client:
        try:
            # First, try to delete from knowledgebase ingestion service (handles uploaded files)
            response = await client.delete(
                f"{KNOWLEDGEBASE_INGESTION_URL}/files/{file_name}",
                timeout=30.0
            )
            
            # If knowledgebase service returns 404 (file not found there), 
            # it might be a scraped website - try website scraping service
            if response.status_code == 404:
                response = await client.delete(
                    f"{WEBSITE_SCRAPING_URL}/files/{file_name}",
                    timeout=30.0
                )
            
            response.raise_for_status()
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error deleting file: {e.response.text}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=e.response.json() if e.response.content else f"Failed to delete file: {file_name}"
            )
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            raise HTTPException(status_code=502, detail=f"Service error: {str(e)}")