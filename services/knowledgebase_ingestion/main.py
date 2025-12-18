@app.delete("/files/{file_name}")
async def delete_file(file_name: str):
    """Delete a file from Gemini FileSearch, R2 storage, and database."""
    try:
        # First, delete from Gemini FileSearch
        try:
            genai.delete_file(file_name)
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