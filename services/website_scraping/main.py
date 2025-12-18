@app.delete("/files/{file_name}")
async def delete_file(file_name: str):
    """Delete a scraped website file from Gemini FileSearch and database."""
    try:
        # Delete from Gemini FileSearch
        try:
            genai.delete_file(file_name)
            logger.info(f"Deleted scraped website from Gemini FileSearch: {file_name}")
        except Exception as e:
            logger.warning(f"Failed to delete from Gemini FileSearch: {e}")
        
        # Delete from database
        if railway_db:
            try:
                deleted = await railway_db.execute(
                    "DELETE FROM scraped_websites WHERE gemini_file_name = $1",
                    file_name
                )
                logger.info(f"Deleted scraped website from database: {deleted} records")
            except Exception as e:
                logger.warning(f"Failed to delete from database: {e}")
        
        return {"success": True, "message": f"Scraped website file {file_name} deleted"}
        
    except Exception as e:
        logger.error(f"Error deleting scraped website file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")