#!/usr/bin/env python3
"""
Cleanup and verification script to check database vs Gemini FileSearch consistency.

This script:
1. Lists all files in the database
2. Verifies each file exists in Gemini FileSearch
3. Lists orphaned files (in DB but not in Gemini)
4. Lists missing files (in Gemini but not in DB)
5. Optionally cleans up orphaned database records
"""

import os
import sys
import asyncio
import logging
from typing import List, Dict, Tuple
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import Settings
from shared import db
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize settings
settings = Settings()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def get_all_database_files():
    """Get all files from database (both file_uploads and scraped_websites)."""
    if not db.railway_db:
        await db.init_railway_db(settings.railway_postgres_url)
    
    # Get uploaded files
    uploaded_files = await db.railway_db.fetch(
        """
        SELECT id, gemini_file_name, original_filename, 'file_uploads' as source_table
        FROM file_uploads
        WHERE gemini_file_name IS NOT NULL
        """
    )
    
    # Get scraped websites
    scraped_files = await db.railway_db.fetch(
        """
        SELECT id, gemini_file_name, original_url as original_filename, 'scraped_websites' as source_table
        FROM scraped_websites
        WHERE gemini_file_name IS NOT NULL
        """
    )
    
    all_files = []
    for row in uploaded_files:
        all_files.append({
            'id': str(row['id']),
            'gemini_file_name': row['gemini_file_name'],
            'original_filename': row['original_filename'],
            'source_table': row['source_table']
        })
    
    for row in scraped_files:
        all_files.append({
            'id': str(row['id']),
            'gemini_file_name': row['gemini_file_name'],
            'original_filename': row['original_filename'],
            'source_table': row['source_table']
        })
    
    return all_files


async def get_all_gemini_files():
    """Get all files from Gemini FileSearch."""
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY not configured")
    
    genai_client = genai.Client(api_key=settings.gemini_api_key)
    
    all_gemini_files = []
    try:
        # List all files (handles pagination)
        files = genai_client.files.list()
        for file in files:
            all_gemini_files.append({
                'name': file.name,
                'display_name': getattr(file, 'display_name', 'N/A'),
                'state': file.state.name if hasattr(file, 'state') else 'UNKNOWN',
                'mime_type': getattr(file, 'mime_type', 'N/A'),
                'size_bytes': getattr(file, 'size_bytes', 0)
            })
    except Exception as e:
        logger.error(f"Error listing Gemini files: {e}")
        raise
    
    return all_gemini_files


async def verify_file_in_gemini(gemini_file_name: str, genai_client) -> Tuple[bool, str]:
    """Verify if a file exists in Gemini and return its state."""
    try:
        file_info = genai_client.files.get(name=gemini_file_name)
        return True, file_info.state.name if hasattr(file_info, 'state') else 'UNKNOWN'
    except Exception as e:
        error_str = str(e)
        if "403" in error_str or "404" in error_str or "PERMISSION_DENIED" in error_str or "NOT_FOUND" in error_str:
            return False, "NOT_FOUND"
        else:
            return False, f"ERROR: {error_str}"


async def main():
    """Main verification function."""
    logger.info("="*60)
    logger.info("Gemini FileSearch Consistency Verification")
    logger.info("="*60)
    
    # Initialize database
    if not db.railway_db:
        railway_postgres_url = os.getenv("RAILWAY_POSTGRES_URL")
        if not railway_postgres_url:
            logger.error("RAILWAY_POSTGRES_URL not configured")
            return
        await db.init_railway_db(railway_postgres_url)
    
    # Initialize Gemini client
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.error("GEMINI_API_KEY not configured")
        return
    
    genai_client = genai.Client(api_key=gemini_api_key)
    
    # Get all files from database
    logger.info("📊 Fetching files from database...")
    db_files = await get_all_database_files()
    logger.info(f"Found {len(db_files)} files in database")
    
    # Get all files from Gemini
    logger.info("📊 Fetching files from Gemini FileSearch...")
    gemini_files = await get_all_gemini_files()
    logger.info(f"Found {len(gemini_files)} files in Gemini FileSearch")
    
    # Create sets for comparison
    db_gemini_names = {f['gemini_file_name'] for f in db_files}
    gemini_names = {f['name'] for f in gemini_files}
    
    # Find orphaned files (in DB but not in Gemini)
    orphaned_files = []
    for db_file in db_files:
        gemini_name = db_file['gemini_file_name']
        exists, state = await verify_file_in_gemini(gemini_name, genai_client)
        if not exists:
            orphaned_files.append({
                **db_file,
                'gemini_status': state
            })
    
    # Find missing files (in Gemini but not in DB)
    missing_files = [f for f in gemini_files if f['name'] not in db_gemini_names]
    
    # Print summary
    logger.info("="*60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total files in database: {len(db_files)}")
    logger.info(f"Total files in Gemini: {len(gemini_files)}")
    logger.info(f"Orphaned files (in DB but not in Gemini): {len(orphaned_files)}")
    logger.info(f"Missing files (in Gemini but not in DB): {len(missing_files)}")
    
    # Print orphaned files
    if orphaned_files:
        logger.info("\n" + "="*60)
        logger.info("ORPHANED FILES (in database but not in Gemini):")
        logger.info("="*60)
        for file in orphaned_files:
            logger.info(f"  - ID: {file['id']}")
            logger.info(f"    Gemini Name: {file['gemini_file_name']}")
            logger.info(f"    Original: {file['original_filename']}")
            logger.info(f"    Table: {file['source_table']}")
            logger.info(f"    Status: {file['gemini_status']}")
            logger.info("")
    
    # Print missing files
    if missing_files:
        logger.info("\n" + "="*60)
        logger.info("MISSING FILES (in Gemini but not in database):")
        logger.info("="*60)
        for file in missing_files:
            logger.info(f"  - Gemini Name: {file['name']}")
            logger.info(f"    Display Name: {file['display_name']}")
            logger.info(f"    State: {file['state']}")
            logger.info(f"    MIME Type: {file['mime_type']}")
            logger.info(f"    Size: {file['size_bytes']} bytes")
            logger.info("")
    
    # Option to clean up orphaned files
    if orphaned_files:
        logger.info("\n" + "="*60)
        logger.info("CLEANUP OPTIONS")
        logger.info("="*60)
        logger.info(f"Found {len(orphaned_files)} orphaned database records.")
        logger.info("These files exist in the database but not in Gemini.")
        logger.info("You can manually delete them using the delete endpoint with their database ID.")
        logger.info("\nOrphaned file IDs:")
        for file in orphaned_files:
            logger.info(f"  - {file['id']} ({file['gemini_file_name']})")
    
    logger.info("\n" + "="*60)
    logger.info("Verification complete!")
    logger.info("="*60)


if __name__ == "__main__":
    asyncio.run(main())

