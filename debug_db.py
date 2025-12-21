import sys
import asyncio
import os

# Add current directory to path so we can import shared
sys.path.append(os.getcwd())

from shared.config import settings
from shared.db import Database

async def test_connection():
    url = settings.railway_postgres_url
    # Mask password for printing
    if url:
        masked_url = url.split('@')[-1] if '@' in url else "..."
        print(f"Testing connection to: postgres://****@{masked_url}")
    else:
        print("RAILWAY_POSTGRES_URL is not set!")
        return

    db = Database(url)
    try:
        print("Connecting...")
        await db.connect()
        print("Connected successfully!")
        
        print("Executing test query...")
        version = await db.fetchval("SELECT version()")
        print(f"Database version: {version}")
        
        await db.disconnect()
        print("Disconnected.")
    except Exception as e:
        print(f"Connection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Ensure event loop is closed properly
    try:
        asyncio.run(test_connection())
    except KeyboardInterrupt:
        print("Interrupted")
