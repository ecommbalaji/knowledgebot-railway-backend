"""Website Scraping Service - Handles website scraping using Crawl4AI and Gemini FileSearch."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any
from google import genai
import os
import logging
from dotenv import load_dotenv
import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
import tempfile
from contextlib import asynccontextmanager

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    try:
        # Startup
        logger.info("🚀 Website scraping service started successfully")
        logger.info("🏥 Health check endpoint: /health")
        logger.info("🔗 Scrape endpoint: POST /scrape")

        yield

        # Shutdown
        logger.info("🛑 Website scraping service shutting down")
    except Exception as e:
        logger.error(f"❌ Error in lifespan handler: {e}")
        raise

app = FastAPI(
    title="Website Scraping Service",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
logger.info(f"GEMINI_API_KEY configured: {'YES' if GEMINI_API_KEY else 'NO'}")

# Do not crash the whole process if GEMINI_API_KEY is missing or Gemini init fails.
# Instead, initialize `genai_client` defensively and let endpoints return 503 when
# the client is required. This prevents the container from exiting at startup
# due to missing optional configuration.
genai_client = None
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY environment variable is not set - Gemini-dependent endpoints will be unavailable")
else:
    try:
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini client initialized successfully")
    except Exception as e:
        genai_client = None
        logger.error(f"Failed to initialize Gemini client: {e}")


class ScrapeRequest(BaseModel):
    url: str
    max_depth: Optional[int] = 1
    max_pages: Optional[int] = 10
    include_patterns: Optional[list[str]] = None
    exclude_patterns: Optional[list[str]] = None
    wait_for: Optional[str] = None
    js_code: Optional[str] = None
    screenshot: Optional[bool] = False


class ScrapeResponse(BaseModel):
    success: bool
    message: str
    file_name: Optional[str] = None
    file_info: Optional[Dict[str, Any]] = None
    scraped_urls: Optional[list[str]] = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "website_scraping"}


@app.post("/scrape", response_model=ScrapeResponse)
async def scrape_website(request: ScrapeRequest):
    """
    Scrape a website and upload to Gemini FileSearch.
    
    Args:
        request: ScrapeRequest with URL and options
    
    Returns:
        ScrapeResponse with upload information
    """
    try:
        # Scrape the website using Crawl4AI
        logger.info(f"Scraping website: {request.url}")
        
        # Configure browser
        browser_config = BrowserConfig(verbose=False, headless=True)
        
        # Configure crawl options using CrawlerRunConfig
        run_config = CrawlerRunConfig()
        
        # Note: crawl4ai 0.7.x API changes - single URL crawling via arun()
        # For multi-page crawling, we'll use deep crawling or arun_many in future
        # For MVP, we'll scrape single page
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            # Execute crawl
            result = await crawler.arun(url=request.url, config=run_config)
            
            if not result.success:
                raise HTTPException(
                    status_code=500,
                    detail=f"Scraping failed: {result.error_message if hasattr(result, 'error_message') else 'Unknown error'}"
                )
            
            # Extract content - use markdown property (0.7.x API)
            scraped_content = ""
            if hasattr(result, 'markdown'):
                # markdown is a MarkdownOutput object with raw_markdown and fit_markdown
                if hasattr(result.markdown, 'raw_markdown'):
                    scraped_content = result.markdown.raw_markdown
                elif hasattr(result.markdown, 'fit_markdown'):
                    scraped_content = result.markdown.fit_markdown
                else:
                    scraped_content = str(result.markdown)
            
            if not scraped_content and hasattr(result, 'html'):
                scraped_content = result.html
            if not scraped_content and hasattr(result, 'cleaned_html'):
                scraped_content = result.cleaned_html
            
            if not scraped_content:
                raise HTTPException(
                    status_code=500,
                    detail="No content extracted from website"
                )
            
            # Get scraped URLs
            scraped_urls = [request.url]
            if hasattr(result, 'links') and result.links:
                # links is a dict with 'internal' and 'external' keys in 0.7.x
                if isinstance(result.links, dict):
                    if 'internal' in result.links:
                        internal_links = result.links['internal']
                        if isinstance(internal_links, list):
                            # Extract href from link dicts
                            urls = [link.get('href', link) if isinstance(link, dict) else link for link in internal_links[:request.max_pages or 10]]
                            scraped_urls.extend(urls)
                elif isinstance(result.links, list):
                    scraped_urls.extend(result.links[:request.max_pages or 10])
            
            # Ensure Gemini client is available before attempting upload
            if not genai_client:
                logger.error("Gemini client not available - cannot upload scraped content")
                from shared.utils import dependency_unavailable_error
                raise dependency_unavailable_error("gemini", "client not configured")

            # Save to temporary file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.md',
                delete=False,
                encoding='utf-8'
            ) as tmp_file:
                tmp_file.write(scraped_content)
                tmp_path = tmp_file.name
            
            try:
                # Generate display name from URL
                from urllib.parse import urlparse
                parsed_url = urlparse(request.url)
                domain = parsed_url.netloc.replace('www.', '')
                display_name = f"scraped_{domain}_{os.path.basename(tmp_path)}.md"
                
                # Upload to Gemini FileSearch
                logger.info(f"Uploading scraped content to Gemini FileSearch: {display_name}")
                uploaded_file = genai_client.files.upload(
                    path=tmp_path,
                    config=dict(display_name=display_name)
                )
                
                # Wait for file processing
                logger.info(f"Uploaded file: {uploaded_file.name}, waiting for processing...")
                
                file_info = {
                    "name": uploaded_file.name,
                    "display_name": uploaded_file.display_name,
                    "mime_type": uploaded_file.mime_type,
                    "state": uploaded_file.state.name if hasattr(uploaded_file, 'state') else None,
                }
                
                return ScrapeResponse(
                    success=True,
                    message=f"Website scraped and uploaded successfully",
                    file_name=uploaded_file.name,
                    file_info=file_info,
                    scraped_urls=scraped_urls
                )
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scraping website: {e}")
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Railway sets PORT, fallback to 8002
    port = int(os.getenv("PORT", "8002"))
    uvicorn.run(app, host="0.0.0.0", port=port)
