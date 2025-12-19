"""Chatbot Orchestration Service - Pydantic AI Agent with intelligent data source routing."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Annotated
import os
import logging
from dataclasses import dataclass
from dotenv import load_dotenv
import uuid
from datetime import datetime
from google import genai
from contextlib import asynccontextmanager
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
from pydantic_ai.models.openai import OpenAIModel
import asyncio
import sys
from pathlib import Path
import json

# Add shared directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.config import settings
from shared.db import init_railway_db, init_neon_db, railway_db, neon_db

# Lazy database initialization for serverless optimization
async def get_railway_db():
    """Get Railway database connection, initializing if needed."""
    global railway_db
    if railway_db is None and settings.railway_postgres_url:
        try:
            logger.info("🔄 Lazy initializing Railway PostgreSQL database...")
            await init_railway_db(settings.railway_postgres_url)
            logger.info("✅ Railway PostgreSQL database initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Railway PostgreSQL database: {e}")
            raise
    return railway_db

async def get_neon_db():
    """Get Neon database connection, initializing if needed."""
    global neon_db
    if neon_db is None and settings.neon_db_url:
        try:
            logger.info("🔄 Lazy initializing Neon database...")
            await init_neon_db(settings.neon_db_url)
            logger.info("✅ Neon database initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Neon database: {e}")
            raise
    return neon_db

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure shared utilities are importable and enable global exception logging
import sys
from pathlib import Path
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
except Exception:
    logger.debug("Could not adjust sys.path for shared imports")

from shared.utils import setup_global_exception_logging, register_fastapi_exception_handlers, dependency_unavailable_error
setup_global_exception_logging("chatbot_orchestration")

# Validate required environment variables for this service
if not settings.gemini_api_key:
    raise dependency_unavailable_error("gemini_api_key", "Chatbot orchestration service requires GEMINI_API_KEY")
if not settings.openai_api_key:
    raise dependency_unavailable_error("openai_api_key", "Chatbot orchestration service requires OPENAI_API_KEY")

# Lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    logger.info("🔄 Chatbot orchestration lifespan starting...")

    # For serverless optimization: Skip heavy DB initialization during startup
    # Databases will be initialized lazily on first request to reduce cold start time
    logger.info("🚀 Chatbot orchestration service started successfully (lazy DB init)")
    logger.info("🏥 Health check endpoint: /health")
    logger.info("💬 Chat endpoint: POST /chat")
    logger.info("📊 Sessions endpoint: GET /sessions")

    yield

    # Shutdown - Close database connections if they exist
    logger.info("🛑 Shutting down chatbot orchestration service...")
    try:
        if railway_db and not railway_db.is_closed:
            await railway_db.disconnect()
            logger.info("✅ Railway PostgreSQL connection closed")
    except Exception as e:
        logger.warning(f"Error closing Railway DB: {e}")

    try:
        if neon_db and not neon_db.is_closed:
            await neon_db.disconnect()
            logger.info("✅ Neon DB connection closed")
    except Exception as e:
        logger.warning(f"Error closing Neon DB: {e}")

    logger.info("✅ Chatbot orchestration service shutdown complete")

app = FastAPI(
    title="Chatbot Orchestration Service",
    version="1.0.0",
    lifespan=lifespan
)

# Register FastAPI-level exception handlers to ensure stack traces are logged
register_fastapi_exception_handlers(app, "chatbot_orchestration")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini
logger.info("🔄 Initializing AI components...")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or settings.gemini_api_key
genai_client = None
if not GEMINI_API_KEY:
    logger.warning("⚠️  GEMINI_API_KEY not set - some features will fail")
else:
    try:
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("✅ Gemini client initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Gemini client: {e}")

# Initialize OpenAI model for Pydantic AI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or settings.openai_api_key
if not OPENAI_API_KEY:
    logger.warning("⚠️  OPENAI_API_KEY not set - chat endpoints will fail")
else:
    logger.info("✅ OpenAI API key configured")

MODEL_NAME = os.getenv("CHATBOT_MODEL", settings.chatbot_model)
TEMPERATURE = float(os.getenv("CHATBOT_TEMPERATURE", str(settings.chatbot_temperature)))
MAX_TOKENS = int(os.getenv("CHATBOT_MAX_TOKENS", str(settings.chatbot_max_tokens)))
logger.info(f"🤖 Model config: {MODEL_NAME}, temp={TEMPERATURE}, max_tokens={MAX_TOKENS}")

# Initialize Tavily for internet search (optional)
tavily_client = None
if settings.tavily_api_key and settings.enable_internet_search:
    try:
        from tavily import TavilyClient
        tavily_client = TavilyClient(api_key=settings.tavily_api_key)
        logger.info("✅ Tavily internet search initialized")
    except Exception as e:
        logger.warning(f"⚠️  Failed to initialize Tavily: {e}")
else:
    logger.info("ℹ️  Tavily not configured or disabled")

# In-memory session storage
sessions: Dict[str, Dict[str, Any]] = {}


# Pydantic models for structured outputs
class SearchResult(BaseModel):
    """Search result from Gemini FileSearch."""
    file_name: str
    content: str
    relevance_score: Optional[float] = None


class ChatResponse(BaseModel):
    """Structured chat response."""
    answer: str = Field(description="The answer to the user's question")
    sources: List[SearchResult] = Field(default_factory=list, description="Sources used for the answer")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score for the answer")
    data_sources_used: List[str] = Field(default_factory=list, description="Data sources used: rag, postgres, neon_db, internet")


class HumanReviewRequest(BaseModel):
    """Request for human review."""
    approved: bool
    feedback: Optional[str] = None
    corrected_answer: Optional[str] = None


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str
    session_id: Optional[str] = None
    use_rag: bool = True
    max_results: int = 5


class ChatSessionResponse(BaseModel):
    """Chat session response."""
    session_id: str
    message: str
    response: ChatResponse
    usage: Optional[Dict[str, Any]] = None
    timestamp: str



# Pydantic AI Agent Setup
# OpenAI model reads API key from OPENAI_API_KEY environment variable automatically
openai_model = None

if OPENAI_API_KEY:
    # Just pass the model name - API key is read from environment
    try:
        openai_model = OpenAIModel(MODEL_NAME)
        logger.info("✅ OpenAI model initialized")
    except Exception as e:
        openai_model = None
        logger.error(f"❌ Failed to initialize OpenAIModel: {e}")
        logger.error("OpenAI model will be unavailable; chat endpoints may return 503 or degraded responses")
else:
    logger.warning("OpenAI model not initialized - OPENAI_API_KEY is missing")


# Tool for querying Railway PostgreSQL (file metadata, user data)
async def query_railway_postgres(
    query: Annotated[str, "SQL-like query description or natural language question about file uploads, users, or metrics"]
) -> str:
    """
    Query Railway PostgreSQL database for file metadata, user information, or metrics.
    
    Use this for questions about:
    - Uploaded files and their metadata
    - User information (non-PII only)
    - System metrics and analytics
    - File upload history
    
    IMPORTANT: Never expose PII (personally identifiable information) like emails, names, or personal data.
    Only return aggregated statistics, file metadata, and anonymized information.
    """
    try:
        db = await get_railway_db()
        if not db:
            return "Railway PostgreSQL database is not configured."
    except Exception as e:
        return f"Failed to initialize Railway PostgreSQL database: {e}"
    
    try:
        # Parse the query and construct appropriate SQL
        query_lower = query.lower()
        
        # File-related queries
        if any(word in query_lower for word in ['file', 'upload', 'document', 'document']):
            if 'count' in query_lower or 'total' in query_lower or 'number' in query_lower:
                result = await railway_db.fetchval(
                    "SELECT COUNT(*) FROM file_uploads WHERE gemini_state = 'ACTIVE'"
                )
                return f"Total active files in the system: {result}"
            elif 'recent' in query_lower or 'latest' in query_lower:
                files = await railway_db.fetch(
                    """
                    SELECT display_name, mime_type, size_bytes, uploaded_at
                    FROM file_uploads
                    WHERE gemini_state = 'ACTIVE'
                    ORDER BY uploaded_at DESC
                    LIMIT 5
                    """
                )
                if files:
                    result = "Recent uploaded files:\n"
                    for f in files:
                        result += f"- {f['display_name']} ({f['mime_type']}, {f['size_bytes']} bytes, uploaded {f['uploaded_at']})\n"
                    return result
                return "No recent files found."
            else:
                # General file info
                files = await railway_db.fetch(
                    """
                    SELECT display_name, mime_type, size_bytes, uploaded_at
                    FROM file_uploads
                    WHERE gemini_state = 'ACTIVE'
                    ORDER BY uploaded_at DESC
                    LIMIT 10
                    """
                )
                if files:
                    result = f"Found {len(files)} active files:\n"
                    for f in files:
                        result += f"- {f['display_name']} ({f['mime_type']})\n"
                    return result
                return "No files found in the database."
        
        # Metrics queries
        elif any(word in query_lower for word in ['metric', 'statistic', 'analytics', 'usage']):
            metrics = await railway_db.fetch(
                """
                SELECT metric_name, SUM(value::numeric) as total_value, unit
                FROM metrics
                WHERE recorded_at > NOW() - INTERVAL '7 days'
                GROUP BY metric_name, unit
                ORDER BY total_value DESC
                LIMIT 10
                """
            )
            if metrics:
                result = "Recent metrics (last 7 days):\n"
                for m in metrics:
                    result += f"- {m['metric_name']}: {m['total_value']} {m['unit'] or ''}\n"
                return result
            return "No metrics found."
        
        # Default: return file count
        count = await railway_db.fetchval("SELECT COUNT(*) FROM file_uploads WHERE gemini_state = 'ACTIVE'")
        return f"Database contains {count} active files. Please be more specific about what information you need."
        
    except Exception as e:
        logger.error(f"Error querying Railway PostgreSQL: {e}")
        return f"Error querying database: {str(e)}"


# Tool for querying Neon DB (business data)
async def query_neon_db(
    query: Annotated[str, "Natural language question about business data: products, orders, customers, sales, inventory"]
) -> str:
    """
    Query Neon DB business database for product, order, customer, sales, or inventory information.
    
    Use this for questions about:
    - Products and product catalog
    - Orders and transactions
    - Sales analytics and trends
    - Inventory levels
    - Customer segments (anonymized, no PII)
    
    IMPORTANT: Never expose PII. Only return aggregated business data, product information, and anonymized statistics.
    """
    try:
        db = await get_neon_db()
        if not db:
            return "Neon DB business database is not configured."
    except Exception as e:
        return f"Failed to initialize Neon DB business database: {e}"
    
    try:
        query_lower = query.lower()
        
        # Product queries
        if any(word in query_lower for word in ['product', 'item', 'catalog']):
            if 'available' in query_lower or 'stock' in query_lower:
                products = await neon_db.fetch(
                    """
                    SELECT product_name, category, price, stock_quantity, rating
                    FROM products
                    WHERE is_available = TRUE AND stock_quantity > 0
                    ORDER BY rating DESC NULLS LAST
                    LIMIT 10
                    """
                )
                if products:
                    result = "Available products:\n"
                    for p in products:
                        result += f"- {p['product_name']} ({p['category']}) - ${p['price']}, Stock: {p['stock_quantity']}, Rating: {p['rating'] or 'N/A'}\n"
                    return result
                return "No available products found."
            elif 'category' in query_lower:
                categories = await neon_db.fetch(
                    """
                    SELECT category, COUNT(*) as count, AVG(price) as avg_price
                    FROM products
                    WHERE is_available = TRUE
                    GROUP BY category
                    ORDER BY count DESC
                    """
                )
                if categories:
                    result = "Products by category:\n"
                    for c in categories:
                        result += f"- {c['category']}: {c['count']} products, Average price: ${float(c['avg_price']):.2f}\n"
                    return result
                return "No category data found."
            else:
                products = await neon_db.fetch(
                    "SELECT product_name, category, price FROM products WHERE is_available = TRUE LIMIT 10"
                )
                if products:
                    result = "Sample products:\n"
                    for p in products:
                        result += f"- {p['product_name']} ({p['category']}) - ${p['price']}\n"
                    return result
                return "No products found."
        
        # Order queries
        elif any(word in query_lower for word in ['order', 'transaction', 'purchase']):
            if 'recent' in query_lower or 'latest' in query_lower:
                orders = await neon_db.fetch(
                    """
                    SELECT order_id, order_status, total_amount, order_date
                    FROM orders
                    ORDER BY order_date DESC
                    LIMIT 5
                    """
                )
                if orders:
                    result = "Recent orders:\n"
                    for o in orders:
                        result += f"- Order {o['order_id']}: ${o['total_amount']} ({o['order_status']}) on {o['order_date']}\n"
                    return result
                return "No recent orders found."
            elif 'total' in query_lower or 'revenue' in query_lower:
                revenue = await neon_db.fetchval(
                    "SELECT SUM(total_amount) FROM orders WHERE order_status != 'cancelled'"
                )
                count = await neon_db.fetchval(
                    "SELECT COUNT(*) FROM orders WHERE order_status != 'cancelled'"
                )
                return f"Total revenue: ${float(revenue or 0):.2f} from {count} orders."
            else:
                orders = await neon_db.fetch(
                    """
                    SELECT order_status, COUNT(*) as count, SUM(total_amount) as total
                    FROM orders
                    GROUP BY order_status
                    """
                )
                if orders:
                    result = "Orders by status:\n"
                    for o in orders:
                        result += f"- {o['order_status']}: {o['count']} orders, Total: ${float(o['total'] or 0):.2f}\n"
                    return result
                return "No order data found."
        
        # Sales analytics
        elif any(word in query_lower for word in ['sales', 'revenue', 'analytics', 'trend']):
            analytics = await neon_db.fetch(
                """
                SELECT category, SUM(total_revenue) as revenue, SUM(total_orders) as orders
                FROM sales_analytics
                WHERE date >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY category
                ORDER BY revenue DESC
                LIMIT 5
                """
            )
            if analytics:
                result = "Sales by category (last 30 days):\n"
                for a in analytics:
                    result += f"- {a['category']}: ${float(a['revenue'] or 0):.2f} revenue, {a['orders']} orders\n"
                return result
            return "No sales analytics data found."
        
        # Inventory queries
        elif any(word in query_lower for word in ['inventory', 'stock', 'warehouse']):
            inventory = await neon_db.fetch(
                """
                SELECT p.product_name, i.quantity_available, i.warehouse_location
                FROM inventory i
                JOIN products p ON i.product_id = p.product_id
                WHERE i.quantity_available < i.reorder_level
                ORDER BY i.quantity_available ASC
                LIMIT 10
                """
            )
            if inventory:
                result = "Low stock items:\n"
                for inv in inventory:
                    result += f"- {inv['product_name']}: {inv['quantity_available']} units at {inv['warehouse_location']}\n"
                return result
            return "All inventory levels are adequate."
        
        # Default response
        return "Please specify what business data you need: products, orders, sales, or inventory."
        
    except Exception as e:
        logger.error(f"Error querying Neon DB: {e}")
        return f"Error querying business database: {str(e)}"


# Tool for internet search
async def search_internet(
    query: Annotated[str, "Search query for current information from the internet"]
) -> str:
    """
    Search the internet for current information using Tavily API.
    
    Use this for questions about:
    - Current events and news
    - Real-time information
    - General knowledge not in the knowledge base
    - Recent developments or updates
    
    Only use when information is not available in RAG, PostgreSQL, or Neon DB.
    """
    if not tavily_client:
        return "Internet search is not configured or enabled."
    
    try:
        response = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=3
        )
        
        if response.get('results'):
            result_text = f"Internet search results for '{query}':\n\n"
            for idx, result in enumerate(response['results'][:3], 1):
                result_text += f"{idx}. {result.get('title', 'No title')}\n"
                result_text += f"   {result.get('content', 'No content')[:200]}...\n"
                if result.get('url'):
                    result_text += f"   Source: {result['url']}\n"
                result_text += "\n"
            return result_text
        return f"No internet search results found for '{query}'."
        
    except Exception as e:
        logger.error(f"Error searching internet: {e}")
        return f"Error performing internet search: {str(e)}"


# Tool for querying Gemini FileSearch (RAG)
async def search_knowledge_base(query: Annotated[str, "The search query to find relevant information in uploaded documents"]) -> List[SearchResult]:
    """
    Search the knowledge base using Gemini FileSearch for relevant information.

    This tool searches through uploaded documents and scraped content to find
    information relevant to the user's query.
    """
    if not genai_client:
        return [SearchResult(
            file_name="System_Error",
            content="Gemini API client not configured - cannot search knowledge base",
            relevance_score=0.0
        )]

    try:
        # List all files in Gemini FileSearch
        # Convert generator to list
        all_files = list(genai_client.files.list())
        
        if not all_files:
            logger.warning("No files found in FileSearch store")
            return []
            
        # Filter for ACTIVE files only
        active_files = [f for f in all_files if f.state.name == "ACTIVE"]
        
        if not active_files:
            logger.warning("No ACTIVE files found in FileSearch store")
            return []
            
        # Sort by creation time (descending) to get the most recent files
        active_files.sort(key=lambda f: f.create_time, reverse=True)
        
        # Use simple heuristic: take up to 5 most recent files to avoid payload limits during debugging
        # In a real app, you might filter by name or metadata
        files_to_search = active_files[:5]
        
        logger.info(f"Searching {len(files_to_search)} files with Gemini 2.5 Flash Lite for query: {query}")

        try:
            # Construct the retrieval prompt
            retrieval_prompt = f"""
            You are a specialized retrieval system. Your task is to extract information from the provided files to answer the user's query.

            User Query: "{query}"

            Instructions:
            1. Search through the attached files for information relevant to the query.
            2. Extract direct quotes, data points, and context that answer the question.
            3. If the files contain the answer, provide a detailed summary of the relevant information.
            4. If the files do NOT contain the answer, state "No relevant information found in the knowledge base."

            Output Format:
            - Source File: [Filename]
            - Relevant Content: [Extracted Information]
            """

            # Generate content using the new API with files attached
            contents = [*files_to_search, retrieval_prompt]
            response = genai_client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=contents
            )

            # Create a single consolidated result from the LLM's retrieval
            # This acts as the "context" for the downstream orchestration agent
            return [SearchResult(
                file_name="Gemini_Neural_Retrieval_2.5_Flash_Lite",
                content=response.text,
                relevance_score=1.0
            )]
            
        except Exception as e:
            logger.error(f"Error in Neural Retrieval: {e}")
            # Fallback (return list of files if retrieval fails)
            # EXPOSE THE ERROR for debugging purposes
            return [SearchResult(
                file_name="System_Error",
                content=f"Error performing semantic search: {str(e)}. Files attempting to search: {', '.join(f.name for f in files_to_search)}",
                relevance_score=0.1
            )]
        
    except Exception as e:
        logger.error(f"Error searching knowledge base: {e}")
        return []


# System prompt with intelligent routing instructions
def get_system_prompt(file_context: Optional[List[SearchResult]] = None) -> str:
    """Generate dynamic system prompt with intelligent data source routing."""
    base_prompt = """You are an intelligent knowledge assistant chatbot with access to multiple data sources.

Your role is to intelligently route user queries to the appropriate data source(s) to provide accurate answers.

AVAILABLE DATA SOURCES AND WHEN TO USE THEM:

1. **search_knowledge_base** (RAG - Gemini FileSearch):
   - Use for questions about content in uploaded documents, PDFs, text files
   - Use for questions about scraped website content
   - Use when the user asks about specific documents or file contents
   - This searches through semantically indexed documents

2. **query_railway_postgres** (Railway PostgreSQL):
   - Use for questions about file uploads, file metadata, upload history
   - Use for system metrics, analytics, and usage statistics
   - Use for questions about the knowledge base system itself
   - NEVER expose PII (personally identifiable information) - only return aggregated/anonymized data

3. **query_neon_db** (Neon DB - Business Database):
   - Use for questions about products, product catalog, pricing
   - Use for questions about orders, transactions, sales
   - Use for questions about inventory, stock levels, warehouse data
   - Use for sales analytics, revenue trends, business metrics
   - NEVER expose PII - only return business data and anonymized statistics

4. **search_internet** (Tavily - Internet Search):
   - Use ONLY when information is not available in other sources
   - Use for current events, real-time information, recent news
   - Use for general knowledge questions not in the knowledge base
   - Use as a last resort after checking other sources

ROUTING STRATEGY:
- Analyze the user's question to determine the most appropriate data source(s)
- You can use multiple sources if needed to provide a complete answer
- Always prioritize RAG (search_knowledge_base) for document-related questions
- Use PostgreSQL for system/file metadata questions
- Use Neon DB for business/product/order questions
- Use internet search only when other sources don't have the information
- Never reveal PII (emails, names, personal customer data) - only return anonymized/aggregated data
- Be transparent about which data sources you used

When answering:
1. Intelligently select the appropriate tool(s) based on the question
2. Combine information from multiple sources if needed
3. Provide accurate, helpful answers
4. Clearly indicate when information is not available
5. Be conversational and helpful"""
    
    if file_context:
        context_section = "\n\nAvailable knowledge base files (from RAG):\n"
        for idx, result in enumerate(file_context, 1):
            context_section += f"{idx}. {result.file_name}\n"
        return base_prompt + context_section
    
    return base_prompt


@dataclass
class ChatSessionDeps:
    """Dependencies for chat session."""
    session_id: str

def create_session_dependency(session_id: str) -> ChatSessionDeps:
    """Create session dependency instance."""
    return ChatSessionDeps(session_id=session_id)

# Initialize base agent with all tools
def create_agent(file_context: Optional[List[SearchResult]] = None) -> Optional[Agent]:
    """Create a Pydantic AI agent with intelligent data source routing."""
    
    # Check if model is available
    if openai_model is None:
        logger.error("Cannot create agent - OpenAI API key not configured")
        return None

    # Build list of available tools
    tools = [search_knowledge_base]
    
    # Add PostgreSQL tool if available
    if railway_db:
        tools.append(query_railway_postgres)
    
    # Add Neon DB tool if available
    if neon_db:
        tools.append(query_neon_db)
    
    # Add internet search tool if available
    if tavily_client:
        tools.append(search_internet)

    # Create agent with system prompt, tools, and dependencies
    agent = Agent(
        openai_model,
        system_prompt=get_system_prompt(file_context),
        tools=tools,
        deps_type=ChatSessionDeps,
    )
    
    return agent


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "chatbot_orchestration"}


@app.get("/ready")
async def readiness_check():
    """Readiness endpoint to check critical dependencies."""
    try:
        # For serverless: Check configuration rather than actual connections
        # Databases will be initialized lazily on first use
        db_checks = []

        # Check if database URLs are configured (lazy init will handle actual connections)
        if settings.railway_postgres_url:
            db_checks.append({"name": "railway_db", "status": "configured", "lazy_init": True})
        else:
            db_checks.append({"name": "railway_db", "status": "not_configured"})

        if settings.neon_db_url:
            db_checks.append({"name": "neon_db", "status": "configured", "lazy_init": True})
        else:
            db_checks.append({"name": "neon_db", "status": "not_configured"})

        # Check AI components (already initialized at module level)
        ai_checks = []
        if GEMINI_API_KEY and genai_client:
            ai_checks.append({"name": "gemini_api", "status": "ready"})
        else:
            ai_checks.append({"name": "gemini_api", "status": "not_configured"})

        if OPENAI_API_KEY:
            ai_checks.append({"name": "openai_api", "status": "configured"})
        else:
            ai_checks.append({"name": "openai_api", "status": "not_configured"})

        return {
            "status": "ready",
            "databases": db_checks,
            "ai_components": ai_checks,
            "serverless_optimized": True,
            "timestamp": "2025-12-19T18:00:00Z"
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


@app.post("/chat", response_model=ChatSessionResponse)
async def chat(request: ChatRequest):
    """
    Handle chat request with Pydantic AI agent and RAG.
    
    Args:
        request: ChatRequest with message and optional session_id
    
    Returns:
        ChatSessionResponse with structured answer
    """
    try:
        # Get or create session
        session_id = request.session_id or str(uuid.uuid4())
        
        # Perform RAG search if enabled (pre-fetch context)
        file_context = []
        if request.use_rag:
            file_context = await search_knowledge_base(request.message)
        
        # Create or get session
        if session_id not in sessions:
            sessions[session_id] = {
                "created_at": datetime.utcnow().isoformat(),
                "messages": [],
            }
        
        session = sessions[session_id]
        
        # Create agent with context (dynamic prompt injection)
        agent = create_agent(file_context)
        
        # Create dependency instance for this run
        session_dep = create_session_dependency(session_id)

        
        # Check if agent was created successfully
        if agent is None:
            raise HTTPException(
                status_code=503,
                detail="Chatbot service not configured - OpenAI API key required"
            )
        
        # Build chat history from session
        chat_history = session["messages"]
        
        # Convert chat history to agent messages
        history_messages = []
        for msg in chat_history[-10:]:  # Keep last 10 messages for context
            if msg["role"] == "user":
                history_messages.append(ModelRequest(parts=[UserPromptPart(content=msg["content"])]))
            elif msg["role"] == "assistant":
                # Ensure content is a string
                content_str = str(msg["content"]) if msg["content"] is not None else ""
                history_messages.append(ModelResponse(parts=[TextPart(content=content_str)]))
        
        # Run agent (with self-correction via model_retry)
        # Pass the current message as prompt and previous messages as history
        # Also pass the dependency instance for this specific run
        result = await agent.run(
            request.message, 
            message_history=history_messages,
            deps=session_dep
        )
        
        # Extract response text
        response_text = ""
        if hasattr(result, 'output'):
             # pydantic-ai v1.32.0 prefers .output (validated result)
             response_text = result.output if isinstance(result.output, str) else str(result.output)
        elif hasattr(result, 'data'):
             # fallback or older versions
             response_text = str(result.data)
        elif hasattr(result, 'response') and result.response:
             # fallback for raw response access attempt
             response_text = result.response.text if hasattr(result.response, 'text') else str(result.response)
        
        # Determine which data sources were used based on tool calls
        data_sources_used = []
        if request.use_rag and file_context:
            data_sources_used.append("rag")
        
        # Check if tools were called (this is a simplified check)
        # In a real implementation, you'd track tool calls from the agent result
        if hasattr(result, 'tool_calls') and result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name = tool_call.get('name', '') if isinstance(tool_call, dict) else str(tool_call)
                if 'postgres' in tool_name.lower() or 'railway' in tool_name.lower():
                    if 'postgres' not in data_sources_used:
                        data_sources_used.append("postgres")
                elif 'neon' in tool_name.lower():
                    if 'neon_db' not in data_sources_used:
                        data_sources_used.append("neon_db")
                elif 'internet' in tool_name.lower() or 'search' in tool_name.lower():
                    if 'internet' not in data_sources_used:
                        data_sources_used.append("internet")
        
        # Build structured response
        response_data = ChatResponse(
            answer=response_text,
            sources=file_context,
            confidence=0.8,  # Default confidence
            data_sources_used=data_sources_used if data_sources_used else ["rag"] if file_context else []
        )
        
        # Save message to database with data source tracking
        if railway_db:
            try:
                # Get or create session in DB
                session_db_id = await railway_db.fetchval(
                    "SELECT id FROM chat_sessions WHERE session_id = $1",
                    session_id
                )
                
                if not session_db_id:
                    session_db_id = await railway_db.fetchval(
                        """
                        INSERT INTO chat_sessions (session_id, is_active, message_count)
                        VALUES ($1, $2, $3)
                        RETURNING id
                        """,
                        session_id,
                        True,
                        0
                    )
                
                # Save user message
                await railway_db.execute(
                    """
                    INSERT INTO chat_messages (session_id, role, content, used_rag, used_postgres, used_neon_db, used_internet_search)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    session_db_id,
                    "user",
                    request.message,
                    "rag" in data_sources_used,
                    "postgres" in data_sources_used,
                    "neon_db" in data_sources_used,
                    "internet" in data_sources_used
                )
                
                # Save assistant message
                await railway_db.execute(
                    """
                    INSERT INTO chat_messages (session_id, role, content, used_rag, used_postgres, used_neon_db, used_internet_search, confidence_score, sources, usage_info)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """,
                    session_db_id,
                    "assistant",
                    response_data.answer,
                    "rag" in data_sources_used,
                    "postgres" in data_sources_used,
                    "neon_db" in data_sources_used,
                    "internet" in data_sources_used,
                    response_data.confidence,
                    json.dumps([{"file_name": s.file_name, "relevance_score": s.relevance_score} for s in response_data.sources]),
                    json.dumps(usage_info) if usage_info else None
                )
                
                # Update session message count
                await railway_db.execute(
                    "UPDATE chat_sessions SET message_count = message_count + 2, last_activity_at = CURRENT_TIMESTAMP WHERE id = $1",
                    session_db_id
                )
            except Exception as e:
                logger.warning(f"Failed to save chat message to database: {e}")
        
        # Update session history
        session["messages"].append({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.utcnow().isoformat()
        })
        session["messages"].append({
            "role": "assistant",
            "content": response_data.answer,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Extract usage information
        usage_info = None
        if hasattr(result, 'usage') and result.usage:
            usage_info = {
                "input_tokens": getattr(result.usage, 'input_tokens', 0),
                "output_tokens": getattr(result.usage, 'output_tokens', 0),
            }
        
        return ChatSessionResponse(
            session_id=session_id,
            message=request.message,
            response=response_data,
            usage=usage_info,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error processing chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@app.get("/sessions")
async def list_sessions():
    """List all active chat sessions."""
    return {
        "sessions": [
            {
                "session_id": sid,
                "created_at": session["created_at"],
                "message_count": len(session["messages"])
            }
            for sid, session in sessions.items()
        ]
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session."""
    if session_id in sessions:
        del sessions[session_id]
        return {"success": True, "message": f"Session {session_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/sessions/{session_id}/review")
async def review_response(session_id: str, review: HumanReviewRequest):
    """
    Human-in-the-loop review endpoint.
    
    Args:
        session_id: Session ID
        review: Review request with approval status and feedback
    
    Returns:
        Confirmation response
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Store review in session
    if "reviews" not in session:
        session["reviews"] = []
    
    session["reviews"].append({
        "approved": review.approved,
        "feedback": review.feedback,
        "corrected_answer": review.corrected_answer,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    # If webhook is configured, send notification
    webhook_url = os.getenv("HUMAN_IN_THE_LOOP_WEBHOOK_URL")
    if webhook_url and os.getenv("HUMAN_IN_THE_LOOP_ENABLED", "false").lower() == "true":
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(
                    webhook_url,
                    json={
                        "session_id": session_id,
                        "review": review.model_dump()
                    },
                    timeout=5.0
                )
        except Exception as e:
            logger.warning(f"Failed to send webhook: {e}")
    
    return {
        "success": True,
        "message": "Review recorded",
        "session_id": session_id
    }


if __name__ == "__main__":
    import uvicorn
    # Railway sets PORT, fallback to 8003
    # Use service-specific port variable, fallback to Railway PORT, then default
    port = int(os.getenv("CHATBOT_ORCH_PORT", os.getenv("PORT", "8003")))
    uvicorn.run(app, host="0.0.0.0", port=port)
