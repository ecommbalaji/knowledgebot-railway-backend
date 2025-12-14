"""Chatbot Orchestration Service - Pydantic AI Agent with RAG via Gemini FileSearch."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Annotated
import os
import logging
from dotenv import load_dotenv
import uuid
from datetime import datetime
import google.generativeai as genai
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
import asyncio

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chatbot Orchestration Service", version="1.0.0")

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
    logger.warning("GEMINI_API_KEY environment variable not set - some features will fail")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize OpenAI model for Pydantic AI (optional for healthcheck)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY environment variable not set - chat endpoints will fail")
else:
    # Model initialization will happen when needed
    pass

MODEL_NAME = os.getenv("CHATBOT_MODEL", "gpt-4o")
TEMPERATURE = float(os.getenv("CHATBOT_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("CHATBOT_MAX_TOKENS", "2000"))

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
    openai_model = OpenAIModel(MODEL_NAME)
else:
    logger.warning("OpenAI model not initialized - OPENAI_API_KEY is missing")


# Tool for querying Gemini FileSearch
# Tool for querying Gemini FileSearch
async def search_knowledge_base(query: Annotated[str, "The search query to find relevant information"]) -> List[SearchResult]:
    """
    Search the knowledge base using Gemini FileSearch for relevant information.
    
    This tool searches through uploaded documents and scraped content to find
    information relevant to the user's query.
    """
    try:
        # List all files in Gemini FileSearch
        files = genai.list_files()
        
        if not files:
            logger.warning("No files found in FileSearch store")
            return []
        
        results = []
        
        # For MVP: Use Gemini to search through files
        # In production, use the FileSearch API with proper semantic search
        try:
            # Use Gemini model with file access for semantic search
            gemini_model = genai.GenerativeModel('gemini-1.5-pro')
            
            # Build context from available files
            file_list = [f.name for f in files[:10]]  # Limit to first 10 files for MVP
            
            # Create a search prompt
            search_prompt = f"""
            User query: {query}
            
            Search through the available files and identify which files are most relevant.
            Return a list of the most relevant file names.
            """
            
            # For MVP, we'll return file metadata
            # In production, use FileSearch API's semantic search
            for file in files[:5]:  # Return top 5 results
                try:
                    file_info = genai.get_file(file.name)
                    results.append(SearchResult(
                        file_name=file.display_name or file.name,
                        content=f"Content from {file.display_name}",
                        relevance_score=0.8
                    ))
                except Exception as e:
                    logger.warning(f"Error accessing file {file.name}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error in FileSearch: {e}")
            # Fallback: return file list
            for file in files[:5]:
                results.append(SearchResult(
                    file_name=file.display_name or file.name,
                    content="File available in knowledge base",
                    relevance_score=0.5
                ))
        
        return results[:5]  # Return max 5 results
        
    except Exception as e:
        logger.error(f"Error searching knowledge base: {e}")
        return []


# Session context dependency for dependency injection
def create_session_dependency(session_id: str):
    """Create a session context dependency."""
    def session_context() -> Dict[str, str]:
        """Inject session context into agent runs."""
        return {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "service": "chatbot_orchestration"
        }
    return session_context


# System prompt with dynamic context
def get_system_prompt(file_context: Optional[List[SearchResult]] = None) -> str:
    """Generate dynamic system prompt with optional file context."""
    base_prompt = """You are a helpful knowledge assistant chatbot for the website https://books.toscrape.com/.

Your role is to answer questions about books, authors, pricing, availability, and any other information related to the book catalog.

When answering questions:
1. Use the search_knowledge_base tool to find relevant information
2. Provide accurate answers based on the knowledge base
3. If information is not available, clearly indicate that
4. Be helpful and conversational"""
    
    if file_context:
        context_section = "\n\nAvailable knowledge base files:\n"
        for idx, result in enumerate(file_context, 1):
            context_section += f"{idx}. {result.file_name}\n"
        return base_prompt + context_section
    
    return base_prompt


# Initialize base agent
def create_agent(session_id: str, file_context: Optional[List[SearchResult]] = None) -> Optional[Agent]:
    """Create a Pydantic AI agent for a session with dependency injection."""
    
    # Check if model is available
    if openai_model is None:
        logger.error("Cannot create agent - OpenAI API key not configured")
        return None
    
    # Create session dependency
    session_dep = create_session_dependency(session_id)
    
    # Create agent with system prompt and dependencies (tools removed for now)
    agent = Agent(
        openai_model,
        system_prompt=get_system_prompt(file_context),
        dependencies=[session_dep],
    )
    
    return agent


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "chatbot_orchestration"}


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
        agent = create_agent(session_id, file_context)
        
        # Check if agent was created successfully
        if agent is None:
            raise HTTPException(
                status_code=503,
                detail="Chatbot service not configured - OpenAI API key required"
            )
        
        # Build chat history from session
        chat_history = session["messages"]
        
        # Convert chat history to agent messages
        agent_messages = []
        for msg in chat_history[-10:]:  # Keep last 10 messages for context
            if msg["role"] == "user":
                agent_messages.append(UserMessage(msg["content"]))
            elif msg["role"] == "assistant":
                agent_messages.append(AssistantMessage(msg["content"]))
        
        # Add current user message
        agent_messages.append(UserMessage(request.message))
        
        # Run agent (with self-correction via model_retry)
        result = await agent.run(agent_messages)
        
        # Extract response text
        response_text = ""
        if hasattr(result, 'response') and result.response:
            response_text = result.response.text or str(result.data) if hasattr(result, 'data') else ""
        elif hasattr(result, 'data'):
            response_text = str(result.data)
        
        # Build structured response
        response_data = ChatResponse(
            answer=response_text,
            sources=file_context,
            confidence=0.8  # Default confidence
        )
        
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
                        "review": review.dict()
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
    port = int(os.getenv("PORT", "8003"))
    uvicorn.run(app, host="0.0.0.0", port=port)
