"""Chatbot Orchestration Service - Pydantic AI Agent with RAG via Gemini FileSearch."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Annotated
from typing import Optional, List, Dict, Any, Annotated
import os
import logging
from dataclasses import dataclass
from dotenv import load_dotenv
import uuid
from datetime import datetime
import google.generativeai as genai
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
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
        # Convert generator to list
        all_files = list(genai.list_files())
        
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
            # Initialize Gemini 2.5 Flash Lite for retrieval (cheaper, fast)
            # We treat it as a "Neural Retriever"
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            
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
            
            # Generate content using the model with the files attached
            # Gemini Python SDK allows passing file objects directly in the content list
            response = await model.generate_content_async(
                contents=[*files_to_search, retrieval_prompt]
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


# System prompt with dynamic context
def get_system_prompt(file_context: Optional[List[SearchResult]] = None) -> str:
    """Generate dynamic system prompt with optional file context."""
    base_prompt = """You are a helpful knowledge assistant chatbot.
    
    Your role is to answer questions based on the information provided in your knowledge base (files, scraped websites, etc.).
    
    When answering questions:
    1. Use the search_knowledge_base tool to find relevant information
    2. Provide accurate answers based on the knowledge base content
    3. If information is not available in the context, clearly indicate that
    4. Be helpful and conversational"""
    
    if file_context:
        context_section = "\n\nAvailable knowledge base files:\n"
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

# Initialize base agent
def create_agent(file_context: Optional[List[SearchResult]] = None) -> Optional[Agent]:
    """Create a Pydantic AI agent with dependency injection configuration."""
    
    # Check if model is available
    if openai_model is None:
        logger.error("Cannot create agent - OpenAI API key not configured")
        return None

    # Create agent with system prompt and dependencies configuration
    # We pass the TYPE of the dependency here, not an instance
    agent = Agent(
        openai_model,
        system_prompt=get_system_prompt(file_context),
        deps_type=ChatSessionDeps,
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
