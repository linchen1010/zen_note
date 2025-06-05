"""
FastAPI backend for Zen Note RAG application
"""

import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import BACKEND_HOST, BACKEND_PORT, CORS_ORIGINS
from rag import rag_system
from model_runner import model_runner

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Pydantic models
class QuestionRequest(BaseModel):
    """Request model for asking questions"""

    question: str = Field(
        ..., min_length=1, max_length=1000, description="The question to ask"
    )
    max_chunks: Optional[int] = Field(
        5, ge=1, le=20, description="Maximum number of chunks to retrieve"
    )
    score_threshold: Optional[float] = Field(
        0.3, ge=0.0, le=1.0, description="Minimum similarity score for chunks"
    )
    temperature: Optional[float] = Field(
        0.7, ge=0.0, le=2.0, description="LLM temperature setting"
    )


class QuestionResponse(BaseModel):
    """Response model for questions"""

    success: bool
    answer: Optional[str] = None
    sources: Optional[list] = None
    metadata: Optional[dict] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health checks"""

    status: str
    components: Optional[dict] = None
    error: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for the application"""
    # Startup
    logger.info("🚀 Starting Zen Note RAG API")

    # Perform health checks on startup
    try:
        health = rag_system.health_check()
        if health["status"] != "healthy":
            logger.warning(f"⚠️  System health check: {health['status']}")
        else:
            logger.info("✅ All systems healthy")
    except Exception as e:
        logger.error(f"❌ Startup health check failed: {e}")

    yield

    # Shutdown
    logger.info("🛑 Shutting down Zen Note RAG API")


# Create FastAPI app
app = FastAPI(
    title="Zen Note RAG API",
    description="Local RAG API for querying markdown knowledge base using local LLM",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Zen Note RAG API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint

    Returns the status of all system components:
    - Vector store (FAISS index)
    - Embedding model
    - Local LLM (Ollama)
    """
    try:
        health = rag_system.health_check()
        return HealthResponse(**health)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(status="error", error=str(e))


@app.post("/ask", response_model=QuestionResponse, tags=["RAG"])
async def ask_question(request: QuestionRequest):
    """
    Ask a question to the RAG system

    This endpoint:
    1. Searches for relevant chunks in the vector store
    2. Builds context from retrieved chunks
    3. Generates an answer using the local LLM
    4. Returns the answer with source attribution

    Args:
        request: Question and optional parameters

    Returns:
        Answer with sources and metadata
    """
    try:
        logger.info(f"📝 Question received: '{request.question[:100]}...'")

        # Process the question through RAG pipeline
        result = rag_system.ask(
            question=request.question,
            max_chunks=request.max_chunks,
            score_threshold=request.score_threshold,
            temperature=request.temperature,
        )

        if result["success"]:
            logger.info(
                f"✅ Answer generated successfully with {len(result['sources'])} sources"
            )
        else:
            logger.error(f"❌ Failed to generate answer: {result['error']}")

        return QuestionResponse(**result)

    except Exception as e:
        error_msg = f"Unexpected error processing question: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/search", tags=["Search"])
async def search_chunks(
    query: str = Query(..., min_length=1, max_length=500, description="Search query"),
    k: int = Query(5, ge=1, le=20, description="Number of results to return"),
    score_threshold: float = Query(
        0.3, ge=0.0, le=1.0, description="Minimum similarity score"
    ),
):
    """
    Search for similar chunks without LLM generation

    Useful for debugging and understanding what content is being retrieved.

    Args:
        query: Search query
        k: Number of results to return
        score_threshold: Minimum similarity score

    Returns:
        List of similar chunks with scores and metadata
    """
    try:
        logger.info(f"🔍 Search query: '{query}'")

        chunks = rag_system.search_similar_chunks(
            query=query, k=k, score_threshold=score_threshold
        )

        return {
            "success": True,
            "query": query,
            "results": chunks,
            "count": len(chunks),
        }

    except Exception as e:
        error_msg = f"Search failed: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/status", tags=["Status"])
async def system_status():
    """
    Get detailed system status and statistics
    """
    try:
        health = rag_system.health_check()

        return {
            "api_status": "running",
            "system_health": health,
            "configuration": {
                "backend_host": BACKEND_HOST,
                "backend_port": BACKEND_PORT,
                "cors_origins": CORS_ORIGINS,
            },
        }

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    logger.info(f"🚀 Starting Zen Note RAG API on {BACKEND_HOST}:{BACKEND_PORT}")

    uvicorn.run(
        "main:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=True, log_level="info"
    )
