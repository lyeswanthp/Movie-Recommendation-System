"""Advanced FastAPI application with AI agents and vector search.

This module provides a state-of-the-art movie recommendation API using:
- ChromaDB vector database with Sentence Transformers
- LangGraph multi-agent system
- Ollama for local LLM inference (no API keys!)
- Conversational memory and context
"""

from __future__ import annotations

import os
import sys

# Fix Windows console encoding for emoji support
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from vector_store import VectorStore
from ai_agent import MovieRecommenderAgent, check_ollama_availability, get_available_models


# Global instances
vector_store: Optional[VectorStore] = None
ai_agent: Optional[MovieRecommenderAgent] = None
ollama_available: bool = False


# Request/Response models
class RecommendationRequest(BaseModel):
    query: str
    top_n: int = 5
    mode: str = "description"  # "description" or "title"


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class StatusResponse(BaseModel):
    status: str
    ollama_available: bool
    ollama_models: list
    vector_store_count: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI application."""
    global vector_store, ai_agent, ollama_available

    print("🚀 Starting Advanced Movie Recommender System...")

    # Initialize vector store
    print("📚 Initializing vector database...")
    data_path = os.path.join(os.path.dirname(__file__), "data", "movielens_metadata.csv")

    vector_store = VectorStore(
        collection_name="movies_advanced",
        persist_directory="./chroma_db",
        model_name="all-MiniLM-L6-v2"
    )

    # Load movies into vector store (if not already loaded)
    try:
        vector_store.load_movies_from_csv(data_path)
    except Exception as e:
        print(f"Warning: Could not load movies: {e}")

    # Check Ollama availability
    print("🤖 Checking Ollama availability...")
    ollama_available = check_ollama_availability()

    if ollama_available:
        available_models = get_available_models()
        print(f"✓ Ollama is available with models: {available_models}")

        # Try to use llama3.2 first, fallback to other models
        model_to_use = "llama3.2:latest"
        if not any("llama3.2" in m for m in available_models):
            if available_models:
                model_to_use = available_models[0]
                print(f"⚠ llama3.2 not found, using {model_to_use} instead")
            else:
                print("⚠ No Ollama models found! AI agent will not be available.")
                print("  Install a model with: ollama pull llama3.2")
                ollama_available = False

        if ollama_available:
            print(f"🧠 Initializing AI Agent with {model_to_use}...")
            try:
                ai_agent = MovieRecommenderAgent(
                    vector_store=vector_store,
                    ollama_model=model_to_use
                )
                print("✓ AI Agent initialized successfully!")
            except Exception as e:
                print(f"⚠ Could not initialize AI agent: {e}")
                ollama_available = False
    else:
        print("⚠ Ollama is not running!")
        print("  To enable AI-powered explanations:")
        print("  1. Install Ollama: https://ollama.ai")
        print("  2. Run: ollama pull llama3.2")
        print("  3. Start Ollama")
        print("  The system will work with vector search only (no AI explanations)")

    print("\n✨ System ready!")
    print(f"   Vector Store: {vector_store.collection.count()} movies indexed")
    print(f"   AI Agent: {'Enabled' if ollama_available else 'Disabled (vector search only)'}\n")

    yield

    # Cleanup
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Advanced Movie Recommendation API",
    version="2.0.0",
    description="AI-powered movie recommendations with vector search and LangGraph agents",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root() -> Dict[str, str]:
    """Root endpoint with welcome message."""
    return {
        "message": "🎬 Welcome to the Advanced Movie Recommendation API!",
        "docs": "/docs",
        "status": "/status"
    }


@app.get("/status")
async def get_status() -> StatusResponse:
    """Get system status."""
    global vector_store, ollama_available

    models = get_available_models() if ollama_available else []

    return StatusResponse(
        status="operational",
        ollama_available=ollama_available,
        ollama_models=models,
        vector_store_count=vector_store.collection.count() if vector_store else 0
    )


@app.post("/recommend")
async def recommend_movies(request: RecommendationRequest) -> Dict[str, Any]:
    """Get movie recommendations.

    If Ollama is available, uses AI agent for intelligent recommendations.
    Otherwise, falls back to pure vector search.
    """
    global vector_store, ai_agent, ollama_available

    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    try:
        if ollama_available and ai_agent:
            # Use AI agent for intelligent recommendations
            result = ai_agent.recommend(
                query=request.query,
                mode=request.mode,
                top_n=request.top_n
            )
            return {
                "mode": "ai_powered",
                "recommendations": result
            }
        else:
            # Fallback to vector search only
            if request.mode == "title":
                movies = vector_store.search_by_title(request.query, top_n=request.top_n)
            else:
                movies = vector_store.search_by_description(request.query, top_n=request.top_n)

            return {
                "mode": "vector_search",
                "recommendations": {
                    "summary": f"Found {len(movies)} movies matching your request (vector search mode)",
                    "movies": [m.to_dict() for m in movies],
                    "total_found": len(movies)
                }
            }
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")


@app.post("/recommend/description")
async def recommend_by_description(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy endpoint for description-based recommendations."""
    query = payload.get("query")
    top_n = payload.get("top_n", 5)

    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' in request body.")

    request = RecommendationRequest(query=query, top_n=top_n, mode="description")
    return await recommend_movies(request)


@app.post("/recommend/title")
async def recommend_by_title(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy endpoint for title-based recommendations."""
    title = payload.get("title")
    top_n = payload.get("top_n", 5)

    if not title:
        raise HTTPException(status_code=400, detail="Missing 'title' in request body.")

    request = RecommendationRequest(query=title, top_n=top_n, mode="title")
    result = await recommend_movies(request)

    # Format for legacy frontend compatibility
    movies = result.get("recommendations", {}).get("movies", [])
    return {
        "local_retrieval": movies,
        "mindsdb": []
    }


@app.post("/chat")
async def chat(request: ChatRequest) -> Dict[str, str]:
    """Conversational endpoint for chatting with the AI agent."""
    global ai_agent, ollama_available

    if not ollama_available or not ai_agent:
        raise HTTPException(
            status_code=503,
            detail="AI agent not available. Please ensure Ollama is running with a model installed."
        )

    try:
        response = ai_agent.chat(
            message=request.message,
            thread_id=request.thread_id
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.post("/initialize")
async def initialize_database(background_tasks: BackgroundTasks) -> Dict[str, str]:
    """Reinitialize the vector database (admin endpoint)."""
    global vector_store

    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not available")

    def reinitialize():
        data_path = os.path.join(os.path.dirname(__file__), "data", "movielens_metadata.csv")
        vector_store.reset()
        vector_store.load_movies_from_csv(data_path)

    background_tasks.add_task(reinitialize)

    return {"message": "Database reinitialization started in background"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
