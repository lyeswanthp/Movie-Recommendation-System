"""FastAPI application for the movie recommendation service.

This module sets up a RESTful API that exposes endpoints for generating
recommendations based on a user’s free‑form description or a specific
movie title. It leverages a LangGraph pipeline for retrieval‑augmented
recommendations and can optionally incorporate MindsDB predictions.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from data_loader import MovieDataLoader
from mindsdb_client import MindsDBClient
from graph import build_graph


# Determine the path to the movie metadata CSV file. Users can override this
# via an environment variable. For demonstration purposes, the file is expected
# to reside in a `data/` directory relative to this module.
DEFAULT_DATA_PATH = os.environ.get(
    "MOVIE_METADATA_PATH",
    os.path.join(os.path.dirname(__file__), "data", "movielens_metadata.csv"),
)


def create_app() -> FastAPI:
    """Factory to create and configure the FastAPI application."""
    app = FastAPI(title="Movie Recommendation API", version="1.0.0")

    # Enable CORS so that the frontend can communicate with the backend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify explicit origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize the loader and graph at application startup
    loader = MovieDataLoader(DEFAULT_DATA_PATH)
    mindsdb_client = MindsDBClient()
    graph = build_graph(loader, mindsdb_client)

    @app.get("/")
    async def read_root() -> Dict[str, str]:
        return {"message": "Welcome to the Movie Recommendation API!"}

    @app.post("/recommend/description")
    async def recommend_by_description(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend movies based on a free‑form description.

        Body parameters:
            query (str): User‑provided description of the desired movie.
            top_n (int, optional): Number of recommendations to return (default 5).
        """
        query = payload.get("query")
        top_n = payload.get("top_n", 5)
        if not query:
            raise HTTPException(status_code=400, detail="Missing 'query' in request body.")
        try:
            result = graph.run({"query": query, "mode": "description", "top_n": top_n})
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        # The graph returns a dictionary that may include intermediate keys. Extract
        # the final recommendations if present.
        return result.get("recommendations", result)

    @app.post("/recommend/title")
    async def recommend_by_title(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend movies based on a movie title.

        Body parameters:
            title (str): Title of a movie the user likes.
            top_n (int, optional): Number of recommendations to return (default 5).
        """
        title = payload.get("title")
        top_n = payload.get("top_n", 5)
        if not title:
            raise HTTPException(status_code=400, detail="Missing 'title' in request body.")
        try:
            result = graph.run({"query": title, "mode": "title", "top_n": top_n})
        except ValueError as ve:
            raise HTTPException(status_code=404, detail=str(ve)) from ve
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return result.get("recommendations", result)

    return app


app = create_app()