"""Advanced vector store using ChromaDB and Sentence Transformers.

This module provides semantic search capabilities using state-of-the-art
embeddings from Sentence Transformers and ChromaDB vector database.
"""

from __future__ import annotations

import os
from typing import List, Optional, Dict, Any
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass


@dataclass
class Movie:
    """Enhanced movie data class with vector search capabilities."""

    movie_id: str
    title: str
    overview: str
    genres: List[str]
    similarity_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "movie_id": self.movie_id,
            "title": self.title,
            "overview": self.overview,
            "genres": self.genres,
            "similarity_score": self.similarity_score,
        }


class VectorStore:
    """Advanced vector store using ChromaDB and Sentence Transformers.

    Uses 'all-MiniLM-L6-v2' model for fast, high-quality embeddings.
    This model achieves 68% accuracy on semantic similarity tasks.
    """

    def __init__(
        self,
        collection_name: str = "movies",
        persist_directory: str = "./chroma_db",
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            model_name: Sentence transformer model to use for embeddings
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.model_name = model_name

        # Initialize Sentence Transformer model
        print(f"Loading embedding model: {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Created new collection: {collection_name}")

    def load_movies_from_csv(self, csv_path: str) -> None:
        """Load movies from CSV and create embeddings.

        Args:
            csv_path: Path to the movie metadata CSV file
        """
        print(f"Loading movies from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Check if collection is already populated
        existing_count = self.collection.count()
        if existing_count > 0:
            print(f"Collection already contains {existing_count} movies. Skipping load.")
            return

        # Prepare data
        df["overview"] = df["overview"].fillna("")

        # Parse genres
        def parse_genres(genre_str: str) -> List[str]:
            if isinstance(genre_str, str):
                return [g.strip() for g in genre_str.split('|')]
            return []

        df["genres_list"] = df["genres"].apply(parse_genres)

        # Create rich text for embedding (title + overview + genres)
        def create_embedding_text(row):
            genres_text = ", ".join(row["genres_list"])
            return f"Title: {row['title']}. {row['overview']}. Genres: {genres_text}"

        df["embedding_text"] = df.apply(create_embedding_text, axis=1)

        # Generate embeddings in batches
        batch_size = 100
        total_movies = len(df)

        print(f"Generating embeddings for {total_movies} movies...")

        for i in range(0, total_movies, batch_size):
            batch_df = df.iloc[i:i+batch_size]
            texts = batch_df["embedding_text"].tolist()

            # Generate embeddings
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

            # Prepare data for ChromaDB
            ids = [str(row["id"]) for _, row in batch_df.iterrows()]
            metadatas = [
                {
                    "title": row["title"],
                    "genres": "|".join(row["genres_list"]),
                    "overview": row["overview"][:500]  # Limit metadata size
                }
                for _, row in batch_df.iterrows()
            ]

            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

            print(f"Processed {min(i+batch_size, total_movies)}/{total_movies} movies")

        print(f"✓ Successfully loaded {total_movies} movies into vector store!")

    def search_by_description(
        self,
        query: str,
        top_n: int = 5,
        genre_filter: Optional[List[str]] = None
    ) -> List[Movie]:
        """Search for movies using semantic similarity.

        Args:
            query: Natural language description of desired movie
            top_n: Number of results to return
            genre_filter: Optional list of genres to filter by

        Returns:
            List of Movie objects sorted by similarity
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]

        # Build where clause for genre filtering
        where_clause = None
        if genre_filter:
            # ChromaDB doesn't support OR operations easily, so we'll filter after retrieval
            pass

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_n * 2 if genre_filter else top_n,  # Get more if filtering
            include=["metadatas", "documents", "distances"]
        )

        # Process results
        movies = []
        for i, (metadata, distance) in enumerate(zip(
            results["metadatas"][0],
            results["distances"][0]
        )):
            # Convert distance to similarity score (cosine distance to similarity)
            similarity = 1 - distance

            # Parse genres
            genres = metadata.get("genres", "").split("|")

            # Apply genre filter if specified
            if genre_filter:
                if not any(genre in genres for genre in genre_filter):
                    continue

            movie = Movie(
                movie_id=results["ids"][0][i],
                title=metadata.get("title", "Unknown"),
                overview=metadata.get("overview", ""),
                genres=genres,
                similarity_score=similarity
            )
            movies.append(movie)

            if len(movies) >= top_n:
                break

        return movies

    def search_by_title(
        self,
        title: str,
        top_n: int = 5
    ) -> List[Movie]:
        """Find similar movies based on a movie title.

        Args:
            title: Movie title to find similar movies for
            top_n: Number of similar movies to return

        Returns:
            List of similar Movie objects
        """
        # Use semantic search on the title to find the best matching movie
        # This is more robust than simple string matching
        title_embedding = self.embedding_model.encode([f"Title: {title}"])[0]

        # Search for the best matching title
        title_results = self.collection.query(
            query_embeddings=[title_embedding.tolist()],
            n_results=10,  # Get top 10 to find best title match
            include=["metadatas", "distances"]
        )

        # Find the movie with the closest title match
        best_match_idx = None
        best_match_score = 0

        for i, metadata in enumerate(title_results["metadatas"][0]):
            movie_title = metadata.get("title", "").lower()
            query_lower = title.lower()

            # Check for exact match first
            if query_lower == movie_title:
                best_match_idx = i
                break
            # Then check for substring match
            elif query_lower in movie_title or movie_title in query_lower:
                # Prefer closer semantic matches
                semantic_score = 1 - title_results["distances"][0][i]
                if semantic_score > best_match_score:
                    best_match_score = semantic_score
                    best_match_idx = i

        # If no substring match, use the closest semantic match
        if best_match_idx is None:
            best_match_idx = 0

        # Get the ID of the matched movie
        matched_movie_id = title_results["ids"][0][best_match_idx]
        matched_metadata = title_results["metadatas"][0][best_match_idx]

        # Get the full data including embedding for this movie
        matched_movie_data = self.collection.get(
            ids=[matched_movie_id],
            include=["embeddings", "metadatas"]
        )

        target_embedding = matched_movie_data["embeddings"][0]

        # Now find similar movies based on this embedding
        similar_results = self.collection.query(
            query_embeddings=[target_embedding],
            n_results=top_n + 1,  # +1 to potentially exclude the movie itself
            include=["metadatas", "documents", "distances"]
        )

        # Process results (skip the matched movie itself)
        movies = []
        for i in range(len(similar_results["ids"][0])):
            if similar_results["ids"][0][i] == matched_movie_id:
                continue  # Skip the source movie

            metadata = similar_results["metadatas"][0][i]
            distance = similar_results["distances"][0][i]
            similarity = 1 - distance

            movie = Movie(
                movie_id=similar_results["ids"][0][i],
                title=metadata.get("title", "Unknown"),
                overview=metadata.get("overview", ""),
                genres=metadata.get("genres", "").split("|"),
                similarity_score=similarity
            )
            movies.append(movie)

            if len(movies) >= top_n:
                break

        return movies

    def reset(self) -> None:
        """Reset the vector store (delete all data)."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print("Vector store reset successfully!")
