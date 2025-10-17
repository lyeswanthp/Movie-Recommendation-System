"""Utility functions for loading movie data and performing vector-based search.

This module provides helper functions to load movie metadata from a CSV file,
build a TF‑IDF vector representation of movie overviews, and perform similarity
searches over the resulting vectors. The retrieval functions are used by the
LangGraph pipeline to find relevant movies based on a user’s description or
selected title.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


@dataclass
class Movie:
    """Simple data class to store movie metadata used for recommendations."""

    movie_id: str
    title: str
    overview: str
    genres: List[str]
    additional_info: dict

    def to_dict(self) -> dict:
        return {
            "movie_id": self.movie_id,
            "title": self.title,
            "overview": self.overview,
            "genres": self.genres,
            **self.additional_info,
        }


class MovieDataLoader:
    """Loader to manage movie data and TF‑IDF vectorization for retrieval."""

    def __init__(self, metadata_path: str) -> None:
        self.metadata_path = metadata_path
        self.df: Optional[pd.DataFrame] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

    def load_data(self) -> pd.DataFrame:
        """Load movie metadata from a CSV file into a pandas DataFrame.

        The expected CSV should contain at least the following columns:
        - id (unique identifier for each movie)
        - title (movie title)
        - overview (text description of the movie)
        - genres (stringified list of genre objects)

        Returns:
            pd.DataFrame: The loaded DataFrame with preprocessed columns.
        """
        df = pd.read_csv(self.metadata_path)
        # Ensure required columns exist
        required_columns = {"id", "title", "overview", "genres"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in metadata file: {', '.join(missing)}"
            )
        # Fill NaNs in overview with empty strings to avoid vectorizer errors
        df["overview"] = df["overview"].fillna("")
        # Parse genres from pipe-separated strings to a list of names
        def parse_genres(genre_str: str) -> List[str]:
            if isinstance(genre_str, str):
                return [genre.strip() for genre in genre_str.split('|')]
            return [] # Return empty list for missing or non-string genre data

        df["genres_list"] = df["genres"].apply(parse_genres)
        self.df = df
        return df

    def build_vectorizer(self) -> None:
        """Build the TF‑IDF vectorizer and compute the document-term matrix."""
        if self.df is None:
            raise RuntimeError("Data must be loaded before building the vectorizer.")
        # Create a TF‑IDF vectorizer with English stop words removed
        self.vectorizer = TfidfVectorizer(stop_words="english")
        # Fit the vectorizer on the overview column and transform the data
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["overview"])

    def _ensure_ready(self) -> None:
        if self.df is None or self.vectorizer is None or self.tfidf_matrix is None:
            raise RuntimeError("Data and vectorizer must be initialized before searching.")

    def search_by_description(self, query: str, top_n: int = 5) -> List[Movie]:
        """Return a list of movies most similar to a textual description.

        Args:
            query (str): Free‑form description provided by the user.
            top_n (int): Number of results to return.

        Returns:
            List[Movie]: Sorted list of movie recommendations.
        """
        self._ensure_ready()
        # Transform the query to the same vector space
        query_vector = self.vectorizer.transform([query])
        # Compute cosine similarity
        cosine_similarities = linear_kernel(query_vector, self.tfidf_matrix).flatten()
        # Get indices of the top matches
        top_indices = cosine_similarities.argsort()[::-1][:top_n]
        results: List[Movie] = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            movie = Movie(
                movie_id=str(row["id"]),
                title=row["title"],
                overview=row["overview"],
                genres=row["genres_list"],
                additional_info={"cosine_score": float(cosine_similarities[idx])},
            )
            results.append(movie)
        return results

    def search_by_title(self, title: str, top_n: int = 5) -> List[Movie]:
        """Return a list of movies similar to a given title based on overview similarity.

        Args:
            title (str): The title of the movie the user likes.
            top_n (int): Number of results to return.

        Returns:
            List[Movie]: Sorted list of movie recommendations similar to the input title.
        """
        self._ensure_ready()
        # Attempt to locate the movie row by title (case insensitive)
        # First try exact match
        matches = self.df[self.df["title"].str.lower() == title.lower()]

        # If no exact match, try partial match (contains)
        if matches.empty:
            matches = self.df[self.df["title"].str.lower().str.contains(title.lower(), na=False)]

        if matches.empty:
            raise ValueError(f"Movie with title '{title}' not found in the dataset.")
        # Use the first match if multiple
        idx = matches.index[0]
        # Compute similarity of this movie to all others
        movie_vector = self.tfidf_matrix[idx]
        cosine_similarities = linear_kernel(movie_vector, self.tfidf_matrix).flatten()
        # Exclude the movie itself from the results
        cosine_similarities[idx] = -1
        top_indices = cosine_similarities.argsort()[::-1][:top_n]
        results: List[Movie] = []
        for i in top_indices:
            row = self.df.iloc[i]
            movie = Movie(
                movie_id=str(row["id"]),
                title=row["title"],
                overview=row["overview"],
                genres=row["genres_list"],
                additional_info={"cosine_score": float(cosine_similarities[i])},
            )
            results.append(movie)
        return results