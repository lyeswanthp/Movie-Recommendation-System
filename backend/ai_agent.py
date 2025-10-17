"""Advanced AI Agent using LangGraph and Ollama.

This module implements a sophisticated multi-agent system using LangGraph
for orchestration and Ollama for local LLM inference (no API keys needed).
"""

from __future__ import annotations

from typing import Dict, Any, List, Annotated, TypedDict, Literal
import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from vector_store import VectorStore, Movie


# Define the state for our agent
class AgentState(TypedDict):
    """State that flows through the agent graph."""
    messages: Annotated[List[BaseMessage], "The conversation history"]
    query: str
    mode: Literal["description", "title"]
    top_n: int
    retrieved_movies: List[Movie]
    recommendations: Dict[str, Any]
    intent: str


class MovieRecommenderAgent:
    """Advanced movie recommender using LangGraph and Ollama.

    This agent uses a multi-step reasoning process:
    1. Intent Classification: Determine if user wants recommendations by description or title
    2. Retrieval: Use vector store to find relevant movies
    3. Reasoning: Use LLM to analyze and explain recommendations
    4. Response Generation: Create a natural language response with explanations
    """

    def __init__(
        self,
        vector_store: VectorStore,
        ollama_model: str = "llama3.2:latest",
        ollama_base_url: str = "http://localhost:11434"
    ):
        """Initialize the agent.

        Args:
            vector_store: Vector store for movie retrieval
            ollama_model: Ollama model to use (llama3.2, mistral, etc.)
            ollama_base_url: Base URL for Ollama API
        """
        self.vector_store = vector_store
        self.ollama_model = ollama_model

        # Initialize Ollama LLM
        print(f"Initializing Ollama with model: {ollama_model}")
        self.llm = Ollama(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=0.7
        )

        # Build the agent graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph agent workflow."""

        # Define nodes
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("classify_intent", self._classify_intent)
        workflow.add_node("retrieve_movies", self._retrieve_movies)
        workflow.add_node("analyze_recommendations", self._analyze_recommendations)
        workflow.add_node("generate_response", self._generate_response)

        # Define edges
        workflow.add_edge(START, "classify_intent")
        workflow.add_edge("classify_intent", "retrieve_movies")
        workflow.add_edge("retrieve_movies", "analyze_recommendations")
        workflow.add_edge("analyze_recommendations", "generate_response")
        workflow.add_edge("generate_response", END)

        # Compile the graph with memory
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def _classify_intent(self, state: AgentState) -> AgentState:
        """Node 1: Classify user intent using LLM."""
        query = state["query"]
        mode = state.get("mode", "description")

        # If mode is explicitly set by API, use it
        if mode:
            state["intent"] = mode
            return state

        # Otherwise, use LLM to classify intent
        prompt = f"""Analyze this user query and determine if they are:
A) Describing their ideal movie (description mode)
B) Providing a specific movie title they like (title mode)

User query: "{query}"

Respond with only one word: "description" or "title"
"""

        try:
            response = self.llm.invoke(prompt).strip().lower()
            state["intent"] = "title" if "title" in response else "description"
        except Exception as e:
            print(f"Intent classification failed: {e}")
            state["intent"] = "description"  # Default fallback

        return state

    def _retrieve_movies(self, state: AgentState) -> AgentState:
        """Node 2: Retrieve relevant movies from vector store."""
        query = state["query"]
        intent = state.get("intent", "description")
        top_n = state.get("top_n", 5)

        try:
            if intent == "title":
                # Search by title
                movies = self.vector_store.search_by_title(query, top_n=top_n)
            else:
                # Search by description
                movies = self.vector_store.search_by_description(query, top_n=top_n)

            state["retrieved_movies"] = movies
        except Exception as e:
            print(f"Retrieval error: {e}")
            state["retrieved_movies"] = []

        return state

    def _analyze_recommendations(self, state: AgentState) -> AgentState:
        """Node 3: Use LLM to analyze and explain recommendations."""
        query = state["query"]
        movies = state.get("retrieved_movies", [])

        if not movies:
            state["recommendations"] = {
                "movies": [],
                "explanation": "No movies found matching your criteria."
            }
            return state

        # Build context for LLM
        movies_context = "\n\n".join([
            f"Movie {i+1}: {movie.title}\n"
            f"Genres: {', '.join(movie.genres)}\n"
            f"Description: {movie.overview}\n"
            f"Similarity Score: {movie.similarity_score:.2f}"
            for i, movie in enumerate(movies)
        ])

        prompt = f"""You are an expert movie recommender. Based on the user's request and the retrieved movies, provide intelligent recommendations with explanations.

User Request: "{query}"

Retrieved Movies:
{movies_context}

Task:
1. Analyze why each movie matches the user's request
2. Highlight key themes, genres, and elements that connect to the user's preferences
3. Provide brief, engaging explanations for why the user would enjoy each movie

Format your response as a JSON object with this structure:
{{
    "overall_analysis": "A brief overview of the recommendations",
    "movie_explanations": [
        {{
            "title": "Movie Title",
            "reason": "Why this movie matches the request"
        }}
    ]
}}

Respond with ONLY the JSON object, no additional text.
"""

        try:
            response = self.llm.invoke(prompt)
            # Try to parse JSON from response
            # LLMs sometimes add extra text, so we'll extract JSON
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            analysis = json.loads(response)
            state["analysis"] = analysis
        except Exception as e:
            print(f"Analysis error: {e}")
            state["analysis"] = {
                "overall_analysis": "Here are some great movie recommendations based on your preferences.",
                "movie_explanations": [
                    {"title": m.title, "reason": f"This movie matches your preferences with a similarity score of {m.similarity_score:.2f}"}
                    for m in movies
                ]
            }

        return state

    def _generate_response(self, state: AgentState) -> AgentState:
        """Node 4: Generate final response with recommendations."""
        movies = state.get("retrieved_movies", [])
        analysis = state.get("analysis", {})

        # Build movie recommendations with explanations
        movie_recommendations = []
        explanations = {exp["title"]: exp["reason"] for exp in analysis.get("movie_explanations", [])}

        for movie in movies:
            movie_dict = movie.to_dict()
            movie_dict["ai_explanation"] = explanations.get(movie.title, "A great match for your preferences!")
            movie_recommendations.append(movie_dict)

        state["recommendations"] = {
            "summary": analysis.get("overall_analysis", "Here are your personalized recommendations."),
            "movies": movie_recommendations,
            "total_found": len(movies)
        }

        return state

    def recommend(
        self,
        query: str,
        mode: str = "description",
        top_n: int = 5,
        thread_id: str = "default"
    ) -> Dict[str, Any]:
        """Get movie recommendations using the AI agent.

        Args:
            query: User's movie description or title
            mode: "description" or "title"
            top_n: Number of recommendations to return
            thread_id: Thread ID for conversation memory

        Returns:
            Dictionary with recommendations and AI-generated explanations
        """
        # Create initial state
        initial_state = AgentState(
            messages=[HumanMessage(content=query)],
            query=query,
            mode=mode,
            top_n=top_n,
            retrieved_movies=[],
            recommendations={},
            intent=""
        )

        # Run the graph
        config = {"configurable": {"thread_id": thread_id}}
        final_state = self.graph.invoke(initial_state, config=config)

        return final_state["recommendations"]

    def chat(
        self,
        message: str,
        thread_id: str = "default"
    ) -> str:
        """Have a conversational interaction with the agent.

        Args:
            message: User message
            thread_id: Thread ID for conversation memory

        Returns:
            AI response as string
        """
        result = self.recommend(
            query=message,
            mode="description",
            top_n=5,
            thread_id=thread_id
        )

        # Format as natural language response
        summary = result.get("summary", "")
        movies = result.get("movies", [])

        response = f"{summary}\n\n"
        for i, movie in enumerate(movies, 1):
            response += f"{i}. **{movie['title']}** ({', '.join(movie['genres'])})\n"
            response += f"   {movie.get('ai_explanation', '')}\n\n"

        return response.strip()


def check_ollama_availability(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama is running and accessible.

    Args:
        base_url: Base URL for Ollama API

    Returns:
        True if Ollama is available, False otherwise
    """
    import httpx

    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def get_available_models(base_url: str = "http://localhost:11434") -> List[str]:
    """Get list of available Ollama models.

    Args:
        base_url: Base URL for Ollama API

    Returns:
        List of model names
    """
    import httpx

    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        return []
    except Exception:
        return []
