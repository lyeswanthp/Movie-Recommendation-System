"""LangGraph workflow definition for the movie recommendation system.

This module defines a directed acyclic graph (DAG) of nodes that together
perform retrieval‑augmented generation (RAG) for movie recommendations.
We build on the data loader defined in :mod:`backend.data_loader` and
optionally integrate predictions from a MindsDB model via
:mod:`backend.mindsdb_client`. The graph can be invoked programmatically
from the API to produce recommendations for a given user query.
"""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from langgraph.graph import Graph, Node
    from langgraph.graph.prompt import PromptNode  # noqa: F401 – imported for completeness
except ImportError:
    # Provide minimal fallbacks when LangGraph is not installed.
    class Node:
        """Minimal base class for nodes in the graph.

        Subclasses should override `run` to implement behavior.
        """

        def __init__(self) -> None:
            pass

        def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[name-defined]
            raise NotImplementedError("Override run() in Node subclasses")


    class Graph:
        """Fallback graph implementation that executes nodes sequentially.

        This simplistic graph does not support asynchronous execution or true
        parallelism but preserves the logical order defined via `connect`.
        Nodes are executed in the order defined by connections starting from
        the start node until reaching the end node.
        """

        def __init__(self) -> None:
            self.nodes: Dict[str, Node] = {}
            self.edges: Dict[str, List[str]] = {}
            self.start_node: str | None = None
            self.end_node: str | None = None

        def add_nodes(self, nodes: List[tuple[str, Node]]) -> None:
            for name, node in nodes:
                self.nodes[name] = node
                self.edges.setdefault(name, [])

        def connect(self, from_node: str, to_node: str) -> None:
            self.edges.setdefault(from_node, []).append(to_node)

        def set_start(self, node_name: str) -> None:
            self.start_node = node_name

        def set_end(self, node_name: str) -> None:
            self.end_node = node_name

        def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[name-defined]
            if self.start_node is None or self.end_node is None:
                raise RuntimeError("Graph must have start and end nodes defined.")
            # We'll traverse the graph by following the edges from the start node until we reach the end.
            current_data = dict(inputs)
            visited = set()
            node_queue = [self.start_node]
            while node_queue:
                node_name = node_queue.pop(0)
                if node_name in visited:
                    continue
                visited.add(node_name)
                node = self.nodes[node_name]
                result = node.run(current_data)
                # Merge result into current_data for subsequent nodes
                if isinstance(result, dict):
                    current_data.update(result)
                # If this node is the end node, return current_data
                if node_name == self.end_node:
                    return current_data
                # Enqueue connected nodes
                node_queue.extend(self.edges.get(node_name, []))
            return current_data

from data_loader import MovieDataLoader, Movie
from mindsdb_client import MindsDBClient


# Define a node to load and prepare the dataset (executed once on startup)
class DataInitNode(Node):
    def __init__(self, loader: MovieDataLoader):
        super().__init__()
        self.loader = loader

    def run(self, _: Dict[str, Any]) -> Dict[str, Any]:
        # Load data and build vectorizer if not already done
        if self.loader.df is None:
            self.loader.load_data()
            self.loader.build_vectorizer()
        return {}


# Node to perform retrieval based on user query
class RetrievalNode(Node):
    def __init__(self, loader: MovieDataLoader):
        super().__init__()
        self.loader = loader

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs.get("query")
        search_mode = inputs.get("mode", "description")
        top_n = inputs.get("top_n", 5)
        if search_mode == "description":
            movies = self.loader.search_by_description(query, top_n)
        elif search_mode == "title":
            movies = self.loader.search_by_title(query, top_n)
        else:
            raise ValueError(f"Unknown search mode: {search_mode}")
        return {"retrieved_movies": movies}


# Node to query MindsDB for additional signals (optional)
class MindsDBNode(Node):
    def __init__(self, client: MindsDBClient):
        super().__init__()
        self.client = client

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        description = inputs.get("query")
        top_n = inputs.get("top_n", 5)
        try:
            recommendations = self.client.predict_similar(description, top_n)
        except Exception:
            # If connection fails or the model is unavailable, return empty list
            recommendations = []
        return {"mindsdb_recommendations": recommendations}


# Node to assemble final response
class ResponseNode(Node):
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        retrieved_movies: List[Movie] = inputs.get("retrieved_movies", [])
        mindsdb_recs: List[dict] = inputs.get("mindsdb_recommendations", [])
        # Convert Movie objects to simple dictionaries
        retrieved_dicts = [m.to_dict() for m in retrieved_movies]
        return {
            "recommendations": {
                "local_retrieval": retrieved_dicts,
                "mindsdb": mindsdb_recs,
            }
        }


def build_graph(loader: MovieDataLoader, mindsdb_client: MindsDBClient) -> Graph:
    """Create and return the LangGraph for movie recommendations.

    The graph wires together the data initialization, retrieval, MindsDB query,
    and response assembly nodes. The resulting graph can be invoked with a
    dictionary containing the keys `query`, `mode`, and optionally `top_n`.

    Args:
        loader (MovieDataLoader): Initialized data loader for movie metadata.
        mindsdb_client (MindsDBClient): Optional client for MindsDB predictions.

    Returns:
        Graph: A LangGraph graph that processes user inputs into recommendations.
    """
    g = Graph()
    data_init = DataInitNode(loader)
    retrieve = RetrievalNode(loader)
    mindsdb_node = MindsDBNode(mindsdb_client)
    response_node = ResponseNode()

    # Add nodes to the graph
    g.add_nodes([
        ("init", data_init),
        ("retrieve", retrieve),
        ("mindsdb", mindsdb_node),
        ("response", response_node),
    ])

    # Edges define execution order
    # Step 1: initialize data
    g.connect("init", "retrieve")
    # Step 2: run retrieval and MindsDB queries in parallel
    g.connect("retrieve", "response")
    g.connect("init", "mindsdb")
    g.connect("mindsdb", "response")

    g.set_start("init")
    g.set_end("response")
    return g