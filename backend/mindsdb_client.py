"""Client for interacting with MindsDB models.

This module wraps the connection logic to a MindsDB server and exposes
convenience functions to run predictions against a trained recommender
model. While the actual model training happens inside MindsDB (typically
configured via their dashboard or SQL interface), this client demonstrates
how to send inputs and receive outputs programmatically.
"""

from __future__ import annotations

from typing import List, Dict, Any

try:
    from mindsdb_sdk import Session
except ImportError:
    # mindsdb_sdk is optional; the rest of the backend can function without it.
    Session = None  # type: ignore


class MindsDBClient:
    """Encapsulates the connection to a MindsDB instance and model invocation."""

    def __init__(self, host: str = "http://127.0.0.1", port: int = 47334,
                 api_key: str | None = None,
                 project_name: str = "mindsdb", model_name: str = "movie_recommender") -> None:
        """Initialize the client with connection details.

        Args:
            host (str): Hostname or IP where MindsDB server is running.
            port (int): Port number for the MindsDB API.
            api_key (str | None): Optional API key for cloud deployments.
            project_name (str): The name of the project containing your model.
            model_name (str): The name of the model to query for predictions.
        """
        self.host = host
        self.port = port
        self.api_key = api_key
        self.project_name = project_name
        self.model_name = model_name
        self.session: Session | None = None

    def connect(self) -> None:
        """Connect to the MindsDB server.

        Raises:
            RuntimeError: If the mindsdb_sdk is not installed.
        """
        if Session is None:
            raise RuntimeError(
                "mindsdb_sdk is not installed. Please add it to your environment and "
                "ensure network access to a MindsDB server."
            )
        # Build the connection URL; for local installations, no API key is needed.
        connection_url = f"{self.host}:{self.port}"
        if self.api_key:
            self.session = Session(api_key=self.api_key)
        else:
            self.session = Session(connection_url)

    def predict_similar(self, description: str, n: int = 5) -> List[Dict[str, Any]]:
        """Query the MindsDB model for similar movies based on a text description.

        Args:
            description (str): Free‑text description provided by the user.
            n (int): Number of results to return.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing recommended movies.
        """
        if self.session is None:
            self.connect()
        assert self.session is not None  # for mypy typing
        # Retrieve the project and model
        project = self.session.get_project(self.project_name)
        model = project.get_model(self.model_name)
        # The model is expected to expose a `predict` method. The input schema
        # depends on how the model was trained. Here we assume a single field
        # named "description". The output will be a DataFrame-like object.
        prediction = model.predict({"description": description})
        # Convert the result to a list of dictionaries and return the top n
        if hasattr(prediction, "to_dict"):
            results = prediction.to_dict(orient="records")
        else:
            results = [dict(row) for row in prediction]
        return results[:n]