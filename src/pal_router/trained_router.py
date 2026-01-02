"""Trained embedding-based router.

Uses sentence embeddings + sklearn classifier for routing decisions.
Research shows trained classifiers outperform heuristics significantly.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pal_router.complexity import estimate_complexity
from pal_router.embeddings import embed_query


@dataclass
class ClassifierPrediction:
    """Prediction from the trained classifier."""
    lane: str
    confidence: float
    probabilities: dict[str, float]


class TrainedRouter:
    """Routes queries using a trained embedding classifier.

    No heuristic fallback - classifier-only routing as per research:
    "Routers trained on augmented datasets outperform random baselines significantly.
    The BERT classifier achieved an APGR improvement of over 50%."
    """

    def __init__(
        self,
        model_dir: Path | str = "models/router_classifier",
        embedding_model: str = "fast",
    ):
        """Initialize the trained router.

        Args:
            model_dir: Directory containing classifier.pkl and label_encoder.pkl
            embedding_model: Which embedding model to use ("fast", "balanced", "accurate")

        Raises:
            FileNotFoundError: If model files are not found. Train the classifier first.
        """
        model_dir = Path(model_dir)

        if not (model_dir / "classifier.pkl").exists():
            raise FileNotFoundError(
                f"Classifier not found at {model_dir}. "
                "Train the classifier first:\n"
                "  python scripts/convert_test_suite.py\n"
                "  python scripts/train_router_classifier.py"
            )

        with open(model_dir / "classifier.pkl", "rb") as f:
            self.classifier = pickle.load(f)

        with open(model_dir / "label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)

        self.embedding_model = embedding_model

    def predict(self, query: str) -> ClassifierPrediction:
        """Predict routing lane from query.

        Args:
            query: The user's query text.

        Returns:
            ClassifierPrediction with lane, confidence, and probabilities.
        """
        # Embed query
        embedding = embed_query(query, model_name=self.embedding_model)

        # Get prediction and probabilities
        proba = self.classifier.predict_proba(embedding.reshape(1, -1))[0]
        pred_idx = np.argmax(proba)
        pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(proba[pred_idx])

        # Build probability dict
        probabilities = {
            label: float(p)
            for label, p in zip(self.label_encoder.classes_, proba)
        }

        return ClassifierPrediction(
            lane=pred_label,
            confidence=confidence,
            probabilities=probabilities,
        )

    def route(self, query: str):
        """Route query to appropriate lane.

        Args:
            query: The user's query text.

        Returns:
            RoutingDecision with lane, confidence, and metadata.
        """
        from pal_router.types import Lane, RoutingDecision

        # Get complexity signals for metadata
        score, signals = estimate_complexity(query)

        # Get classifier prediction
        prediction = self.predict(query)

        return RoutingDecision(
            lane=Lane(prediction.lane),
            complexity_score=score,
            signals=signals,
            reason=f"Classifier: {prediction.lane} ({prediction.confidence:.1%} confidence)",
            confidence=prediction.confidence,
            probabilities=prediction.probabilities,
        )
