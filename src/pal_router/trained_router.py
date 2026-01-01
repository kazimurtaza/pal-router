"""Trained embedding-based router."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pal_router.complexity import ComplexitySignals, estimate_complexity
from pal_router.embeddings import embed_query
from pal_router.router import Lane, RoutingDecision


@dataclass
class ClassifierPrediction:
    """Prediction from the trained classifier."""
    lane: Lane
    confidence: float
    probabilities: dict[str, float]


class TrainedRouter:
    """Routes queries using a trained embedding classifier."""

    def __init__(
        self,
        model_dir: Path | str = "models/router_classifier",
        embedding_model: str = "fast",
        confidence_threshold: float = 0.6,
    ):
        """Initialize the trained router.

        Args:
            model_dir: Directory containing classifier.pkl and label_encoder.pkl
            embedding_model: Which embedding model to use ("fast", "balanced", "accurate")
            confidence_threshold: Below this, fall back to heuristics
        """
        model_dir = Path(model_dir)

        with open(model_dir / "classifier.pkl", "rb") as f:
            self.classifier = pickle.load(f)

        with open(model_dir / "label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)

        self.embedding_model = embedding_model
        self.confidence_threshold = confidence_threshold

    def predict(self, query: str) -> ClassifierPrediction:
        """Predict routing lane from query."""
        # Embed query
        embedding = embed_query(query, model_name=self.embedding_model)

        # Get prediction and probabilities
        proba = self.classifier.predict_proba(embedding.reshape(1, -1))[0]
        pred_idx = np.argmax(proba)
        pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
        confidence = proba[pred_idx]

        # Build probability dict
        probabilities = {
            label: float(p)
            for label, p in zip(self.label_encoder.classes_, proba)
        }

        return ClassifierPrediction(
            lane=Lane(pred_label),
            confidence=confidence,
            probabilities=probabilities,
        )

    def route(self, query: str) -> RoutingDecision:
        """Route query to appropriate lane."""
        # Get complexity signals for metadata
        score, signals = estimate_complexity(query)

        # Get classifier prediction
        prediction = self.predict(query)

        # Use prediction if confident enough
        if prediction.confidence >= self.confidence_threshold:
            return RoutingDecision(
                lane=prediction.lane,
                complexity_score=score,
                signals=signals,
                reason=f"Classifier: {prediction.lane.value} ({prediction.confidence:.1%} confidence)",
            )

        # Fall back to heuristics if low confidence
        self._log_for_review(query, prediction)

        # Import here to avoid circular dependency
        from pal_router.router import TernaryRouter
        fallback_router = TernaryRouter(use_routellm=False, use_trained_classifier=False)
        fallback_decision = fallback_router.route(query)
        fallback_decision.reason = f"Fallback (classifier confidence {prediction.confidence:.1%}): " + fallback_decision.reason

        return fallback_decision

    def _log_for_review(self, query: str, prediction: ClassifierPrediction):
        """Log queries that need manual labeling."""
        review_path = Path("data/needs_review.jsonl")
        review_path.parent.mkdir(parents=True, exist_ok=True)

        with open(review_path, "a") as f:
            f.write(json.dumps({
                "query": query,
                "predicted": prediction.lane.value,
                "confidence": prediction.confidence,
                "probabilities": prediction.probabilities,
            }) + "\n")
