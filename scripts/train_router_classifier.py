"""Train the routing classifier on labeled query data."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pal_router.embeddings import embed_queries


def load_training_data(path: Path) -> tuple[list[str], list[str]]:
    """Load training queries and labels."""
    with open(path) as f:
        data = json.load(f)

    texts = [q["text"] for q in data["queries"]]
    labels = [q["lane"] for q in data["queries"]]
    return texts, labels


def train_classifier(
    texts: list[str],
    labels: list[str],
    model_type: str = "mlp",  # "logistic" or "mlp"
    embedding_model: str = "fast",
) -> tuple[object, LabelEncoder, dict]:
    """Train and evaluate the routing classifier."""

    print(f"Embedding {len(texts)} queries with '{embedding_model}' model...")
    X = embed_queries(texts, model_name=embedding_model)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    print(f"Classes: {le.classes_}")
    print(f"Distribution: {np.bincount(y)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Choose classifier
    if model_type == "logistic":
        clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
    else:
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            early_stopping=True,
            random_state=42,
        )

    # Cross-validation on training set
    print(f"\nCross-validation (5-fold)...")
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # Train final model
    print(f"\nTraining final model...")
    clf.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = clf.predict(X_test)

    print(f"\nTest Set Results:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Compute metrics
    test_accuracy = (y_pred == y_test).mean()

    metrics = {
        "cv_accuracy": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "test_accuracy": float(test_accuracy),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "embedding_model": embedding_model,
        "classifier_type": model_type,
    }

    return clf, le, metrics


def save_model(
    clf: object,
    label_encoder: LabelEncoder,
    metrics: dict,
    output_dir: Path,
):
    """Save trained model and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save classifier
    with open(output_dir / "classifier.pkl", "wb") as f:
        pickle.dump(clf, f)

    # Save label encoder
    with open(output_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/training_queries.json"))
    parser.add_argument("--output", type=Path, default=Path("models/router_classifier"))
    parser.add_argument("--model-type", choices=["logistic", "mlp"], default="mlp")
    parser.add_argument("--embedding", choices=["fast", "balanced", "accurate"], default="fast")
    args = parser.parse_args()

    texts, labels = load_training_data(args.data)
    clf, le, metrics = train_classifier(
        texts, labels,
        model_type=args.model_type,
        embedding_model=args.embedding,
    )
    save_model(clf, le, metrics, args.output)


if __name__ == "__main__":
    main()
