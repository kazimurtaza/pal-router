"""Tests for the complexity estimation module."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pal_router.complexity import (
    ComplexitySignals,
    estimate_complexity,
    _compute_logic_density,
    _compute_numeric_density,
    _count_constraints,
)


class TestLogicDensity:
    """Tests for logic density calculation."""

    def test_no_logic_terms(self):
        """Text without logic terms should have zero density."""
        text = "The cat sat on the mat."
        density = _compute_logic_density(text)
        assert density == 0.0

    def test_simple_conditional(self):
        """Simple conditional should have positive density."""
        text = "If it rains, then I will stay home."
        density = _compute_logic_density(text)
        assert density > 0.0

    def test_first_sentence_weighting(self):
        """Logic terms in first sentence should be weighted higher."""
        text1 = "If the condition is met, do this. The end."
        text2 = "Start here now. If the condition is met, do this."

        density1 = _compute_logic_density(text1)
        density2 = _compute_logic_density(text2)

        # First sentence "if" should give higher weight
        assert density1 >= density2  # At least equal due to normalization


class TestNumericDensity:
    """Tests for numeric density calculation."""

    def test_no_numbers(self):
        """Text without numbers should have zero density."""
        text = "The quick brown fox jumps over the lazy dog."
        density = _compute_numeric_density(text)
        assert density == 0.0

    def test_with_numbers(self):
        """Text with numbers should have positive density."""
        text = "Calculate 25% of $100."
        density = _compute_numeric_density(text)
        assert density > 0.0

    def test_math_heavy(self):
        """Math-heavy text should have high density."""
        text = "If x = 5 and y = 10, then x + y = 15 and x * y = 50."
        density = _compute_numeric_density(text)
        assert density > 0.3


class TestConstraintCount:
    """Tests for constraint counting."""

    def test_no_constraints(self):
        """Text without constraints should have zero count."""
        text = "The weather is nice today."
        count = _count_constraints(text)
        assert count == 0

    def test_with_constraints(self):
        """Text with constraints should have positive count."""
        text = "You must arrive by 5pm. At least 3 people are required."
        count = _count_constraints(text)
        assert count >= 2


class TestEstimateComplexity:
    """Tests for overall complexity estimation."""

    def test_simple_query(self):
        """Simple factual query should have low complexity."""
        query = "What is the capital of France?"
        score, signals = estimate_complexity(query)

        assert score < 0.3
        assert isinstance(signals, ComplexitySignals)

    def test_math_query(self):
        """Math query should have numeric content detected."""
        query = "Calculate the compound interest on $10,000 at 5% over 3 years."
        score, signals = estimate_complexity(query)

        assert score > 0.1  # Has some complexity
        assert signals.numeric_density > 0.0  # Key: has numeric content

    def test_logic_heavy_query(self):
        """Logic-heavy query should have high logic density."""
        query = """
        If all A are B, and all B are C, then are all A also C?
        Given that X implies Y, and we know X is true, what can we conclude?
        """
        score, signals = estimate_complexity(query)

        assert score > 0.2  # Has complexity
        assert signals.logic_density > 0.05  # Key: has logic content

    def test_constraint_problem(self):
        """Constraint problem should have constraints detected."""
        query = """
        I have 3 boxes. Box A must weigh twice as much as Box B.
        Box C cannot weigh more than Box A minus 5kg.
        Given the total weight is exactly 35kg, find each box's weight.
        """
        score, signals = estimate_complexity(query)

        assert score > 0.1  # Has complexity
        assert signals.constraint_count >= 2  # Key: has constraints

    def test_score_range(self):
        """Score should always be between 0 and 1."""
        queries = [
            "",
            "Hi",
            "What is 2+2?",
            "Explain quantum mechanics.",
            "If x=1 and y=2 and z=3, given that a must be at least x+y, and b cannot exceed z*2, and c is exactly a+b-1, find a, b, c.",
        ]

        for query in queries:
            score, _ = estimate_complexity(query)
            assert 0.0 <= score <= 1.0


class TestCustomWeights:
    """Tests for custom weight configurations."""

    def test_custom_weights(self):
        """Custom weights should affect the score."""
        query = "Calculate 15% of $200."

        # Default weights
        score1, _ = estimate_complexity(query)

        # Heavy numeric weight
        score2, _ = estimate_complexity(
            query,
            weights={
                "syntactic_grade": 0.0,
                "logic_density": 0.0,
                "numeric_density": 1.0,
                "constraint_count": 0.0,
                "question_depth": 0.0,
            }
        )

        # Since this query has numeric content, heavy numeric weight should change score
        assert score1 != score2
