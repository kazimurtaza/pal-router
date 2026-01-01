"""Multi-signal complexity estimation for query routing."""

from __future__ import annotations

import re
from dataclasses import dataclass

import textstat


@dataclass
class ComplexitySignals:
    """Detailed breakdown of complexity signals for debugging."""

    syntactic_grade: float  # Flesch-Kincaid normalized to 0-1
    logic_density: float  # Weighted logic term presence
    numeric_density: float  # Numbers, equations, units
    constraint_count: int  # Constraint indicators
    question_depth: int  # Nested sub-questions
    estimated_steps: int  # Heuristic step count


# Logic terms that indicate reasoning complexity
LOGIC_TERMS = [
    # Conditionals
    "if",
    "then",
    "else",
    "unless",
    "when",
    "whenever",
    # Constraints
    "given",
    "assuming",
    "suppose",
    "let",
    "such that",
    "must",
    "cannot",
    "at least",
    "at most",
    "exactly",
    # Logical operators
    "and",
    "or",
    "not",
    "implies",
    "therefore",
    "thus",
    "because",
    "since",
    "hence",
    # Multi-step indicators
    "first",
    "then",
    "next",
    "finally",
    "step",
    # Optimization
    "maximize",
    "minimize",
    "optimal",
    "best",
    "worst",
    # Comparison
    "compare",
    "difference",
    "between",
    "versus",
    "relative",
]

# Constraint-specific terms (subset of logic terms with higher weight)
CONSTRAINT_TERMS = [
    "must",
    "cannot",
    "at least",
    "at most",
    "exactly",
    "given",
    "such that",
    "constraint",
    "require",
    "condition",
]


def _get_first_sentence(text: str) -> str:
    """Extract the first sentence from text."""
    # Simple sentence boundary detection
    match = re.match(r"^[^.!?]*[.!?]", text)
    if match:
        return match.group(0)
    # If no sentence boundary found, take first 100 chars
    return text[:100]


def _compute_syntactic_grade(text: str) -> float:
    """Compute normalized syntactic complexity using Flesch-Kincaid.

    Returns a value between 0 and 1, where higher means more complex.
    """
    # Flesch-Kincaid grade level (typically 0-18+)
    grade = textstat.flesch_kincaid_grade(text)
    # Normalize to 0-1, capping at grade 18
    return min(max(grade, 0) / 18.0, 1.0)


def _compute_logic_density(text: str) -> float:
    """Compute logic term density with position weighting.

    Terms in the first sentence are weighted 2x to handle
    'If X, then Y' patterns vs buried conditionals.
    """
    text_lower = text.lower()
    first_sentence = _get_first_sentence(text_lower)
    words = text_lower.split()

    if not words:
        return 0.0

    # Count terms with position weighting
    weighted_count = 0
    for term in LOGIC_TERMS:
        # Check first sentence (2x weight)
        first_count = first_sentence.count(term)
        weighted_count += first_count * 2

        # Check rest of text (1x weight)
        rest_count = text_lower.count(term) - first_count
        weighted_count += rest_count

    # Normalize by word count
    density = weighted_count / len(words)
    # Cap at 1.0
    return min(density, 1.0)


def _compute_numeric_density(text: str) -> float:
    """Detect math-heavy queries that benefit from code execution.

    Returns ratio of numeric tokens to total tokens.
    """
    patterns = [
        r"\d+\.?\d*",  # Numbers (integers and decimals)
        r"[\+\-\*\/\^]",  # Arithmetic operators
        r"\$\d+",  # Currency
        r"\d+%",  # Percentages
        r"[=<>]",  # Comparisons
        r"\d+:\d+",  # Ratios or times
    ]

    # Word-form numbers (count as 0.5 each since less explicit)
    word_number_pattern = r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|half|quarter|third|fourth|fifth|double|triple|twice)\b"

    # Count matches
    total_matches = 0
    for pattern in patterns:
        total_matches += len(re.findall(pattern, text))

    # Add word-form numbers (weighted 0.5)
    word_numbers = len(re.findall(word_number_pattern, text.lower()))
    total_matches += word_numbers * 0.5

    # Normalize by word count
    words = text.split()
    if not words:
        return 0.0

    density = total_matches / len(words)
    return min(density, 1.0)


def _count_constraints(text: str) -> int:
    """Count constraint indicators in the text."""
    text_lower = text.lower()
    count = 0
    for term in CONSTRAINT_TERMS:
        count += text_lower.count(term)
    return count


def _compute_question_depth(text: str) -> int:
    """Estimate the depth of nested sub-questions.

    Counts question marks and multi-part question indicators.
    """
    # Count question marks
    question_count = text.count("?")

    # Check for multi-part indicators
    multi_part_patterns = [
        r"\b(?:first|1\.)\b.*\b(?:second|2\.)\b",
        r"\b(?:a\))\b.*\b(?:b\))\b",
        r"\band\s+(?:also|then)\b",
    ]

    for pattern in multi_part_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            question_count += 1

    return question_count


def _estimate_steps(signals: ComplexitySignals) -> int:
    """Heuristically estimate the number of reasoning steps needed."""
    base_steps = 1

    # Add steps based on signals
    if signals.logic_density > 0.1:
        base_steps += int(signals.logic_density * 5)
    if signals.numeric_density > 0.1:
        base_steps += int(signals.numeric_density * 3)
    if signals.constraint_count > 0:
        base_steps += signals.constraint_count
    if signals.question_depth > 1:
        base_steps += signals.question_depth - 1

    return base_steps


def estimate_complexity(
    prompt: str,
    weights: dict[str, float] | None = None,
) -> tuple[float, ComplexitySignals]:
    """Estimate the complexity of a prompt.

    Args:
        prompt: The user's query.
        weights: Optional custom weights for each signal.
            Keys: syntactic_grade, logic_density, numeric_density,
                  constraint_count, question_depth

    Returns:
        Tuple of (overall_score 0.0-1.0, detailed signals)
    """
    default_weights = {
        "syntactic_grade": 0.15,
        "logic_density": 0.30,
        "numeric_density": 0.25,
        "constraint_count": 0.15,
        "question_depth": 0.15,
    }
    weights = weights or default_weights

    # Compute individual signals
    syntactic_grade = _compute_syntactic_grade(prompt)
    logic_density = _compute_logic_density(prompt)
    numeric_density = _compute_numeric_density(prompt)
    constraint_count = _count_constraints(prompt)
    question_depth = _compute_question_depth(prompt)

    # Build signals object (before estimated_steps)
    signals = ComplexitySignals(
        syntactic_grade=syntactic_grade,
        logic_density=logic_density,
        numeric_density=numeric_density,
        constraint_count=constraint_count,
        question_depth=question_depth,
        estimated_steps=0,  # Will be updated below
    )

    # Estimate steps
    signals.estimated_steps = _estimate_steps(signals)

    # Compute weighted score
    # Normalize constraint_count and question_depth to 0-1 range
    normalized_constraints = min(constraint_count / 5.0, 1.0)
    normalized_depth = min(question_depth / 3.0, 1.0)

    score = (
        weights["syntactic_grade"] * syntactic_grade
        + weights["logic_density"] * logic_density
        + weights["numeric_density"] * numeric_density
        + weights["constraint_count"] * normalized_constraints
        + weights["question_depth"] * normalized_depth
    )

    # Ensure score is in [0, 1]
    score = min(max(score, 0.0), 1.0)

    return score, signals
