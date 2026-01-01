#!/usr/bin/env python3
"""Evaluation harness for the ternary LLM router."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pal_router import Config, TernaryRouter
from pal_router.complexity import estimate_complexity


@dataclass
class QueryResult:
    """Result for a single query evaluation."""

    query_id: str
    prompt: str
    expected_lane: str
    actual_lane: str
    routing_correct: bool
    complexity_score: float
    expected_answer: Any | None
    actual_answer: str
    answer_correct: bool | None  # None if not verifiable
    cost_usd: float
    latency_ms: float
    category: str
    difficulty: str


@dataclass
class EvaluationReport:
    """Complete evaluation report."""

    timestamp: str
    total_queries: int
    holdout_excluded: int
    routing_accuracy: float
    answer_accuracy: float | None
    total_cost_usd: float
    cost_per_correct: float | None
    avg_latency_ms: float
    results_by_lane: dict[str, dict[str, Any]] = field(default_factory=dict)
    results_by_category: dict[str, dict[str, Any]] = field(default_factory=dict)
    query_results: list[QueryResult] = field(default_factory=list)


def load_test_suite(path: Path, exclude_holdout: bool = True) -> list[dict]:
    """Load test suite from JSON file.

    Args:
        path: Path to test_suite.json.
        exclude_holdout: If True, exclude holdout queries.

    Returns:
        List of query dicts.
    """
    with open(path) as f:
        data = json.load(f)

    queries = data["queries"]
    holdout_ids = set(data.get("holdout_ids", []))

    if exclude_holdout:
        queries = [q for q in queries if q["id"] not in holdout_ids]

    return queries


def check_answer(actual: str, expected: Any) -> bool | None:
    """Check if the actual answer matches the expected answer.

    Args:
        actual: The model's answer.
        expected: Expected answer (string, number, or dict).

    Returns:
        True if match, False if mismatch, None if not verifiable.
    """
    if expected is None:
        return None

    actual_lower = actual.lower().strip()

    if isinstance(expected, bool):
        return ("yes" in actual_lower) == expected or ("true" in actual_lower) == expected

    if isinstance(expected, (int, float)):
        # Try to find the number in the response
        import re

        numbers = re.findall(r"[-+]?\d*\.?\d+", actual)
        for num_str in numbers:
            try:
                num = float(num_str)
                if abs(num - expected) < 0.1:  # Allow small tolerance
                    return True
            except ValueError:
                continue
        return False

    if isinstance(expected, str):
        return expected.lower() in actual_lower

    if isinstance(expected, dict):
        # Check if all key-value pairs are mentioned
        for key, value in expected.items():
            if isinstance(value, (int, float)):
                if str(int(value)) not in actual and str(value) not in actual:
                    return False
            elif str(value).lower() not in actual_lower:
                return False
        return True

    return None


def run_evaluation(
    test_suite_path: Path,
    output_dir: Path,
    routing_only: bool = False,
    exclude_holdout: bool = True,
) -> EvaluationReport:
    """Run the full evaluation.

    Args:
        test_suite_path: Path to test_suite.json.
        output_dir: Directory for output files.
        routing_only: If True, only test routing without execution.
        exclude_holdout: If True, exclude holdout queries.

    Returns:
        EvaluationReport with all results.
    """
    console = Console()
    load_dotenv()

    # Load test suite
    queries = load_test_suite(test_suite_path, exclude_holdout)
    total_queries = len(queries)

    console.print(f"\n[bold]Running evaluation on {total_queries} queries[/bold]\n")

    # Initialize router
    config = Config()
    router = TernaryRouter(config=config)

    results: list[QueryResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating...", total=total_queries)

        for query in queries:
            query_id = query["id"]
            prompt = query["prompt"]
            expected_lane = query["expected_lane"]
            expected_answer = query.get("expected_answer")
            category = query.get("category", "unknown")
            difficulty = query.get("difficulty", "unknown")

            progress.update(task, description=f"[cyan]{query_id}[/cyan]")

            if routing_only:
                # Just test routing decision
                decision = router.route(prompt)
                actual_lane = decision.lane.value
                routing_correct = actual_lane == expected_lane

                results.append(
                    QueryResult(
                        query_id=query_id,
                        prompt=prompt,
                        expected_lane=expected_lane,
                        actual_lane=actual_lane,
                        routing_correct=routing_correct,
                        complexity_score=decision.complexity_score,
                        expected_answer=expected_answer,
                        actual_answer="",
                        answer_correct=None,
                        cost_usd=0.0,
                        latency_ms=0.0,
                        category=category,
                        difficulty=difficulty,
                    )
                )
            else:
                # Full execution
                result = router.execute(prompt)
                actual_lane = result.decision.lane.value
                routing_correct = actual_lane == expected_lane
                answer_correct = check_answer(result.answer, expected_answer)

                results.append(
                    QueryResult(
                        query_id=query_id,
                        prompt=prompt,
                        expected_lane=expected_lane,
                        actual_lane=actual_lane,
                        routing_correct=routing_correct,
                        complexity_score=result.decision.complexity_score,
                        expected_answer=expected_answer,
                        actual_answer=result.answer[:500],  # Truncate long answers
                        answer_correct=answer_correct,
                        cost_usd=result.total_cost_usd,
                        latency_ms=result.total_latency_ms,
                        category=category,
                        difficulty=difficulty,
                    )
                )

            progress.advance(task)

    # Compute metrics
    routing_correct_count = sum(1 for r in results if r.routing_correct)
    routing_accuracy = routing_correct_count / total_queries

    verifiable = [r for r in results if r.answer_correct is not None]
    if verifiable:
        answer_correct_count = sum(1 for r in verifiable if r.answer_correct)
        answer_accuracy = answer_correct_count / len(verifiable)
    else:
        answer_accuracy = None

    total_cost = sum(r.cost_usd for r in results)
    avg_latency = sum(r.latency_ms for r in results) / total_queries if total_queries else 0

    if answer_accuracy is not None and answer_correct_count > 0:
        cost_per_correct = total_cost / answer_correct_count
    else:
        cost_per_correct = None

    # Results by lane
    results_by_lane: dict[str, dict[str, Any]] = {}
    for lane in ["FAST", "REASONING", "AGENTIC"]:
        lane_results = [r for r in results if r.actual_lane == lane]
        if lane_results:
            results_by_lane[lane] = {
                "count": len(lane_results),
                "routing_accuracy": sum(1 for r in lane_results if r.routing_correct)
                / len(lane_results),
                "total_cost": sum(r.cost_usd for r in lane_results),
                "avg_latency": sum(r.latency_ms for r in lane_results) / len(lane_results),
            }

    # Results by category
    results_by_category: dict[str, dict[str, Any]] = {}
    categories = set(r.category for r in results)
    for category in categories:
        cat_results = [r for r in results if r.category == category]
        if cat_results:
            results_by_category[category] = {
                "count": len(cat_results),
                "routing_accuracy": sum(1 for r in cat_results if r.routing_correct)
                / len(cat_results),
                "total_cost": sum(r.cost_usd for r in cat_results),
            }

    # Build report
    report = EvaluationReport(
        timestamp=datetime.now().isoformat(),
        total_queries=total_queries,
        holdout_excluded=50 - total_queries if exclude_holdout else 0,
        routing_accuracy=routing_accuracy,
        answer_accuracy=answer_accuracy,
        total_cost_usd=total_cost,
        cost_per_correct=cost_per_correct,
        avg_latency_ms=avg_latency,
        results_by_lane=results_by_lane,
        results_by_category=results_by_category,
        query_results=results,
    )

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON output
    json_path = output_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)

    # Print summary table
    print_summary(console, report)

    console.print(f"\n[green]Results saved to {json_path}[/green]\n")

    return report


def print_summary(console: Console, report: EvaluationReport) -> None:
    """Print a summary table of the evaluation results."""
    console.print("\n[bold]Evaluation Summary[/bold]\n")

    # Overall metrics
    overall = Table(title="Overall Metrics")
    overall.add_column("Metric", style="cyan")
    overall.add_column("Value", style="magenta")

    overall.add_row("Total Queries", str(report.total_queries))
    overall.add_row("Routing Accuracy", f"{report.routing_accuracy:.1%}")
    if report.answer_accuracy is not None:
        overall.add_row("Answer Accuracy", f"{report.answer_accuracy:.1%}")
    overall.add_row("Total Cost", f"${report.total_cost_usd:.4f}")
    if report.cost_per_correct is not None:
        overall.add_row("Cost per Correct", f"${report.cost_per_correct:.4f}")
    overall.add_row("Avg Latency", f"{report.avg_latency_ms:.0f}ms")

    console.print(overall)

    # By lane
    if report.results_by_lane:
        lane_table = Table(title="Results by Lane")
        lane_table.add_column("Lane", style="cyan")
        lane_table.add_column("Count", justify="right")
        lane_table.add_column("Routing Acc", justify="right")
        lane_table.add_column("Cost", justify="right")
        lane_table.add_column("Avg Latency", justify="right")

        for lane, data in report.results_by_lane.items():
            lane_table.add_row(
                lane,
                str(data["count"]),
                f"{data['routing_accuracy']:.1%}",
                f"${data['total_cost']:.4f}",
                f"{data['avg_latency']:.0f}ms",
            )

        console.print(lane_table)

    # By category
    if report.results_by_category:
        cat_table = Table(title="Results by Category")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Count", justify="right")
        cat_table.add_column("Routing Acc", justify="right")

        for category, data in report.results_by_category.items():
            cat_table.add_row(
                category,
                str(data["count"]),
                f"{data['routing_accuracy']:.1%}",
            )

        console.print(cat_table)


def run_routing_only(test_suite_path: Path) -> None:
    """Quick test of routing decisions without API calls.

    Args:
        test_suite_path: Path to test_suite.json.
    """
    console = Console()
    load_dotenv()

    queries = load_test_suite(test_suite_path, exclude_holdout=False)

    # Initialize router (uses actual routing logic, no API calls for route())
    config = Config()
    router = TernaryRouter(config=config, use_routellm=False)  # Disable RouteLLM for quick mode

    table = Table(title="Routing Decisions (No API Calls)")
    table.add_column("ID", style="cyan", max_width=15)
    table.add_column("Expected", style="green")
    table.add_column("Actual", style="yellow")
    table.add_column("Match", style="magenta")
    table.add_column("Score", justify="right")
    table.add_column("Num", justify="right")
    table.add_column("Logic", justify="right")

    correct = 0
    for query in queries:
        # Use actual router logic instead of reimplementing it
        decision = router.route(query["prompt"])
        actual = decision.lane.value
        signals = decision.signals

        expected = query["expected_lane"]
        match = "OK" if actual == expected else "MISS"
        if actual == expected:
            correct += 1

        table.add_row(
            query["id"],
            expected,
            actual,
            match,
            f"{decision.complexity_score:.2f}",
            f"{signals.numeric_density:.2f}",
            f"{signals.logic_density:.2f}",
        )

    console.print(table)
    console.print(f"\n[bold]Routing Accuracy: {correct}/{len(queries)} ({correct/len(queries):.1%})[/bold]\n")


def main():
    parser = argparse.ArgumentParser(description="Run evaluation for the ternary router")
    parser.add_argument(
        "--test-suite",
        type=Path,
        default=Path(__file__).parent / "test_suite.json",
        help="Path to test suite JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--routing-only",
        action="store_true",
        help="Only test routing decisions (no API calls)",
    )
    parser.add_argument(
        "--include-holdout",
        action="store_true",
        help="Include holdout queries in evaluation",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick routing check without API calls",
    )

    args = parser.parse_args()

    if args.quick:
        run_routing_only(args.test_suite)
    else:
        run_evaluation(
            test_suite_path=args.test_suite,
            output_dir=args.output_dir,
            routing_only=args.routing_only,
            exclude_holdout=not args.include_holdout,
        )


if __name__ == "__main__":
    main()
