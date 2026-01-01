#!/usr/bin/env python3
"""Cost comparison between routing strategies."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pal_router import Config, TernaryRouter


def main():
    """Compare costs across routing strategies."""
    load_dotenv()
    console = Console()

    # Sample queries representing different complexity levels
    queries = [
        ("What is the capital of France?", "factual"),
        ("Who wrote Romeo and Juliet?", "factual"),
        ("Calculate 15% of 80.", "math_simple"),
        ("If I invest $10,000 at 5% annual interest compounded yearly, how much after 3 years?", "math_complex"),
        ("A train travels at 60 mph for 2.5 hours, then 45 mph for 1.5 hours. Total distance?", "math_complex"),
        ("Explain the trolley problem and its ethical implications.", "reasoning"),
        ("Compare utilitarianism and deontological ethics.", "reasoning"),
        ("Three friends split a $105 bill. Alice paid $15 more than Bob. Carol paid twice what Bob paid. How much did each pay?", "constraint"),
    ]

    config = Config()
    router = TernaryRouter(config=config)

    # Track totals
    totals = {
        "FAST_ONLY": {"cost": 0.0, "correct": 0, "count": 0},
        "REASONING_ONLY": {"cost": 0.0, "correct": 0, "count": 0},
        "ROUTED": {"cost": 0.0, "correct": 0, "count": 0},
    }

    console.print("\n[bold]Running Cost Comparison[/bold]\n")
    console.print("Comparing: Always Weak | Always Strong | Ternary Router\n")

    results_table = Table(title="Per-Query Results")
    results_table.add_column("Query", style="cyan", max_width=40)
    results_table.add_column("Type", style="dim")
    results_table.add_column("Routed To", style="yellow")
    results_table.add_column("Weak Cost", justify="right")
    results_table.add_column("Strong Cost", justify="right")
    results_table.add_column("Router Cost", justify="right", style="green")

    for query, category in queries:
        console.print(f"[dim]Processing: {query[:40]}...[/dim]")

        # Run through all lanes for comparison
        try:
            comparison = router.compare_baselines(query)

            fast_cost = comparison["FAST"].total_cost_usd
            reasoning_cost = comparison["REASONING"].total_cost_usd
            routed_cost = comparison["ROUTED"].total_cost_usd
            routed_lane = comparison["ROUTED"].decision.lane.value

            totals["FAST_ONLY"]["cost"] += fast_cost
            totals["FAST_ONLY"]["count"] += 1

            totals["REASONING_ONLY"]["cost"] += reasoning_cost
            totals["REASONING_ONLY"]["count"] += 1

            totals["ROUTED"]["cost"] += routed_cost
            totals["ROUTED"]["count"] += 1

            results_table.add_row(
                query[:40] + "..." if len(query) > 40 else query,
                category,
                routed_lane,
                f"${fast_cost:.5f}",
                f"${reasoning_cost:.5f}",
                f"${routed_cost:.5f}",
            )
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            continue

    console.print(results_table)

    # Summary table
    summary = Table(title="Cost Summary")
    summary.add_column("Strategy", style="cyan")
    summary.add_column("Total Cost", justify="right", style="magenta")
    summary.add_column("Queries", justify="right")
    summary.add_column("Avg Cost", justify="right")

    for strategy, data in totals.items():
        avg = data["cost"] / data["count"] if data["count"] > 0 else 0
        summary.add_row(
            strategy.replace("_", " "),
            f"${data['cost']:.5f}",
            str(data["count"]),
            f"${avg:.5f}",
        )

    console.print(summary)

    # Savings calculation
    if totals["REASONING_ONLY"]["cost"] > 0:
        savings = (
            (totals["REASONING_ONLY"]["cost"] - totals["ROUTED"]["cost"])
            / totals["REASONING_ONLY"]["cost"]
            * 100
        )
        console.print(
            f"\n[bold green]Router saves {savings:.1f}% compared to always using the strong model[/bold green]\n"
        )


if __name__ == "__main__":
    main()
