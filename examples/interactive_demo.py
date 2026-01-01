#!/usr/bin/env python3
"""Interactive CLI demo for the Ternary LLM Router."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pal_router import Config, TernaryRouter


def main():
    """Run the interactive demo."""
    load_dotenv()
    console = Console()

    console.print(
        Panel.fit(
            "[bold blue]Ternary LLM Router Demo[/bold blue]\n\n"
            "Routes queries to:\n"
            "  [green]FAST[/green] → Weak model for simple queries\n"
            "  [yellow]REASONING[/yellow] → Strong model for complex reasoning\n"
            "  [cyan]AGENTIC[/cyan] → Weak model + code execution\n\n"
            "Type [bold]quit[/bold] or [bold]exit[/bold] to stop.",
            title="Welcome",
        )
    )

    config = Config()
    router = TernaryRouter(config=config)

    while True:
        console.print()
        try:
            query = console.input("[bold green]Enter query:[/bold green] ")
        except (EOFError, KeyboardInterrupt):
            break

        if query.lower().strip() in ("quit", "exit", "q"):
            break

        if not query.strip():
            continue

        # Show routing decision
        console.print("\n[dim]Analyzing query...[/dim]")
        decision = router.route(query)

        decision_table = Table(title="Routing Decision", show_header=True)
        decision_table.add_column("Signal", style="cyan")
        decision_table.add_column("Value", style="magenta")

        lane_color = {
            "FAST": "green",
            "REASONING": "yellow",
            "AGENTIC": "cyan",
        }
        lane_style = lane_color.get(decision.lane.value, "white")

        decision_table.add_row("Lane", f"[bold {lane_style}]{decision.lane.value}[/bold {lane_style}]")
        decision_table.add_row("Complexity Score", f"{decision.complexity_score:.3f}")
        decision_table.add_row("Logic Density", f"{decision.signals.logic_density:.3f}")
        decision_table.add_row("Numeric Density", f"{decision.signals.numeric_density:.3f}")
        decision_table.add_row("Constraint Count", str(decision.signals.constraint_count))
        decision_table.add_row("Estimated Steps", str(decision.signals.estimated_steps))
        decision_table.add_row("Reason", decision.reason)

        console.print(decision_table)

        # Ask if user wants to execute
        console.print()
        try:
            execute = console.input("[bold]Execute query? (y/n):[/bold] ")
        except (EOFError, KeyboardInterrupt):
            break

        if execute.lower().strip() not in ("y", "yes"):
            continue

        # Execute
        console.print("\n[dim]Executing...[/dim]")
        result = router.execute(query)

        # Show result
        result_table = Table(title="Execution Result", show_header=True)
        result_table.add_column("Metric", style="cyan")
        result_table.add_column("Value", style="magenta")

        result_table.add_row("Cost", f"${result.total_cost_usd:.6f}")
        result_table.add_row("Latency", f"{result.total_latency_ms:.0f}ms")

        if result.agentic_result:
            result_table.add_row("Agentic Attempts", str(result.agentic_result.attempts))
            result_table.add_row("Fallback Used", str(result.agentic_result.fallback_used))
            if result.agentic_result.code:
                result_table.add_row("Code Generated", "Yes")

        console.print(result_table)

        # Show answer
        console.print(
            Panel(
                result.answer,
                title="Answer",
                border_style="green" if result.agentic_result is None or result.agentic_result.success else "red",
            )
        )

        # Show code if agentic
        if result.agentic_result and result.agentic_result.code:
            console.print(
                Panel(
                    result.agentic_result.code,
                    title="Generated Code",
                    border_style="cyan",
                )
            )

    console.print("\n[bold]Goodbye![/bold]\n")


if __name__ == "__main__":
    main()
