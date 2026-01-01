"""Generate training data using Groq API."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Load .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value

from groq import Groq

GENERATION_PROMPTS = {
    "FAST": """Generate 50 diverse simple queries that a basic LLM can answer directly without deep reasoning.

Categories to include:
- Factual lookups (capitals, dates, famous people, definitions)
- Simple translations (short phrases)
- Basic yes/no questions
- Simple pattern completion (easy sequences)
- Trivial math (single digit, no calculation needed)
- Simple creative (haiku, short poems)

Requirements:
- Vary length from 3-15 words
- Cover different domains (science, history, geography, language)
- Must NOT require complex reasoning or calculations
- Output as JSON array of strings only

Examples:
["What is the capital of France?", "Who wrote Hamlet?", "Is the sun a star?", "What comes after A, B, C?"]

Generate 50 new examples (JSON array only):""",

    "REASONING": """Generate 50 diverse complex queries requiring deep reasoning, analysis, or nuanced thinking.

Categories to include:
- Ethical dilemmas and moral philosophy
- Comparative analysis (pros/cons, trade-offs)
- Multi-perspective discussions
- Strategic thinking and planning
- Historical/societal analysis
- Philosophical questions
- Complex explanations requiring judgment

Requirements:
- Vary length from 10-40 words
- Require nuanced thinking, not just facts
- Should NOT be solvable with simple calculation
- Need synthesis, judgment, or multiple viewpoints
- Output as JSON array of strings only

Examples:
["What are the ethical implications of AI in healthcare?", "Compare capitalism and socialism", "Discuss the trolley problem"]

Generate 50 new examples (JSON array only):""",

    "AGENTIC": """Generate 50 diverse queries that benefit from code execution or precise calculation.

Categories to include:
- Percentage calculations (tips, discounts, taxes)
- Compound interest and financial math
- Multi-step word problems
- Geometry (area, volume, perimeter)
- Algebra (solve for x)
- Logic puzzles requiring systematic solving
- Combinatorics and probability
- Unit conversions with math
- Time/distance/rate problems

Requirements:
- Must involve actual calculation or logical deduction
- Vary complexity from simple to multi-step
- Include specific numbers that need processing
- Output as JSON array of strings only

Examples:
["Calculate 18% tip on $75.50", "A train travels 60mph for 2 hours. How far?", "Solve 3x + 7 = 22"]

Generate 50 new examples (JSON array only):""",
}


def generate_queries(client: Groq, lane: str, count: int = 50) -> list[str]:
    """Generate queries for a specific lane."""
    prompt = GENERATION_PROMPTS[lane]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=4000,
    )

    text = response.choices[0].message.content.strip()

    # Parse JSON from response
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    # Find JSON array in text
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    try:
        queries = json.loads(text)
        if isinstance(queries, list):
            return [q for q in queries if isinstance(q, str) and len(q) > 5]
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON for {lane}: {e}")
        print(f"Text: {text[:500]}...")
        return []

    return []


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50, help="Queries per lane per batch")
    parser.add_argument("--batches", type=int, default=2, help="Number of batches per lane")
    args = parser.parse_args()

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    # Load existing training data
    train_path = Path("data/training_queries.json")
    if train_path.exists():
        with open(train_path) as f:
            existing = json.load(f)
        existing_queries = existing.get("queries", [])
    else:
        existing_queries = []

    existing_texts = {q["text"].lower().strip() for q in existing_queries}
    print(f"Existing queries: {len(existing_queries)}")

    # Load test suite to avoid contamination
    test_path = Path("eval/test_suite.json")
    with open(test_path) as f:
        test_suite = json.load(f)
    test_texts = {q["prompt"].lower().strip() for q in test_suite["queries"]}
    print(f"Test queries (to avoid): {len(test_texts)}")

    # Generate for each lane
    new_queries = []

    for lane in ["FAST", "REASONING", "AGENTIC"]:
        print(f"\nGenerating for {lane}...")

        for batch in range(args.batches):
            print(f"  Batch {batch + 1}/{args.batches}...")
            queries = generate_queries(client, lane, args.count)
            print(f"  Generated {len(queries)} queries")

            # Filter duplicates and test contamination
            for q in queries:
                q_lower = q.lower().strip()
                if q_lower not in existing_texts and q_lower not in test_texts:
                    new_queries.append({"text": q, "lane": lane})
                    existing_texts.add(q_lower)

    print(f"\nNew unique queries: {len(new_queries)}")

    # Merge and save
    all_queries = existing_queries + new_queries

    output = {
        "version": "3.0.0",
        "note": "Generated with Groq LLM, no test contamination",
        "queries": all_queries,
    }

    with open(train_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Total queries: {len(all_queries)}")

    # Show distribution
    from collections import Counter
    lane_counts = Counter(q["lane"] for q in all_queries)
    print(f"Distribution: {dict(lane_counts)}")


if __name__ == "__main__":
    main()
