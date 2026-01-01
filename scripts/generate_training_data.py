"""Generate synthetic training queries for each lane using an LLM."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

GENERATION_PROMPT = """Generate {count} diverse user queries that should be routed to {lane}.

{lane_description}

Requirements:
- Vary length (5-50 words)
- Vary domains (finance, science, everyday, academic)
- Vary complexity within the lane
- No overlap with other lanes
- Output as a JSON array of strings only, no explanation

Examples for reference:
{examples}

Generate {count} new diverse examples (JSON array only):"""

LANE_DESCRIPTIONS = {
    "FAST": """FAST lane is for simple queries that a weak LLM can handle directly:
- Factual lookups (capitals, dates, definitions)
- Simple translations
- Basic explanations that don't require deep reasoning
- Greetings and chitchat
- Simple creative tasks (haiku, short poems)
- Pattern recognition (simple sequences)
- Yes/no factual questions""",

    "REASONING": """REASONING lane is for complex queries needing a strong LLM:
- Nuanced analysis (pros/cons, comparisons)
- Ethical discussions and moral dilemmas
- Multi-perspective explanations
- Strategic advice and planning
- Complex creative writing with constraints
- Anything requiring nuance, judgment, or deep knowledge
- Historical/societal analysis
- Philosophical discussions""",

    "AGENTIC": """AGENTIC lane is for queries that benefit from code execution:
- Math calculations (percentages, compound interest, geometry)
- Multi-step word problems
- Constraint satisfaction (scheduling, allocation)
- Logic puzzles (syllogisms, ordering, deduction)
- Data processing (sorting, filtering, statistics)
- Algebraic equations
- Combinatorics and probability calculations
- Anything where precise computation beats LLM estimation""",
}

SEED_EXAMPLES = {
    "FAST": [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "Translate 'hello' to Spanish",
        "What color is the sky?",
        "Is 17 a prime number?",
        "Name three primary colors.",
    ],
    "REASONING": [
        "Explain the trolley problem and its ethical implications",
        "What are the pros and cons of remote work?",
        "Compare democracy and authoritarianism",
        "Discuss cognitive biases in decision-making",
        "Analyze the long-term impacts of AI in healthcare",
    ],
    "AGENTIC": [
        "Calculate 15% tip on a $84.50 bill",
        "If I invest $10,000 at 5% for 3 years, how much will I have?",
        "A farmer has chickens and cows. 20 heads, 56 legs. How many of each?",
        "Find the sum of all prime numbers between 1 and 100",
        "A train travels 60 mph for 2.5 hours. What is the distance?",
    ],
}


def generate_with_gemini(prompt: str) -> list[str]:
    """Generate queries using Gemini."""
    import google.generativeai as genai

    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")

    response = model.generate_content(prompt)
    text = response.text.strip()

    # Parse JSON array from response
    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    try:
        queries = json.loads(text)
        if isinstance(queries, list):
            return [q for q in queries if isinstance(q, str)]
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {text[:200]}...")
        return []

    return []


def generate_with_groq(prompt: str) -> list[str]:
    """Generate queries using Groq."""
    from groq import Groq

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )

    text = response.choices[0].message.content.strip()

    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    try:
        queries = json.loads(text)
        if isinstance(queries, list):
            return [q for q in queries if isinstance(q, str)]
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {text[:200]}...")
        return []

    return []


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50, help="Queries per lane")
    parser.add_argument("--provider", choices=["gemini", "groq"], default="gemini")
    parser.add_argument("--output", type=Path, default=Path("data/generated_queries.json"))
    args = parser.parse_args()

    generate_fn = generate_with_gemini if args.provider == "gemini" else generate_with_groq

    all_queries = []

    for lane, description in LANE_DESCRIPTIONS.items():
        print(f"\nGenerating {args.count} queries for {lane}...")

        prompt = GENERATION_PROMPT.format(
            count=args.count,
            lane=lane,
            lane_description=description,
            examples=json.dumps(SEED_EXAMPLES[lane], indent=2),
        )

        queries = generate_fn(prompt)
        print(f"  Generated {len(queries)} queries")

        for q in queries:
            all_queries.append({"text": q, "lane": lane})

    # Load existing training data
    existing_path = Path("data/training_queries.json")
    if existing_path.exists():
        with open(existing_path) as f:
            existing = json.load(f)
        existing_queries = existing.get("queries", [])
        print(f"\nLoaded {len(existing_queries)} existing queries")
    else:
        existing_queries = []

    # Merge and deduplicate
    existing_texts = {q["text"].lower() for q in existing_queries}
    new_queries = [q for q in all_queries if q["text"].lower() not in existing_texts]
    print(f"Adding {len(new_queries)} new unique queries")

    merged = existing_queries + new_queries

    # Save
    output = {
        "version": "1.0.0",
        "queries": merged,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(merged)} total queries to {args.output}")

    # Show distribution
    from collections import Counter
    lane_counts = Counter(q["lane"] for q in merged)
    print(f"Distribution: {dict(lane_counts)}")


if __name__ == "__main__":
    main()
