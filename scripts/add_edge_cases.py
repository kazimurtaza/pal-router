"""Add edge cases for commonly misclassified queries."""

from __future__ import annotations

import json
from pathlib import Path

# Edge cases based on test errors
EDGE_CASES = {
    # Logic puzzles that SHOULD be AGENTIC (need systematic solving)
    "AGENTIC": [
        # Syllogisms
        "If all cats are mammals and all mammals breathe, do all cats breathe?",
        "All birds have feathers. Penguins are birds. Do penguins have feathers?",
        "If no fish can fly and all trout are fish, can trout fly?",
        "All squares are rectangles. All rectangles have four sides. How many sides does a square have?",
        "If some dogs bark and all barking animals make noise, what can we conclude?",

        # Ordering/ranking puzzles
        "Tom is older than Sue. Sue is older than Bob. Who is the youngest?",
        "In a queue, Mary is ahead of John but behind Lisa. What is the order?",
        "Red is darker than blue. Yellow is lighter than blue. Rank the colors by darkness.",
        "A is heavier than B. C is lighter than B. D is heavier than A. Rank by weight.",
        "Team X beat Team Y. Team Y beat Team Z. Team Z beat Team W. Who is best?",

        # Constraint puzzles
        "A code must have 6 digits, at least 2 odd numbers. What's max even digits?",
        "A team needs 5 members: at least 2 seniors, at least 1 junior. How many options?",
        "A meal must have protein, vegetable, grain. I have 3 proteins, 4 veggies, 2 grains. How many combos?",

        # Causal/logical chains
        "It rained yesterday. When it rains, the grass gets wet. Is the grass wet?",
        "If power is out, the fridge stops. The milk spoils if the fridge stops. Power went out. What happens to milk?",
        "Plants need sunlight to grow. It was cloudy all week. What happened to the plants?",

        # Math that looks simple but needs calculation
        "What is 15% of 67?",
        "Calculate 8.5% tax on $129.99",
        "What is the average of 23, 45, 67, and 89?",
        "How many seconds in 3 hours and 45 minutes?",
        "If a car gets 28 mpg and gas is $3.50/gallon, cost for 200 miles?",
    ],

    # Queries that look complex but are simple (FAST)
    "FAST": [
        # Simple substitution that doesn't need calculation
        "If x equals 5, what is x?",
        "If the answer is 42, what is the answer?",
        "Given y = 10, what is y?",

        # Simple sequences
        "What number comes after 1, 2, 3, 4?",
        "Complete the pattern: A, B, C, D, ?",
        "What is the next even number after 8?",

        # Basic math that LLMs know
        "What is 1 + 1?",
        "What is 10 minus 7?",
        "What is 2 times 3?",
        "What is 8 divided by 2?",
        "What is half of 100?",
        "What is double 25?",

        # Trivial word problems
        "I have 3 apples. I eat 1. How many left?",
        "There are 5 birds. 2 fly away. How many remain?",
    ],

    # Complex reasoning (not calculation)
    "REASONING": [
        # Correlation vs causation
        "Does correlation imply causation? Explain.",
        "Ice cream sales and drownings both rise in summer. Are they related?",
        "Countries with more internet have higher GDP. Does internet cause wealth?",

        # Nuanced explanations
        "Explain the difference between sympathy and empathy.",
        "What distinguishes knowledge from belief?",
        "Describe the relationship between rights and responsibilities.",

        # Strategic/judgment
        "Should a startup prioritize growth or profitability?",
        "What factors should influence hiring decisions?",
        "How should a company respond to negative reviews?",

        # Comparative analysis
        "Compare the benefits of remote vs in-office work.",
        "What are the trade-offs between speed and accuracy?",
        "Analyze the pros and cons of open source software.",
    ],
}


def main():
    # Load existing training data
    train_path = Path("data/training_queries.json")
    with open(train_path) as f:
        data = json.load(f)

    existing = data.get("queries", [])
    existing_texts = {q["text"].lower().strip() for q in existing}

    # Load test suite to avoid contamination
    test_path = Path("eval/test_suite.json")
    with open(test_path) as f:
        test = json.load(f)
    test_texts = {q["prompt"].lower().strip() for q in test["queries"]}

    # Add edge cases
    new_queries = []
    for lane, queries in EDGE_CASES.items():
        for q in queries:
            q_lower = q.lower().strip()
            if q_lower not in existing_texts and q_lower not in test_texts:
                new_queries.append({"text": q, "lane": lane})
                existing_texts.add(q_lower)

    print(f"Existing queries: {len(existing)}")
    print(f"New edge cases: {len(new_queries)}")

    # Merge and save
    all_queries = existing + new_queries

    output = {
        "version": "3.1.0",
        "note": "Added edge cases for misclassified queries",
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
