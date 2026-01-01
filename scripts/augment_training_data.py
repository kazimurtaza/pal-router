"""Augment training data with variations and additional examples."""

from __future__ import annotations

import json
import random
from pathlib import Path

# Edge cases - queries that commonly get misclassified
EDGE_CASES_FAST = [
    # "How many" questions that are factual, not math
    "How many days are in a week?",
    "How many months are in a year?",
    "How many hours are in a day?",
    "How many states are in the USA?",
    "How many letters are in the alphabet?",
    "How many players are on a basketball team?",
    "How many sides does a hexagon have?",
    "How many strings does a guitar have?",

    # Simple math that doesn't need code
    "What is 2 + 2?",
    "What is 10 - 3?",
    "What is 5 times 6?",
    "What is 100 divided by 10?",
    "What is half of 20?",

    # Simple patterns that don't need code
    "What comes after 1, 2, 3?",
    "What is the next letter after A, B, C?",
    "Given the sequence 2, 4, 6, 8, what comes next?",
    "Given the sequence 1, 2, 4, 8, what comes next?",

    # Simple algebra that's trivial
    "If x = 5, what is x + 1?",
    "If y = 10, what is y - 5?",
]

# Additional curated training examples for each lane
ADDITIONAL_FAST = [
    # Factual lookups
    "What is the capital of Germany?",
    "What is the capital of Japan?",
    "What is the capital of Australia?",
    "What is the capital of Brazil?",
    "What is the capital of Canada?",
    "Who invented the telephone?",
    "Who discovered penicillin?",
    "Who wrote Pride and Prejudice?",
    "Who painted Starry Night?",
    "Who composed Symphony No. 5?",
    "What year did World War II end?",
    "What year was the moon landing?",
    "What is the chemical symbol for gold?",
    "What is the chemical symbol for oxygen?",
    "What is the boiling point of water?",
    "How many planets are in our solar system?",
    "What is the largest mammal?",
    "What is the smallest country in the world?",
    "What language is spoken in Brazil?",
    "What is the currency of Japan?",

    # Simple translations
    "Translate 'goodbye' to French",
    "Translate 'thank you' to German",
    "How do you say 'yes' in Italian?",
    "What is 'water' in Spanish?",

    # Simple definitions
    "What is photosynthesis?",
    "Define democracy",
    "What does 'ubiquitous' mean?",
    "Explain what a noun is",

    # Simple creative
    "Write a haiku about nature",
    "Give me a short poem about the moon",
    "What rhymes with 'cat'?",

    # Yes/no factual
    "Is the Earth round?",
    "Is water wet?",
    "Can birds fly?",
    "Do fish have lungs?",
    "Is the sun a star?",

    # Simple lookups
    "What color is grass?",
    "How many legs does a spider have?",
    "What sound does a cow make?",
    "Name three fruits",
    "List five colors",
]

ADDITIONAL_REASONING = [
    # Ethical discussions
    "Should AI systems be given rights?",
    "Is it ethical to use animals in medical research?",
    "Discuss the ethics of genetic engineering in humans",
    "What are the moral implications of self-driving cars?",
    "Is capital punishment ever justified?",

    # Comparative analysis
    "Compare socialism and capitalism",
    "What are the differences between Stoicism and Epicureanism?",
    "Contrast Eastern and Western philosophy",
    "Compare the French and American revolutions",
    "What are the pros and cons of nuclear energy?",

    # Complex explanations
    "Explain quantum entanglement and its implications",
    "How does climate change affect biodiversity?",
    "Discuss the butterfly effect in chaos theory",
    "Explain the relationship between language and thought",
    "How does social media influence democracy?",

    # Strategic thinking
    "What strategies can reduce income inequality?",
    "How should governments approach AI regulation?",
    "What are effective approaches to conflict resolution?",
    "Discuss strategies for sustainable urban development",
    "How can organizations foster innovation?",

    # Multi-perspective analysis
    "Analyze the impact of globalization from different viewpoints",
    "Discuss vaccination from medical, social, and political perspectives",
    "Examine the gig economy from worker and business perspectives",
    "Analyze space exploration priorities: science vs colonization",
    "Discuss free speech limits from various ethical frameworks",

    # Historical/societal analysis
    "What factors led to the fall of the Roman Empire?",
    "How did the printing press change society?",
    "Analyze the causes and effects of the Great Depression",
    "How has technology changed warfare throughout history?",
    "Discuss the evolution of human rights concepts",

    # Philosophical discussions
    "What is the meaning of consciousness?",
    "Is free will an illusion?",
    "Discuss the ship of Theseus paradox",
    "What makes something art?",
    "Can machines ever truly understand meaning?",
]

ADDITIONAL_AGENTIC = [
    # Math calculations
    "Calculate 23% of 847",
    "What is 15% tip on a $127.50 bill?",
    "Calculate 8.5% sales tax on $249.99",
    "What is 12.5% of 3200?",
    "Calculate the compound interest on $5000 at 3.5% for 7 years",
    "What is 18% annual interest on a $15000 loan?",

    # Multi-step word problems
    "A store has a 30% off sale, then takes an additional 15% off. What is the final price of a $200 item?",
    "If a car gets 32 miles per gallon and gas costs $3.89 per gallon, how much does it cost to drive 450 miles?",
    "A rectangle has a perimeter of 56 cm. If the length is 3 times the width, what are the dimensions?",
    "John earns $4200 per month. He spends 28% on rent, 15% on food, and 8% on utilities. How much is left?",
    "A pool fills at 15 gallons per minute but leaks 3 gallons per minute. How long to fill a 720 gallon pool?",

    # Constraint satisfaction
    "Schedule 4 meetings of 30 minutes each between 9am and 12pm with 15 minute breaks between them",
    "Distribute $1000 between savings (at least 40%), investments (at least 20%), and spending",
    "Allocate 8 hours between work, exercise (min 1 hour), meals (1.5 hours total), and leisure",
    "Divide 24 students into groups of 4, 5, or 6 with at least 4 groups",

    # Logic puzzles
    "If A is taller than B, B is taller than C, and D is shorter than C, rank them by height",
    "Alice, Bob, and Carol finished a race. Alice wasn't first. Bob beat Carol. Who won?",
    "In a family of 5, there are 2 parents and 3 children. At least one child is a boy. The oldest is a girl. What are possible gender combinations?",
    "If all roses are flowers and some flowers fade quickly, what can we conclude about roses?",

    # Algebra
    "Solve for x: 2x + 7 = 3x - 5",
    "If 3y - 12 = 2y + 8, what is y?",
    "Find x if 4(x - 3) = 2(x + 5)",
    "Solve the system: x + y = 10, x - y = 4",

    # Geometry
    "Calculate the area of a triangle with base 12 cm and height 8 cm",
    "What is the volume of a cylinder with radius 5 cm and height 10 cm?",
    "Find the hypotenuse of a right triangle with legs 9 and 12",
    "Calculate the circumference of a circle with diameter 14 inches",

    # Statistics/probability
    "What is the average of 23, 45, 67, 89, and 101?",
    "Calculate the median of: 12, 45, 23, 67, 34, 56, 78",
    "If you flip a coin 3 times, what's the probability of getting exactly 2 heads?",
    "A bag has 5 red and 3 blue marbles. What's the probability of picking 2 red in a row?",

    # Time calculations
    "If I leave at 9:45 AM and arrive at 2:15 PM, how long was the journey?",
    "A flight takes 7 hours 40 minutes. If it departs at 11:30 PM, what time does it arrive?",
    "How many minutes are there between 8:17 AM and 3:52 PM?",

    # Unit conversions with math
    "Convert 72 km/h to m/s",
    "How many seconds in 2.5 hours?",
    "If a recipe needs 2.5 cups and I'm making 1.5x the recipe, how many cups do I need?",

    # Financial calculations
    "Calculate monthly payments for a $250,000 mortgage at 4.5% over 30 years",
    "If I save $400/month with 5% annual return, how much after 10 years?",
    "A stock goes from $45 to $52. What is the percentage gain?",
    "Calculate the present value of $10,000 in 5 years at 6% discount rate",

    # Combinatorics
    "How many ways can you arrange 5 books on a shelf?",
    "In how many ways can you choose 3 people from a group of 8?",
    "How many 4-digit PINs are possible with digits 0-9?",
    "How many different 3-letter combinations can be made from A, B, C, D, E?",
]


def main():
    # Load existing data
    data_path = Path("data/training_queries.json")
    if data_path.exists():
        with open(data_path) as f:
            existing = json.load(f)
        existing_queries = existing.get("queries", [])
    else:
        existing_queries = []

    print(f"Existing queries: {len(existing_queries)}")

    # Get existing texts to avoid duplicates
    existing_texts = {q["text"].lower().strip() for q in existing_queries}

    # Add new examples
    new_queries = []

    # Add edge cases first (higher priority)
    for text in EDGE_CASES_FAST:
        if text.lower().strip() not in existing_texts:
            new_queries.append({"text": text, "lane": "FAST"})
            existing_texts.add(text.lower().strip())

    for text in ADDITIONAL_FAST:
        if text.lower().strip() not in existing_texts:
            new_queries.append({"text": text, "lane": "FAST"})
            existing_texts.add(text.lower().strip())

    for text in ADDITIONAL_REASONING:
        if text.lower().strip() not in existing_texts:
            new_queries.append({"text": text, "lane": "REASONING"})
            existing_texts.add(text.lower().strip())

    for text in ADDITIONAL_AGENTIC:
        if text.lower().strip() not in existing_texts:
            new_queries.append({"text": text, "lane": "AGENTIC"})
            existing_texts.add(text.lower().strip())

    print(f"New queries to add: {len(new_queries)}")

    # Merge
    all_queries = existing_queries + new_queries

    # Save
    output = {
        "version": "1.1.0",
        "queries": all_queries,
    }

    with open(data_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Total queries: {len(all_queries)}")

    # Show distribution
    from collections import Counter
    lane_counts = Counter(q["lane"] for q in all_queries)
    print(f"Distribution: {dict(lane_counts)}")


if __name__ == "__main__":
    main()
