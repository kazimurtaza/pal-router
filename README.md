# PAL-Router

A ternary LLM routing system inspired by [The Illusion of Thinking](https://arxiv.org/abs/2506.06941) (Apple, 2025) which showed that weak models fall off a "reasoning cliff" on complex tasks. Instead of binary weak/strong routing, PAL-Router adds a third lane: weak model + code execution via [PAL (Program-Aided Language Models)](https://arxiv.org/abs/2211.10435).

**Three lanes:**
- **FAST** → Weak model (simple queries)
- **REASONING** → Strong model (complex reasoning)
- **AGENTIC** → Weak model + Python execution (math/computation)

## How It Works

PAL-Router uses a **trained embedding classifier** for routing decisions:

```
Query → Embedding (all-MiniLM-L6-v2) → MLP Classifier → Lane
```

Research shows trained classifiers significantly outperform heuristics:
> "Routers trained on augmented datasets outperform random baselines significantly. The BERT classifier achieved an APGR improvement of over 50%." — [RouteLLM](https://lmsys.org/blog/2024-07-01-routellm/)

## Setup

```bash
pip install -e .
cp .env.example .env
# Edit .env with your API keys
```

## Training the Router

The router requires a trained classifier:

```bash
# Create/augment training data (avoiding test contamination)
python scripts/create_clean_split.py
python scripts/augment_training_data.py

# Train the classifier
python scripts/train_router_classifier.py

# Evaluate on held-out test set
python scripts/test_routing_only.py
```

### Current Results

| Metric | Value |
|--------|-------|
| Training queries | 457 (no test overlap) |
| Held-out test queries | 60 |
| **Test accuracy** | **91.7%** (55/60) |
| Cross-validation | 97% |
| Latency | ~9ms/query |

## Research Validation

The implementation validates core claims from the cited research papers:

```bash
python eval/research_validation_suite.py --report
```

| Experiment | Paper | Result | Status |
|------------|-------|--------|--------|
| Classifier vs Heuristics | RouteLLM | 96% vs 64% (+50%) | ✓ PASS |
| Code Execution Accuracy | PAL | 85% (17/20) | ✓ PASS |
| Complexity Discrimination | Apple | AGENTIC=0.35, FAST=0.00 | ✓ PASS |

**All 3 research claims validated on held-out test data.**

### Generating More Training Data

```bash
# Generate with Groq LLM
python scripts/generate_with_groq.py --batches 2

# Add edge cases for specific patterns
python scripts/add_edge_cases.py

# Retrain
python scripts/train_router_classifier.py
```

## Usage

```python
from pal_router import TernaryRouter

router = TernaryRouter()
result = router.execute("Calculate 15% of $200")

print(result.decision.lane)        # AGENTIC
print(result.decision.confidence)  # 0.87
print(result.answer)               # $30.00
```

## Project Structure

```
pal-router/
├── src/pal_router/
│   ├── router.py          # TernaryRouter (uses classifier)
│   ├── trained_router.py  # Embedding classifier
│   ├── embeddings.py      # Sentence transformer wrapper
│   ├── agentic.py         # Code execution workflow
│   └── complexity.py      # Complexity signal extraction
├── scripts/
│   ├── train_router_classifier.py
│   ├── create_clean_split.py    # Prevents train/test contamination
│   ├── augment_training_data.py
│   └── test_routing_only.py
├── models/router_classifier/
│   ├── classifier.pkl
│   └── label_encoder.pkl
├── data/
│   └── training_queries.json    # Training data (no test overlap)
└── eval/
    ├── test_suite.json              # Held-out evaluation set
    └── research_validation_suite.py # Validates research claims
```

## References

- [The Illusion of Thinking](https://arxiv.org/abs/2506.06941) - Reasoning cliff findings
- [PAL: Program-Aided Language Models](https://arxiv.org/abs/2211.10435) - Code execution for reasoning
- [RouteLLM](https://lmsys.org/blog/2024-07-01-routellm/) - Trained routers outperform heuristics
