# PAL-Router

A ternary LLM routing system inspired by [The Illusion of Thinking](https://arxiv.org/abs/2506.06941) (Apple, 2025) which showed that weak models fall off a "reasoning cliff" on complex tasks. Instead of binary weak/strong routing, PAL-Router adds a third lane: weak model + code execution via [PAL (Program-Aided Language Models)](https://arxiv.org/abs/2211.10435).

**Three lanes:**
- **FAST** → Weak model (simple queries)
- **REASONING** → Strong model (complex reasoning)
- **AGENTIC** → Weak model + Python execution (math/computation)

## Setup

```bash
pip install -e .
cp .env.example .env
# Edit .env with your API keys
```

## Run

```bash
# Test routing (no API needed)
python eval/run_eval.py --quick

# Test with APIs
python examples/test_providers.py

# Interactive demo
python examples/interactive_demo.py
```

## Usage

```python
from pal_router import TernaryRouter

router = TernaryRouter()
result = router.execute("Calculate 15% of $200")

print(result.decision.lane)  # AGENTIC
print(result.answer)         # $30.00
```

## Results

- Routing accuracy: **100%** (60/60)
- Unit tests: **55 passing**

## References

- [The Illusion of Thinking](https://arxiv.org/abs/2506.06941) - Reasoning cliff findings
- [PAL: Program-Aided Language Models](https://arxiv.org/abs/2211.10435) - Code execution for reasoning
- [RouteLLM](https://arxiv.org/abs/2406.18665) - Binary routing with preference learning
