# Pal Router

## Overview
Ternary LLM routing system that classifies queries into three lanes: **FAST** (weak model), **REASONING** (strong model), or **AGENTIC** (weak model + Python code execution). Inspired by PAL (Program-Aided Language Models) and RouteLLM research. Also includes an **OrchestratorRouter** that uses an LLM (Nemotron-Orchestrator-8B) to dynamically orchestrate tool calls across multiple rounds.

## Stack
- **Language**: Python 3.10+
- **Build**: hatchling (pyproject.toml)
- **Linting**: ruff (line-length=100, rules: E, F, I, UP)
- **ML/NLP**: sentence-transformers (all-MiniLM-L6-v2), scikit-learn (MLP classifier)
- **LLM Providers**: openai, anthropic, google-generativeai, Groq (via openai SDK), llama.cpp (via openai SDK)
- **Utilities**: pydantic, python-dotenv, rich, textstat
- **Optional**: pandas, matplotlib (eval extras), tavily (web search)

## Commands
```bash
# Install (editable)
pip install -e .

# Install with dev/eval extras
pip install -e ".[dev]"
pip install -e ".[eval]"

# Lint
ruff check src/ tests/

# Run tests (unit, no API keys needed)
pytest tests/

# Quick routing-only evaluation (no API calls)
python eval/run_eval.py --quick

# Full evaluation (requires API keys)
python eval/run_eval.py --routing-only

# Research validation
python eval/research_validation_suite.py --report

# Train the router classifier
python scripts/train_router_classifier.py

# Generate more training data
python scripts/generate_with_groq.py --batches 2

# Create/augment training data
python scripts/create_clean_split.py
python scripts/augment_training_data.py
```

## Architecture
```
src/pal_router/
  router.py          # TernaryRouter - main entry point, routes via trained classifier
  trained_router.py  # TrainedRouter - embedding + sklearn MLP classifier for lane prediction
  embeddings.py      # Sentence transformer wrapper (all-MiniLM-L6-v2)
  complexity.py      # Heuristic complexity estimation (syntactic, logic, numeric signals)
  orchestrator.py    # OrchestratorRouter - LLM-based multi-round tool orchestration
  agentic.py         # PAL workflow: code gen -> execute -> retry -> fallback
  tools.py           # ToolRegistry and tool definitions (fast_model, strong_model, code_executor, web_search, final_answer)
  models.py          # Model client abstractions (OpenAI, Anthropic, Groq, Gemini, LlamaCpp, FallbackClient)
  config.py          # Config, ModelConfig dataclasses and provider-specific model presets
  types.py           # Lane enum, RoutingDecision dataclass
  conversation.py    # OrchestratorConfig, ConversationContext, ToolCall, ToolResult, OrchestratorDecision
  presets.py         # Factory functions: create_fast_router, create_quality_router, create_groq_orchestrator_router, etc.
```

Key data flow:
- **TernaryRouter**: query -> TrainedRouter.route() (embed + classify) -> execute on appropriate lane
- **OrchestratorRouter**: query -> orchestrator LLM selects tools -> execute tools -> accumulate context -> repeat until final_answer or budget exceeded

The trained classifier (MLP, 91.7% accuracy) lives in `models/router_classifier/` as pickle files. Training data in `data/training_queries.json`, held-out eval set in `eval/test_suite.json`.

## Conventions
- All source code in `src/pal_router/` (hatchling src layout)
- Dataclasses over plain dicts for structured data
- `from __future__ import annotations` used throughout
- Google-style docstrings on all public methods
- Type hints with `|` union syntax (not `Optional` or `Union`)
- Tests in `tests/` use pytest with `unittest.mock.Mock`/`patch`; scripts in root are ad-hoc/manual tests
- Tests add `src/` to `sys.path` manually
- No `__init__.py` in `src/` - package is `src/pal_router/`
- `ruff` for linting, no formatter specified (use ruff format if needed)
- Complexity signals: syntactic_grade, logic_density, numeric_density, constraint_count, question_depth

## Environment
Copy `.env.example` to `.env`. Required keys depend on provider choice:
- `GROQ_API_KEY` - default provider, free tier
- `GOOGLE_API_KEY` - Gemini models, free tier
- `OPENAI_API_KEY` - paid
- `ANTHROPIC_API_KEY` - paid
- `LLAMACPP_URL` - defaults to `http://localhost:8080/v1` for local llama.cpp
- `TAVILY_API_KEY` - optional, for web_search tool

Default config uses Groq as provider with Llama 8B (weak) and Llama 70B (strong).

## Git Workflow
- Single `master` branch, linear commits
- Commit prefixes: `feat:`, `fix:`, `test:`, `chore:`
- Conventional commit style with scope
