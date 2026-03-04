# Assignment 2 — [Task Title Here]

> Estimated time: 24 hours
> Received: [date]
> Submitted: [date]

## Problem Statement

[Paste the exact problem statement here after receiving it.]

## Architecture

```
src/pharma_agent/
├── agent/          # Agentic workflow orchestration (tool-calling loop)
├── retrieval/      # RAG pipeline (embed → store → query)
└── evaluation/     # Metrics and automated evaluation harness
```

The agent treats retrieval as a pluggable tool. The evaluation module is
a black-box harness that runs any `AgentWorkflow` over a JSONL dataset.

```
User query
    │
    ▼
AgentWorkflow (agent/workflow.py)
    │
    ├─► retrieve_context() ──► RetrievalPipeline ──► VectorStore (ChromaDB)
    │                                             └──► EmbeddingModel
    │
    └─► LLM (OpenAI / Anthropic)
            │
            ▼
        Final answer
```

## Setup

```bash
# From the assignment_2/ directory

# Using uv (recommended)
uv sync
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY

# Using pip
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

## Running the Solution

```bash
# Run the agent workflow interactively
uv run python -m pharma_agent.agent.workflow

# Ingest documents into the vector store
uv run python -m pharma_agent.retrieval.pipeline --ingest ./data/

# Run the evaluation harness
uv run python -m pharma_agent.evaluation.evaluator \
    --dataset ./data/eval.jsonl \
    --output ./results/eval_results.json
```

## Running Tests

```bash
# Fast unit tests only (no API keys needed)
uv run pytest -m "not integration"

# Full suite including integration tests
uv run pytest

# With coverage report
uv run pytest --cov=src/pharma_agent --cov-report=html
```

## Linting & Type Checks

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy src/
```

## Design Decisions

- **Src layout**: Prevents accidental root imports during testing.
- **Embedding backend abstraction**: `sentence-transformers` (local, no API key) by default; swap to `OpenAIEmbeddings` via `EMBEDDING_BACKEND=openai`.
- **ChromaDB**: Python-native persistent store, no compilation step. FAISS available as `pip install -e ".[faiss]"`.
- **Abstract `BaseAgent`**: Demonstrates understanding of the tool-calling loop from first principles, not just LangChain glue.
- **Offline metrics first**: `token_f1` and `context_recall_lexical` work without API calls — CI-friendly, fast feedback loop.

## Assumptions

- [Assumption 1]
- [Assumption 2]

## Evaluation Results

| Metric | Score |
|---|---|
| Answer Correctness (token F1) | — |
| Context Recall (lexical) | — |
| Mean Latency (s) | — |

_Fill in after running the eval harness on the provided dataset._

## If I Had More Time

- Add LLM-graded faithfulness (RAGAS-style) for higher-fidelity metrics
- Implement Reciprocal Rank Fusion across multiple retrieval sources
- Add a FastAPI layer with `/query` and `/health` endpoints
- Set up async embedding and retrieval for lower latency
