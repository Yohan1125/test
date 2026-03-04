# Assignment 1 — [Task Title Here]

> Estimated time: 1.5 hours
> Received: [date]
> Submitted: [date]

## Problem Statement

[Paste the exact problem statement here after receiving it.]

## Approach

[Brief 2–4 sentence description of your approach after reading the problem.]

## Setup

```bash
# From the assignment_1/ directory

# Using uv (recommended)
uv sync
cp .env.example .env
# Edit .env with your API keys

# Using pip
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

## Running the Solution

```bash
uv run python -m solution.main
```

## Running Tests

```bash
uv run pytest
uv run pytest -v --tb=short          # verbose
uv run pytest --cov=src/solution     # with coverage
```

## Linting & Type Checks

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy src/
```

## Design Decisions

- [Decision 1: e.g., "Chose X over Y because..."]
- [Decision 2]
- [Decision 3]

## Assumptions

- [Assumption 1]
- [Assumption 2]

## If I Had More Time

- [What you'd improve or extend]
