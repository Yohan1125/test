# PhnyX Lab — Backend Engineer Take-Home Assignments

This repository contains my solutions to the PhnyX Lab Backend Engineer
take-home assignments. PhnyX Lab builds agentic AI workflows for
pharmaceutical research and clinical intelligence (Cheiron platform).

## Repository Structure

| Directory | Scope | Time Budget |
|---|---|---|
| [`assignment_1/`](./assignment_1/README.md) | Focused task (TBD on receipt) | ~1.5 hours |
| [`assignment_2/`](./assignment_2/README.md) | Comprehensive system (TBD on receipt) | ~24 hours |

## Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or pip

Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Quick Start

Each assignment is a self-contained Python project.

```bash
# Assignment 1
cd assignment_1
uv sync
cp .env.example .env   # fill in API keys
uv run pytest

# Assignment 2
cd assignment_2
uv sync
cp .env.example .env   # fill in API keys
uv run pytest -m "not integration"
```

## Tooling Conventions

| Tool | Purpose |
|------|---------|
| `uv` | Dependency & venv management |
| `pytest` | Test runner with coverage |
| `ruff` | Linting and formatting |
| `mypy` | Static type checking (strict mode) |
| `hatchling` | Build backend |
