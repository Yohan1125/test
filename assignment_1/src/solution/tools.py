"""Decorator-based tool registry and executor."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object


# Registry: tool name -> (definition, callable)
_REGISTRY: dict[str, tuple[ToolDefinition, Callable[..., Any]]] = {}


def tool(
    definition: ToolDefinition,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a function as an agent tool."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        _REGISTRY[definition.name] = (definition, fn)
        return fn

    return decorator


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return OpenAI-compatible tool schema list for all registered tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": defn.name,
                "description": defn.description,
                "parameters": defn.parameters,
            },
        }
        for defn, _ in _REGISTRY.values()
    ]


def execute_tool(name: str, arguments_json: str) -> Any:
    """Look up and call a registered tool by name."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown tool: {name!r}")
    _, fn = _REGISTRY[name]
    args = json.loads(arguments_json)
    return fn(**args)


# ---------------------------------------------------------------------------
# Placeholder tools — replace/extend with task-specific implementations
# ---------------------------------------------------------------------------


@tool(
    ToolDefinition(
        name="search_pubmed",
        description="Search PubMed for biomedical literature abstracts.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    )
)
def search_pubmed(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Stub — replace with real PubMed E-utilities call."""
    raise NotImplementedError("Implement PubMed search here")
