"""Shared Pydantic data models."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    role: Role
    content: str
    tool_call_id: str | None = None


class ToolResult(BaseModel):
    tool_call_id: str
    output: Any
    error: str | None = None


class AgentConfig(BaseModel):
    model: str = Field(default="gpt-4o-mini")
    max_iterations: int = Field(default=10, ge=1, le=50)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    system_prompt: str = "You are a helpful assistant."
