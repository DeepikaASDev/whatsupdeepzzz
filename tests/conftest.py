from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from booking_agent.store import AppointmentStore

# Monday 16 Nov 2026, 8:00 AM: the clinic opens at 9.
FIXED_NOW = datetime(2026, 11, 16, 8, 0)


class ScriptedChatModel(BaseChatModel):
    """A fake chat model that returns pre-written replies in order.

    Lets tests drive the real LangGraph loop (tool calls, tool results,
    memory) without calling a paid LLM API.
    """

    responses: list[AIMessage]
    received: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.received.append(list(messages))
        if not self.responses:
            raise RuntimeError("ScriptedChatModel ran out of responses")
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


def tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


@pytest.fixture
def store() -> AppointmentStore:
    return AppointmentStore(":memory:", now_fn=lambda: FIXED_NOW)
