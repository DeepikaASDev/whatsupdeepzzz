from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from deepzzzz.agent import Deepzzzz
from deepzzzz.store import Store

# Monday 16 Nov 2026, 6:00 PM
FIXED_NOW = datetime(2026, 11, 16, 18, 0)


class ScriptedChatModel(BaseChatModel):
    """Fake chat model that returns pre-written replies in order.

    Drives the real LangGraph loop, tools and storage without a paid API.
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
def store() -> Store:
    return Store(":memory:", now_fn=lambda: FIXED_NOW)


@pytest.fixture
def make_bot(store):
    def _make(responses, **kwargs):
        llm = ScriptedChatModel(responses=list(responses))
        bot = Deepzzzz(llm, store, user_name="Deepika", now_fn=lambda: FIXED_NOW, **kwargs)
        return bot, llm
    return _make
