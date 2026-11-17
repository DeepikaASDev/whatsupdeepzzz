"""The LangGraph agent: input guardrails -> model <-> tools loop -> output guardrails.

Graph:
    START -> agent --(tool calls?)--> tools -> agent ... --(no tool calls)--> END

Conversation memory is kept per session by a LangGraph checkpointer, keyed by
`thread_id = session_id`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .clinic import CLINIC_PHONE
from .guardrails import check_action_claims, check_input, check_output
from .prompts import build_system_prompt
from .store import AppointmentStore
from .tools import make_tools

log = logging.getLogger(__name__)

# Each model call or tool step counts toward this; stops runaway tool loops.
RECURSION_LIMIT = 16

FALLBACK_REPLY = (
    "Sorry, I'm having trouble right now. Please try again in a moment, "
    f"or call us at {CLINIC_PHONE}."
)


@dataclass
class ChatResult:
    reply: str
    tool_calls: list[str] = field(default_factory=list)
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    guardrail: str | None = None  # why a guardrail changed the reply, if one did
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    # Some providers return a list of content blocks.
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


class BookingAgent:
    def __init__(
        self,
        llm: BaseChatModel,
        store: AppointmentStore,
        checkpointer: BaseCheckpointSaver | None = None,
        max_message_chars: int = 1000,
        now_fn: Callable[[], datetime] | None = None,
    ):
        self.store = store
        self.max_message_chars = max_message_chars
        self.now = now_fn or store.now
        self.tools = make_tools(store)
        self.llm = llm.bind_tools(self.tools)
        self.graph = self._build_graph(checkpointer or InMemorySaver())

    def _build_graph(self, checkpointer: BaseCheckpointSaver):
        graph = StateGraph(MessagesState)
        graph.add_node("agent", self._call_model)
        graph.add_node("tools", ToolNode(self.tools))
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition)  # -> "tools" or END
        graph.add_edge("tools", "agent")
        return graph.compile(checkpointer=checkpointer)

    def _call_model(self, state: MessagesState) -> dict:
        system = SystemMessage(build_system_prompt(self.now()))
        response = self.llm.invoke([system, *state["messages"]])
        return {"messages": [response]}

    def history(self, session_id: str) -> list[BaseMessage]:
        state = self.graph.get_state({"configurable": {"thread_id": session_id}})
        return list(state.values.get("messages", [])) if state and state.values else []

    def chat(self, session_id: str, message: str) -> ChatResult:
        guard = check_input(message, self.max_message_chars)
        if not guard.allowed:
            return ChatResult(reply=guard.reply or "", guardrail=guard.reason)

        config = {"configurable": {"thread_id": session_id}, "recursion_limit": RECURSION_LIMIT}
        seen = len(self.history(session_id))
        started = time.perf_counter()
        try:
            state = self.graph.invoke({"messages": [HumanMessage(message)]}, config=config)
        except GraphRecursionError:
            log.warning("Recursion limit hit for session %s", session_id)
            return ChatResult(reply=FALLBACK_REPLY, error="recursion_limit",
                              latency_ms=int((time.perf_counter() - started) * 1000))
        except Exception as exc:  # model/provider errors should not crash the chat
            log.exception("Agent failed for session %s", session_id)
            return ChatResult(reply=FALLBACK_REPLY, error=type(exc).__name__,
                              latency_ms=int((time.perf_counter() - started) * 1000))
        latency_ms = int((time.perf_counter() - started) * 1000)

        new_messages = state["messages"][seen:]
        ai_messages = [m for m in new_messages if isinstance(m, AIMessage)]
        result = ChatResult(reply="", latency_ms=latency_ms)
        for m in ai_messages:
            result.tool_calls += [tc["name"] for tc in m.tool_calls]
            usage = m.usage_metadata or {}
            result.input_tokens += usage.get("input_tokens", 0)
            result.output_tokens += usage.get("output_tokens", 0)

        original = _text(ai_messages[-1]) if ai_messages else ""
        reply, reason = check_output(original or FALLBACK_REPLY)
        if reason is None:
            tool_results = [(m.name or "", _text(m)) for m in state["messages"] if isinstance(m, ToolMessage)]
            reply, reason = check_action_claims(reply, tool_results)
        if reason and ai_messages:
            # Replace the bad reply in memory too, so the model doesn't build on it next turn.
            self.graph.update_state(config, {"messages": [AIMessage(content=reply, id=ai_messages[-1].id)]}, as_node="agent")
        result.reply = reply
        result.guardrail = reason
        return result


def build_default_agent() -> BookingAgent:
    """Create an agent from environment settings (used by the API and CLI)."""
    from zoneinfo import ZoneInfo

    from langchain.chat_models import init_chat_model

    from .config import get_settings

    settings = get_settings()
    tz = ZoneInfo(settings.timezone)

    def now_local() -> datetime:
        return datetime.now(tz).replace(tzinfo=None, second=0, microsecond=0)

    llm = init_chat_model(settings.llm_model, temperature=0)
    store = AppointmentStore(settings.db_path, now_fn=now_local)
    return BookingAgent(llm, store, max_message_chars=settings.max_message_chars)
