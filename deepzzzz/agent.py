"""Deepzzzz: a LangGraph agent with long-term memory, persistent chat history and tools.

Graph:   START -> agent --(tool calls?)--> tools -> agent ... --(no tool calls)--> END

- Chat history is saved by a LangGraph checkpointer (SQLite in the app), keyed by thread_id,
  so conversations survive restarts. Only the most recent messages go to the model each turn.
- Long-term facts ("memories"), due reminders and a calendar are built into the system
  prompt fresh on every model call, so Deepzzzz always knows what it saved.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .guardrails import CRISIS_NOTE, check_action_claims, check_input, check_output, ensure_crisis_resources
from .prompts import build_system_prompt
from .store import Store
from .tools import make_tools

log = logging.getLogger(__name__)

RECURSION_LIMIT = 16
DEFAULT_THREAD = "main"
FALLBACK_REPLY = "Ugh, my brain glitched for a sec 😵 Can you say that again?"


class ChatState(MessagesState):
    crisis: bool  # sent with every user message; True only for turns the input guardrail flags


@dataclass
class ChatResult:
    reply: str
    tool_calls: list[str] = field(default_factory=list)
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    guardrail: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


class Deepzzzz:
    def __init__(
        self,
        llm: BaseChatModel,
        store: Store,
        checkpointer: BaseCheckpointSaver | None = None,
        user_name: str = "friend",
        max_message_chars: int = 4000,
        history_messages: int = 40,
        now_fn: Callable[[], datetime] | None = None,
    ):
        self.store = store
        self.user_name = user_name
        self.max_message_chars = max_message_chars
        self.history_messages = history_messages
        self.now = now_fn or store.now
        self.checkpointer = checkpointer or InMemorySaver()
        self.tools = make_tools(store)
        self.llm = llm.bind_tools(self.tools)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ChatState)
        graph.add_node("agent", self._call_model)
        graph.add_node("tools", ToolNode(self.tools))
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition)
        graph.add_edge("tools", "agent")
        return graph.compile(checkpointer=self.checkpointer)

    def system_prompt(self) -> str:
        recent = [m for day in self.store.mood_days(7) for m in day.entries]
        return build_system_prompt(self.now(), self.user_name, self.store.list_memories(), self.store.due_reminders(),
                                   recent_moods=recent, logged_today=self.store.logged_mood_today())

    def _call_model(self, state: ChatState) -> dict:
        # Keep the last N messages, starting on a human turn so tool calls stay paired with their results.
        recent = trim_messages(
            state["messages"], max_tokens=self.history_messages, token_counter=len,
            strategy="last", start_on="human", include_system=False, allow_partial=False,
        )
        messages: list[BaseMessage] = [SystemMessage(self.system_prompt())]
        if state.get("crisis"):
            messages.append(SystemMessage(CRISIS_NOTE))
        response = self.llm.invoke([*messages, *recent])
        return {"messages": [response]}

    # ---------- public API ----------

    def history(self, thread_id: str = DEFAULT_THREAD) -> list[BaseMessage]:
        state = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        return list(state.values.get("messages", [])) if state and state.values else []

    def transcript(self, thread_id: str = DEFAULT_THREAD) -> list[dict]:
        """User-visible messages only (no tool calls or tool results)."""
        out = []
        for m in self.history(thread_id):
            if isinstance(m, HumanMessage):
                out.append({"role": "user", "text": _text(m)})
            elif isinstance(m, AIMessage) and _text(m).strip():
                out.append({"role": "assistant", "text": _text(m)})
        return out

    def reset(self, thread_id: str = DEFAULT_THREAD) -> None:
        """Start a fresh conversation. Memories, reminders and notes are kept."""
        self.checkpointer.delete_thread(thread_id)

    def chat(self, message: str, thread_id: str = DEFAULT_THREAD) -> ChatResult:
        guard = check_input(message, self.max_message_chars)
        if not guard.allowed:
            return ChatResult(reply=guard.reply or "", guardrail=guard.reason)

        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
        seen = len(self.history(thread_id))
        started = time.perf_counter()
        try:
            state = self.graph.invoke({"messages": [HumanMessage(message)], "crisis": guard.crisis}, config=config)
        except GraphRecursionError:
            log.warning("Recursion limit hit")
            return ChatResult(reply=FALLBACK_REPLY, error="recursion_limit",
                              latency_ms=int((time.perf_counter() - started) * 1000))
        except Exception as exc:
            log.exception("Model call failed")
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
            turn_results = [(m.name or "", _text(m)) for m in new_messages if isinstance(m, ToolMessage)]
            reply, reason = check_action_claims(reply, turn_results)
        if guard.crisis:
            reply = ensure_crisis_resources(reply)
            reason = reason or guard.reason
        if reply != original and ai_messages:
            # Keep memory consistent with what the user actually saw.
            self.graph.update_state(config, {"messages": [AIMessage(content=reply, id=ai_messages[-1].id)]},
                                    as_node="agent")
        result.reply = reply
        result.guardrail = reason
        return result


def build_default() -> Deepzzzz:
    """Create Deepzzzz from .env settings, with chat history saved to disk."""
    import sqlite3
    from pathlib import Path
    from zoneinfo import ZoneInfo

    from langchain.chat_models import init_chat_model
    from langgraph.checkpoint.sqlite import SqliteSaver

    from .config import get_settings

    settings = get_settings()
    tz = ZoneInfo(settings.timezone)

    def now_local() -> datetime:
        return datetime.now(tz).replace(tzinfo=None, second=0, microsecond=0)

    data = Path(settings.data_dir)
    data.mkdir(parents=True, exist_ok=True)
    llm = init_chat_model(settings.llm_model, temperature=0.7)
    store = Store(str(data / "deepzzzz.db"), now_fn=now_local)
    checkpointer = SqliteSaver(sqlite3.connect(str(data / "chat_history.db"), check_same_thread=False))
    return Deepzzzz(llm, store, checkpointer, user_name=settings.user_name,
                    max_message_chars=settings.max_message_chars, history_messages=settings.history_messages)
