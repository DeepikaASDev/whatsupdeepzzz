from datetime import datetime

import pytest

from deepzzzz.prompts import build_system_prompt
from deepzzzz.store import StoreError
from deepzzzz.tools import make_tools

from .conftest import FIXED_NOW


def _tools(store):
    return {t.name: t for t in make_tools(store)}


def test_memories_dedupe_and_delete(store):
    a = store.add_memory("Deepika is preparing for AI engineer interviews.")
    b = store.add_memory("deepika is preparing for AI engineer interviews.")
    assert a.id == b.id and len(store.list_memories()) == 1
    store.delete_memory(a.id)
    assert store.list_memories() == []
    with pytest.raises(StoreError):
        store.delete_memory(a.id)
    with pytest.raises(StoreError):
        store.add_memory("   ")


def test_reminders_order_due_and_complete(store):
    later = store.add_reminder("Mock interview", datetime(2026, 11, 20, 10, 0))
    undated = store.add_reminder("Buy groceries")
    overdue = store.add_reminder("Call mom", datetime(2026, 11, 15, 19, 0))
    today = store.add_reminder("Gym", datetime(2026, 11, 16, 20, 0))
    assert [r.id for r in store.list_reminders()] == [overdue.id, today.id, later.id, undated.id]
    assert [r.id for r in store.due_reminders()] == [overdue.id, today.id]
    store.complete_reminder(overdue.id)
    assert overdue.id not in [r.id for r in store.list_reminders()]
    assert overdue.id in [r.id for r in store.list_reminders(include_done=True)]
    with pytest.raises(StoreError, match="already done"):
        store.complete_reminder(overdue.id)


def test_tools_return_checkable_results(store):
    t = _tools(store)
    assert t["remember_fact"].invoke({"fact": "Deepika's favorite food is biryani."}).startswith("Saved memory #1")
    assert t["forget_fact"].invoke({"memory_id": 99}).startswith("Error:")
    out = t["add_reminder"].invoke({"text": "Apply to Podium", "due": "2026-11-17"})
    assert out.startswith("Reminder added #1") and "9:00 AM" in out  # date only -> 9 AM
    assert t["add_reminder"].invoke({"text": "x", "due": "next friday"}).startswith("Error:")
    assert "Apply to Podium" in t["list_reminders"].invoke({})
    assert t["complete_reminder"].invoke({"reminder_id": 1}).startswith("Completed reminder #1")
    assert t["list_reminders"].invoke({}) == "No reminders."
    assert t["save_note"].invoke({"title": "STAR stories", "content": "Hallucinated booking fix"}).startswith("Note saved #1")
    assert "Hallucinated booking" in t["search_notes"].invoke({"query": "booking"})
    assert t["search_notes"].invoke({"query": "zzz"}) == "No matching notes."


def test_system_prompt_includes_memories_due_reminders_and_calendar(store):
    store.add_memory("Deepika has a sister named Priya.")
    store.add_reminder("Call mom", datetime(2026, 11, 16, 19, 0))
    store.add_reminder("Far future", datetime(2026, 12, 1, 9, 0))
    prompt = build_system_prompt(FIXED_NOW, "Deepika", store.list_memories(), store.due_reminders())
    assert "You are Deepzzzz, Deepika's personal AI companion" in prompt
    assert "- #1 Deepika has a sister named Priya." in prompt
    assert "Call mom" in prompt and "Far future" not in prompt
    assert "- Tuesday = 2026-11-17 (tomorrow)" in prompt
    assert "You are an AI, not a human" in prompt


def test_empty_prompt_sections(store):
    prompt = build_system_prompt(FIXED_NOW, "Deepika", [], [])
    assert "(nothing yet: get to know them!)" in prompt and "- (none)" in prompt
