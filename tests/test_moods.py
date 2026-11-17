import sqlite3
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from deepzzzz.api import create_app
from deepzzzz.prompts import build_system_prompt
from deepzzzz.store import Store, StoreError
from deepzzzz.tools import make_tools

from .conftest import FIXED_NOW, tool_call


def _seed_week(store, clock):
    """Mood check-ins on Tue 10, Thu 12 (two), Sat 14 and Mon 16 Nov 2026."""
    for day, hour, score, label, note in [(10, 9, 2, "stressed", "two interviews"), (12, 9, 4, "excited", "callback"),
                                          (12, 20, 5, "happy", ""), (14, 9, 3, "tired", ""), (16, 9, 2, "stressed", "")]:
        clock[0] = datetime(2026, 11, day, hour)
        store.log_mood(score, label, note)
    clock[0] = FIXED_NOW


@pytest.fixture
def clock_store():
    clock = [FIXED_NOW]
    return Store(":memory:", now_fn=lambda: clock[0]), clock


def test_log_mood_validates(store):
    mood = store.log_mood(4, "  Excited ", "got a callback")
    assert (mood.score, mood.label, mood.note) == (4, "excited", "got a callback")
    for bad in [0, 6, "high"]:
        with pytest.raises(StoreError):
            store.log_mood(bad, "x")
    with pytest.raises(StoreError):
        store.log_mood(3, "  ")


def test_mood_days_and_week_summary(clock_store):
    store, clock = clock_store
    _seed_week(store, clock)
    days = store.mood_days(7)
    assert [d.day.day for d in days] == [10, 12, 14, 16]
    assert days[1].average == 4.5 and len(days[1].entries) == 2
    assert store.logged_mood_today()

    r = store.add_reminder("Practice STAR")
    store.complete_reminder(r.id)
    store.add_reminder("Email recruiter", datetime(2026, 11, 15, 9, 0))
    store.add_memory("Deepika got a callback from Podium.")
    store.add_note("Interview prep", "Hallucinated booking story")

    w = store.week_summary(7)
    assert (w.start.day, w.end.day, w.check_ins, w.average) == (10, 16, 5, 3.2)
    assert w.top_labels[0] == ("stressed", 2)
    assert w.best_day.day.day == 12 and w.hardest_day.day.day == 10
    assert w.reminders_completed == ["Practice STAR"]
    assert w.reminders_overdue == ["Email recruiter"]
    assert w.memories_added == ["Deepika got a callback from Podium."]
    assert w.notes_added == ["Interview prep"]


def test_week_summary_ignores_older_check_ins(clock_store):
    store, clock = clock_store
    clock[0] = datetime(2026, 11, 2, 9)
    store.log_mood(1, "awful")
    clock[0] = FIXED_NOW
    w = store.week_summary(7)
    assert w.check_ins == 0 and w.average is None and w.best_day is None
    assert not store.logged_mood_today()


def test_recap_tool_formats_data(clock_store):
    store, clock = clock_store
    _seed_week(store, clock)
    tools = {t.name: t for t in make_tools(store)}
    out = tools["get_weekly_recap"].invoke({})
    assert "Mood check-ins: 5, average 3.2/5" in out
    assert "Best day: Thursday" in out and "Hardest day: Tuesday" in out
    assert "- Thu Nov 12: excited 4/5 (callback); happy 5/5" in out
    one_day = tools["get_weekly_recap"].invoke({"days": 1})
    assert "Recap data for Mon Nov 16 - Mon Nov 16" in one_day
    assert "- Mon Nov 16: stressed 2/5" in one_day and "Thu Nov 12" not in one_day
    assert "No mood check-ins" in make_tools(Store(":memory:", now_fn=lambda: FIXED_NOW))[-1].invoke({})
    assert tools["log_mood"].invoke({"score": 9, "label": "x"}).startswith("Error:")


def test_prompt_mood_status():
    sunday = datetime(2026, 11, 22, 10, 0)
    prompt = build_system_prompt(sunday, "Deepika", [], [], [], logged_today=False)
    assert "casually ask how Deepika is feeling" in prompt and "It's Sunday" in prompt
    prompt = build_system_prompt(FIXED_NOW, "Deepika", [], [], [], logged_today=True)
    assert "Already checked in today" in prompt and "It's Sunday" not in prompt


def test_agent_logs_mood_and_writes_recap(make_bot, store):
    bot, llm = make_bot([
        tool_call("log_mood", {"score": 2, "label": "nervous", "note": "Podium round 2"}),
        AIMessage("Nervous is normal before round 2! I've logged your mood. Want to practice a question?"),
        tool_call("get_weekly_recap", {}, call_id="call_2"),
        AIMessage("Your week: one check-in so far, feeling nervous about Podium. You've got this! 💪"),
    ])
    first = bot.chat("feeling nervous about Podium round 2")
    assert first.tool_calls == ["log_mood"] and first.guardrail is None
    assert store.logged_mood_today()

    recap = bot.chat("how was my week?")
    assert recap.tool_calls == ["get_weekly_recap"]
    tool_msg = [m for m in llm.received[-1] if isinstance(m, ToolMessage)][-1]
    assert "nervous 2/5 (Podium round 2)" in tool_msg.content
    system = llm.received[-1][0]
    assert isinstance(system, SystemMessage) and "Already checked in today" in system.content


def test_unbacked_mood_claim_is_caught(make_bot, store):
    bot, _ = make_bot([AIMessage("Aw, I've logged your mood as tired. Rest up!")])
    result = bot.chat("so tired today")
    assert result.guardrail == "unverified_action"
    assert not store.logged_mood_today()


def test_moods_endpoint(make_bot, clock_store):
    store, clock = clock_store
    _seed_week(store, clock)
    bot, _ = make_bot([])
    bot.store = store
    client = TestClient(create_app(bot))
    data = client.get("/moods?days=7").json()
    assert [d["date"] for d in data] == ["2026-11-10", "2026-11-12", "2026-11-14", "2026-11-16"]
    assert data[1]["average"] == 4.5
    assert data[1]["entries"][0] == {"score": 4, "label": "excited", "note": "callback", "time": "09:00"}
    assert len(client.get("/moods?days=1").json()) == 1


def test_old_database_gets_completed_at_column(tmp_path):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE reminders (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, due TEXT,"
                 " done INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL)")
    conn.execute("INSERT INTO reminders (text, created_at) VALUES ('Old one', '2026-11-01T09:00')")
    conn.commit()
    conn.close()
    store = Store(str(db), now_fn=lambda: FIXED_NOW)
    store.complete_reminder(1)
    assert store.week_summary(7).reminders_completed == ["Old one"]
