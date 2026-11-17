import sqlite3

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from deepzzzz.agent import FALLBACK_REPLY, Deepzzzz
from deepzzzz.api import create_app
from deepzzzz.store import Store

from .conftest import FIXED_NOW, ScriptedChatModel, tool_call


def test_remembers_a_fact_and_uses_it_next_turn(make_bot, store):
    bot, llm = make_bot([
        tool_call("remember_fact", {"fact": "Deepika is interviewing at Podium."}),
        AIMessage("Ooh, Podium! Got it, I'll remember that. Rooting for you! 🎉"),
        AIMessage("You've got your Podium interview coming up, right? How's prep going?"),
    ])
    r1 = bot.chat("I have an interview at Podium next week!")
    assert r1.tool_calls == ["remember_fact"] and r1.guardrail is None
    assert [m.content for m in store.list_memories()] == ["Deepika is interviewing at Podium."]

    bot.chat("hey")
    system = llm.received[-1][0]
    assert isinstance(system, SystemMessage) and "#1 Deepika is interviewing at Podium." in system.content


def test_false_memory_claim_is_corrected_in_reply_and_history(make_bot, store):
    bot, llm = make_bot([
        AIMessage("Got it, I'll remember that your birthday is May 3!"),
        AIMessage("ok!"),
    ])
    result = bot.chat("my birthday is May 3")
    assert result.guardrail == "unverified_action"
    assert "didn't actually do it" in result.reply
    assert store.list_memories() == []
    assert bot.history()[-1].content == result.reply
    bot.chat("yes do it")
    assert all("I'll remember that your birthday" not in str(m.content) for m in llm.received[-1])


def test_reminder_flow_and_tool_results_reach_model(make_bot, store):
    bot, llm = make_bot([
        tool_call("add_reminder", {"text": "Mock interview", "due": "2026-11-17T10:00"}),
        AIMessage("Done! I've set a reminder for tomorrow at 10 AM 💪"),
    ])
    result = bot.chat("remind me about my mock interview tomorrow at 10")
    assert result.guardrail is None
    assert store.list_reminders()[0].text == "Mock interview"
    assert any(isinstance(m, ToolMessage) and m.content.startswith("Reminder added") for m in llm.received[1])


def test_crisis_message_gets_care_note_and_resources(make_bot):
    bot, llm = make_bot([AIMessage("I'm really glad you told me. I'm here. What's been going on?"),
                         AIMessage("haha same")])
    result = bot.chat("honestly sometimes I want to die")
    assert result.guardrail == "crisis_support"
    assert "988" in result.reply
    assert any(isinstance(m, SystemMessage) and "may be in crisis" in m.content for m in llm.received[0])
    bot.chat("anyway what's for dinner")
    assert not any(isinstance(m, SystemMessage) and "may be in crisis" in m.content for m in llm.received[1])


def test_history_is_trimmed_for_the_model_but_kept_in_full(make_bot):
    bot, llm = make_bot([AIMessage(f"reply {i}") for i in range(6)], history_messages=4)
    for i in range(6):
        bot.chat(f"message {i}")
    sent = llm.received[-1]
    assert len(sent) == 1 + 3  # system + last human/ai/human (trimmed to start on a human message)
    assert sent[1].content == "message 4" and sent[-1].content == "message 5"
    assert len(bot.transcript()) == 12


def test_chat_history_survives_restart(tmp_path):
    db = str(tmp_path / "history.db")
    store = Store(str(tmp_path / "store.db"), now_fn=lambda: FIXED_NOW)

    def new_bot(responses):
        saver = SqliteSaver(sqlite3.connect(db, check_same_thread=False))
        return Deepzzzz(ScriptedChatModel(responses=responses), store, saver, now_fn=lambda: FIXED_NOW)

    new_bot([AIMessage("Nice to meet you!")]).chat("Hi, I'm Deepika")
    restarted = new_bot([])
    assert restarted.transcript() == [{"role": "user", "text": "Hi, I'm Deepika"},
                                      {"role": "assistant", "text": "Nice to meet you!"}]
    restarted.reset()
    assert restarted.transcript() == []


def test_errors_and_runaway_loops_fall_back(make_bot):
    bot, _ = make_bot([])
    assert bot.chat("hello").reply == FALLBACK_REPLY
    looping, _ = make_bot([tool_call("list_reminders", {}, call_id=f"c{i}") for i in range(40)])
    result = looping.chat("what's on my list?")
    assert result.error == "recursion_limit" and result.reply == FALLBACK_REPLY


def test_api_endpoints(make_bot, store):
    bot, _ = make_bot([AIMessage("Hey Deepika! 👋")])
    store.add_memory("Deepika likes chai.")
    store.add_reminder("Stretch")
    client = TestClient(create_app(bot))

    assert client.get("/health").json() == {"status": "ok"}
    body = client.post("/chat", json={"message": "hi"}).json()
    assert body["reply"] == "Hey Deepika! 👋"
    assert client.get("/history").json()[-1] == {"role": "assistant", "text": "Hey Deepika! 👋"}
    assert client.get("/memories").json() == [{"id": 1, "content": "Deepika likes chai."}]
    assert client.get("/reminders").json()[0]["text"] == "Stretch"
    assert client.post("/reminders/1/done").json() == {"status": "done"}
    assert client.get("/reminders").json() == []
    assert client.delete("/memories/1").json() == {"status": "deleted"}
    assert client.delete("/memories/1").status_code == 404
    assert client.post("/reset").json() == {"status": "reset"}
    assert client.get("/history").json() == []
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert "Deepzzzz" in client.get("/").text


def test_api_explains_missing_setup(monkeypatch):
    import deepzzzz.api as api

    def broken():
        raise RuntimeError("Missing GROQ_API_KEY")

    monkeypatch.setattr(api, "build_default", broken)
    res = TestClient(api.create_app()).post("/chat", json={"message": "hi"})
    assert res.status_code == 503 and "GROQ_API_KEY" in res.json()["detail"]
