# What's up, Deepzzzz!!!!! 👋

**Deepzzzz** is my personal AI companion with memory, mood check-ins and reminders.

It's an AI friend you can chat with. It **remembers what you tell it across conversations**, does **mood check-ins with a weekly recap**, keeps your **reminders and notes**, and talks like a warm, upbeat friend. It is built with **LangGraph** and **tool calling**, stores everything locally in **SQLite**, and has **guardrails** that run in code, so they work even when the model slips.

![tests](https://github.com/DeepikaASDev/whatsupdeepzzz/actions/workflows/tests.yml/badge.svg)

> This project grew out of an AI appointment-booking agent for a dental clinic, which is kept on the [`booking-agent-v1`](https://github.com/DeepikaASDev/whatsupdeepzzz/tree/booking-agent-v1) branch.

## Example conversation

```text
You:       I have an interview at Podium next week, kind of nervous
Deepzzzz:  [remember_fact] Ooh, Podium! That's exciting. Nervous is normal, it means you care.
           Want to do a quick mock question together?
You:       remind me to practice STAR stories tomorrow at 7pm
Deepzzzz:  [add_reminder] Done! Reminder set for Tue Nov 17 at 7:00 PM 💪

You:       ugh I'm exhausted, barely slept
Deepzzzz:  [log_mood: tired 2/5] Oof, that's rough before a big week. Be gentle with yourself today 💜

(Sunday)
You:       how was my week?
Deepzzzz:  [get_weekly_recap] Big week! 5 check-ins, avg 3.2/5. Thursday was your high point (the
           Podium callback 🎉), Tuesday was the toughest with two interviews back to back. You also
           finished your STAR practice. Proud of you, keep that momentum!

(next day, after restarting the app)
You:       hey
Deepzzzz:  Hey Deepika! Don't forget your STAR stories practice tonight. How are you feeling about Podium?
```

## How it works

```text
 you ──▶ input guardrails ──▶ LangGraph agent ◀──▶ tools ──▶ SQLite (memories, reminders, notes)
           (length, crisis)        │   ▲
                                   │   └── system prompt rebuilt every turn:
                                   │       persona + your memories + due reminders + 14-day calendar
                                   ▼
                          output guardrails ──▶ reply + latency, tokens, tools used
                (prompt-leak canary, action-claim check)
```

- **Two kinds of memory.**
  - *Short-term:* the full chat history is saved by LangGraph's **SQLite checkpointer**, so conversations survive restarts. Each turn the model sees only the last 40 messages (`trim_messages`, starting on a user turn so tool calls stay paired with their results). That keeps cost and latency flat as the history grows.
  - *Long-term:* facts Deepzzzz chooses to save with `remember_fact` (your goals, people, likes) are stored in SQLite and **rebuilt into the system prompt on every turn**, so it remembers them even after you start a new chat.
- **Nine tools:** `remember_fact`, `forget_fact`, `add_reminder`, `list_reminders`, `complete_reminder`, `save_note`, `search_notes`, `log_mood`, `get_weekly_recap`. Errors come back as text, so the model can explain what went wrong.
- **Mood check-ins and weekly recap:**
  - *Structured extraction:* when you say how you feel ("ugh, exhausted"), the model turns it into a record: score 1–5, a label and a short reason, saved with a timestamp.
  - *Proactive check-ins:* the prompt says whether you've checked in today, so Deepzzzz asks once a day, naturally. On Sundays it offers a recap.
  - *Recap = SQL + LLM:* `get_weekly_recap` computes the numbers in code (daily averages, best and hardest day, top feelings, reminders finished, new memories), and the LLM only writes the friendly summary. That way the numbers are always right.
  - *Mood chart:* the side panel shows the last 14 days as colored bars (red = low, green = great). Hover a bar for that day's check-ins.
- **Proactive reminders:** reminders due today or overdue go into the prompt, so Deepzzzz brings them up on its own.
- **Guardrails:**
  1. *Crisis support:* if a message suggests self-harm, the model gets an extra instruction to respond with care, and the reply is **guaranteed** to include the 988 Suicide & Crisis Lifeline.
  2. *Action-claim check:* if Deepzzzz says "I'll remember that", "reminder set" or "I've logged your mood" but the matching tool didn't succeed in that turn, the reply is replaced with an honest one, and memory is corrected too. This came from the booking-agent version, where a small model said "you're booked" after the booking tool had failed.
  3. *Prompt-leak canary* and message-length limits.
- **Honest persona:** friendly and casual, but it always says it's an AI if asked, and encourages talking to people you trust when things are heavy.
- **Provider-agnostic:** Groq (free, fast), Ollama (local, no key), Gemini, OpenAI or Anthropic through LangChain's `init_chat_model`. Switching takes one line in `.env`.

## Run it

```bash
git clone https://github.com/DeepikaASDev/whatsupdeepzzz.git
cd whatsupdeepzzz
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env     # Windows: copy .env.example .env ; then add your GROQ_API_KEY and USER_NAME
python -m deepzzzz.check_setup        # tells you exactly what's missing
uvicorn deepzzzz.api:app --reload     # then open http://localhost:8000
```

The web app shows the chat plus a side panel with your **mood chart**, **what Deepzzzz remembers about you** (click ✕ to make it forget) and your **reminders** (click ✓ to complete). **Weekly recap** asks for your recap. **New chat** clears the conversation but keeps memories, moods and reminders.

Terminal chat: `python -m deepzzzz.cli`

Docker: `docker build -t deepzzzz . && docker run -p 8000:8000 --env-file .env -v deepzzzz-data:/app/data deepzzzz`

Your data stays on your machine in `data/` (`deepzzzz.db` for memories, reminders and notes; `chat_history.db` for conversations). It is git-ignored.

### API

| Method | Path | What it does |
|---|---|---|
| POST | `/chat` | `{"message": "..."}` → reply, tools used, latency, tokens, guardrail |
| GET | `/history` | The conversation so far |
| POST | `/reset` | Start a new chat (memories kept) |
| GET / DELETE | `/memories`, `/memories/{id}` | See or delete what Deepzzzz remembers |
| GET / POST | `/reminders`, `/reminders/{id}/done` | See or complete reminders |
| GET | `/moods?days=14` | Mood check-ins grouped by day, with daily averages |

## Tests

```bash
pytest -q
```

The 27 tests need no API key: a scripted fake chat model drives the **real** LangGraph loop, tools and SQLite storage. They cover memory saving and recall, mood logging and the weekly recap math, the action-claim check, the crisis flow, upgrading an older database, history trimming, chat history surviving a restart, error fallbacks, the recursion limit and every API endpoint. GitHub Actions runs them plus a Docker build on every push.

## Project layout

```text
deepzzzz/
  agent.py         LangGraph graph, persistent history, trimming, guardrail wiring, metrics
  tools.py         the 9 tools
  store.py         SQLite: memories, reminders, notes, moods + weekly summary
  prompts.py       persona + memories + due reminders + mood status + calendar
  guardrails.py    crisis support, action-claim check, leak canary
  api.py           FastAPI app
  cli.py           terminal chat
  check_setup.py   setup checker
  static/          web chat with mood chart, memory and reminder panel
tests/             pytest suite with a scripted fake LLM
```

## Ideas for next

- Semantic memory search with embeddings once there are hundreds of memories
- Reminder notifications (desktop or SMS) instead of only in-chat mentions
- An evaluation set of conversations to compare models on memory recall and honesty
