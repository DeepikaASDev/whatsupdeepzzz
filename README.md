# AI Booking Agent

An AI front-desk agent for a dental clinic. Patients chat with it to ask about services, hours and prices, check open times, and **book, reschedule or cancel appointments**. It is built with **LangGraph**, uses **tool calling** against a real appointment database, and has **guardrails** at three layers.

![tests](https://github.com/DeepikaASDev/ai-booking-agent/actions/workflows/tests.yml/badge.svg)

## Example conversation

```text
You:   Do you have any cleaning slots next Tuesday?
Agent: [check_availability] Tuesday Nov 17 has 9:00, 9:30 and 10:00 AM open. Which works?
       I'll also need your name and phone number.
You:   9am. Ana Lopez, 408-555-0101
Agent: Please confirm: teeth cleaning for Ana Lopez (408-555-0101) on Tue Nov 17 at 9:00 AM?
You:   Yes
Agent: [book_appointment] You're booked! Your appointment ID is 3F9A21C0.
```

## How it works

```text
             ┌──────────────────┐
 message ──▶ │ input guardrails │──(emergency / injection / too long)──▶ fixed safe reply
             └────────┬─────────┘
                      ▼
             ┌──────────────────┐  tool calls   ┌──────────────────────────┐
             │   LLM agent node │ ────────────▶ │ tools (ToolNode)         │
             │  (system prompt) │ ◀──────────── │ clinic info, availability│
             └────────┬─────────┘  tool results │ book, find, reschedule,  │
                      │ final answer            │ cancel  →  SQLite store  │
                      ▼                         └──────────────────────────┘
             ┌──────────────────┐
             │ output guardrail │──▶ reply + latency, tokens, tools used
             └──────────────────┘
```

- **LangGraph state graph** (`booking_agent/agent.py`): an agent node and a tool node in a loop, with per-session conversation memory through a checkpointer (`thread_id = session_id`) and a recursion limit that stops runaway tool loops.
- **Tools** (`booking_agent/tools.py`): six tools the model calls with structured arguments. Errors come back as readable text so the agent can recover ("that time is taken, here are other options").
- **Business rules in code, not in the prompt** (`booking_agent/store.py`): opening hours, service durations, no double-booking, no past bookings, a 1-hour minimum lead time, and phone-number ownership checks for changes. The model can't book an invalid slot even if it tries.
- **Guardrails in three layers:**
  1. *Input* (`guardrails.py`): blocks prompt-injection attempts and oversized messages, and sends possible medical emergencies a fixed "call 911" message without calling the model.
  2. *Tools*: booking, rescheduling and cancelling all require `patient_confirmed=true`, which the prompt allows only after the patient says yes to a summary.
  3. *Output*: a canary token in the system prompt detects and blocks prompt leaks.
- **Context engineering** (`prompts.py`): today's date and weekday go into the prompt so "next Tuesday" resolves correctly, and every clinic fact comes from a tool rather than the prompt.
- **Observability**: every reply returns latency, input/output tokens, tools called, and which guardrail fired, if any.
- **Provider-agnostic**: works with Google Gemini (free tier), OpenAI or Anthropic models through LangChain's `init_chat_model`, by changing one line in `.env`.

## Run it

```bash
git clone https://github.com/DeepikaASDev/ai-booking-agent.git
cd ai-booking-agent
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env          # then add a free GOOGLE_API_KEY (or OpenAI/Anthropic settings)
```

**Check your setup first:** `python -m booking_agent.check_setup` tells you exactly what is missing (key, .env file, time zone) and sends one test message to the model.

**Web chat:** `uvicorn booking_agent.api:app --reload`, then open http://localhost:8000

**Terminal chat:** `python -m booking_agent.cli`

**Docker:**

```bash
docker build -t ai-booking-agent .
docker run -p 8000:8000 --env-file .env ai-booking-agent
```

### API

`POST /chat`

```json
{ "session_id": "abc123", "message": "Can I book a checkup on Friday?" }
```

Returns:

```json
{ "reply": "...", "tool_calls": ["check_availability"], "latency_ms": 1240,
  "input_tokens": 1830, "output_tokens": 64, "guardrail": null, "error": null }
```

`GET /health` returns `{"status": "ok"}`.

## Tests

```bash
pytest -q
```

The 20 tests run without an API key. A scripted fake chat model drives the **real** LangGraph loop, so the tests check tool execution, conversation memory, guardrails, error fallbacks, the recursion limit, the booking rules and the API. GitHub Actions runs them, plus a Docker build, on every push.

## Project layout

```text
booking_agent/
  agent.py        LangGraph graph, memory, metrics, error handling
  tools.py        tools the LLM can call
  store.py        SQLite calendar + booking rules
  guardrails.py   input/output guardrails
  prompts.py      system prompt
  clinic.py       clinic hours, services, prices, policies
  api.py          FastAPI app
  cli.py          terminal chat
  static/         web chat page
tests/            pytest suite with a scripted fake LLM
```

## Next

- Evaluation suite: scripted conversations scored on booking accuracy, guardrail compliance, latency and cost per conversation.
- SMS channel through a Twilio webhook.
