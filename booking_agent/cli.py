"""Chat with the agent in your terminal:  python -m booking_agent.cli"""

from __future__ import annotations

import uuid

from .agent import build_default_agent


def main() -> None:
    agent = build_default_agent()
    session_id = uuid.uuid4().hex
    print("Bright Smile Dental assistant. Type 'quit' to exit.\n")
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if message.lower() in {"quit", "exit"}:
            break
        if not message:
            continue
        result = agent.chat(session_id, message)
        print(f"Agent: {result.reply}")
        meta = f"  [{result.latency_ms} ms"
        if result.tool_calls:
            meta += f", tools: {', '.join(result.tool_calls)}"
        if result.guardrail:
            meta += f", guardrail: {result.guardrail}"
        print(meta + "]\n")


if __name__ == "__main__":
    main()
