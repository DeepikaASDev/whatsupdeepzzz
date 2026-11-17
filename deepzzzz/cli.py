"""Chat with Deepzzzz in your terminal:  python -m deepzzzz.cli"""

from __future__ import annotations

from .agent import build_default


def main() -> None:
    bot = build_default()
    print("Deepzzzz is here! Type 'quit' to exit, '/reset' for a new chat.\n")
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if message.lower() in {"quit", "exit"}:
            break
        if message == "/reset":
            bot.reset()
            print("Deepzzzz: Fresh start! What's up?\n")
            continue
        if not message:
            continue
        result = bot.chat(message)
        print(f"Deepzzzz: {result.reply}")
        meta = f"  [{result.latency_ms / 1000:.1f}s"
        if result.tool_calls:
            meta += f", tools: {', '.join(result.tool_calls)}"
        if result.guardrail:
            meta += f", guardrail: {result.guardrail}"
        print(meta + "]\n")


if __name__ == "__main__":
    main()
