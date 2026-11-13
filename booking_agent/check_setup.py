"""Check your setup and print what's wrong:  python -m booking_agent.check_setup"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    from dotenv import load_dotenv

    ok = True
    env_file = Path(".env")
    if env_file.is_file():
        print("[ok]   .env file found")
    else:
        print("[fail] No .env file in this folder. Run:  copy .env.example .env   (Mac/Linux: cp)")
        ok = False
        if Path(".env.txt").is_file():
            print("       Found .env.txt instead. Rename it to exactly .env")
    load_dotenv()

    model = os.getenv("LLM_MODEL", "openai:gpt-4o-mini")
    print(f"[info] LLM_MODEL = {model}")
    provider = model.split(":", 1)[0] if ":" in model else "openai"
    key_name = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(provider)
    if key_name:
        key = os.getenv(key_name, "").strip()
        if not key:
            print(f"[fail] {key_name} is empty. Paste your key after '{key_name}=' in .env and save the file.")
            ok = False
        elif key.startswith(("'", '"')) or " " in key:
            print(f"[fail] {key_name} has quotes or spaces. Remove them.")
            ok = False
        else:
            print(f"[ok]   {key_name} is set ({key[:7]}...{key[-4:]})")

    tz = os.getenv("CLINIC_TIMEZONE", "America/Los_Angeles")
    try:
        from zoneinfo import ZoneInfo

        ZoneInfo(tz)
        print(f"[ok]   Time zone {tz} works")
    except Exception:
        print(f"[fail] Time zone {tz} not found. Run:  pip install tzdata")
        ok = False

    if ok:
        try:
            from .agent import build_default_agent

            agent = build_default_agent()
        except Exception as exc:
            print(f"[fail] Could not start the agent: {exc}")
            return 1
        print("[ok]   Agent created. Sending a test message to the model...")
        try:
            agent.llm.invoke("Reply with the single word: ready")
        except Exception as exc:
            print(f"[fail] The model call failed: {type(exc).__name__}: {str(exc)[:300]}")
            print("       Common causes: invalid key, no API credit/billing, or a wrong model name in LLM_MODEL.")
            return 1
        result = agent.chat("setup-check", "What are your hours?")
        print(f"[ok]   Agent replied in {result.latency_ms} ms: {result.reply[:120]}")

    print("\nAll good! Start the app with:  uvicorn booking_agent.api:app --reload" if ok
          else "\nFix the [fail] lines above, then run this check again.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
