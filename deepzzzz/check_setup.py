"""Check your setup and print what's wrong:  python -m deepzzzz.check_setup"""

from __future__ import annotations

import os
import sys
from pathlib import Path

KEY_NAMES = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
             "google_genai": "GOOGLE_API_KEY", "groq": "GROQ_API_KEY"}


def main() -> int:
    from dotenv import load_dotenv

    ok = True
    if Path(".env").is_file():
        print("[ok]   .env file found")
    else:
        print("[fail] No .env file in this folder. Run:  copy .env.example .env   (Mac/Linux: cp)")
        if Path(".env.txt").is_file():
            print("       Found .env.txt instead. Rename it to exactly .env")
        ok = False
    load_dotenv()

    from .config import get_settings

    settings = get_settings()
    model = settings.llm_model
    print(f"[info] LLM_MODEL = {model}")
    print(f"[info] USER_NAME = {settings.user_name}")
    provider = model.split(":", 1)[0]
    key_name = KEY_NAMES.get(provider)
    if key_name:
        key = os.getenv(key_name, "").strip()
        if not key:
            print(f"[fail] {key_name} is empty. Paste your key after '{key_name}=' in .env and save the file.")
            ok = False
        elif key.startswith(("'", '"')) or " " in key:
            print(f"[fail] {key_name} has quotes or spaces. Remove them.")
            ok = False
        elif "xxxx" in key or "PASTE" in key.upper():
            print(f"[fail] {key_name} is still the example placeholder. Paste your real key.")
            ok = False
        else:
            print(f"[ok]   {key_name} is set ({key[:7]}...{key[-4:]})")

    try:
        from zoneinfo import ZoneInfo

        ZoneInfo(settings.timezone)
        print(f"[ok]   Time zone {settings.timezone} works")
    except Exception:
        print(f"[fail] Time zone {settings.timezone} not found. Run:  pip install tzdata")
        ok = False

    if not ok:
        print("\nFix the [fail] lines above, then run this check again.")
        return 1

    try:
        from .agent import build_default

        bot = build_default()
    except Exception as exc:
        print(f"[fail] Could not start Deepzzzz: {exc}")
        return 1
    print("[ok]   Deepzzzz created. Sending a test message to the model...")
    try:
        bot.llm.invoke("Reply with the single word: ready")
    except Exception as exc:
        print(f"[fail] The model call failed: {type(exc).__name__}: {str(exc)[:300]}")
        if provider == "ollama":
            print("       Make sure the Ollama app is running and you ran:  ollama pull " + model.split(":", 1)[1])
        else:
            print("       Common causes: invalid key, no credit, a retired model name, or too many requests "
                  "per minute on a free tier (wait a minute).")
        return 1
    print("[ok]   The model answered.")
    print("\nAll good! Start the app with:  uvicorn deepzzzz.api:app --reload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
