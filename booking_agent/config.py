from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    llm_model: str
    db_path: str
    timezone: str
    max_message_chars: int


def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        # Any "provider:model" string supported by LangChain's init_chat_model,
        # e.g. "openai:gpt-4o-mini" or an Anthropic Claude model.
        llm_model=os.getenv("LLM_MODEL", "openai:gpt-4o-mini"),
        db_path=os.getenv("DB_PATH", "data/appointments.db"),
        timezone=os.getenv("CLINIC_TIMEZONE", "America/Los_Angeles"),
        max_message_chars=int(os.getenv("MAX_MESSAGE_CHARS", "1000")),
    )
