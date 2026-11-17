from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    llm_model: str
    data_dir: str
    timezone: str
    user_name: str
    max_message_chars: int
    history_messages: int


def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        # Any "provider:model" string supported by LangChain's init_chat_model.
        llm_model=os.getenv("LLM_MODEL", "groq:llama-3.3-70b-versatile"),
        data_dir=os.getenv("DATA_DIR", "data"),
        timezone=os.getenv("TIMEZONE", "America/Los_Angeles"),
        user_name=os.getenv("USER_NAME", "friend"),
        max_message_chars=int(os.getenv("MAX_MESSAGE_CHARS", "4000")),
        # How many recent messages the model sees each turn (long-term facts live in memory).
        history_messages=int(os.getenv("HISTORY_MESSAGES", "40")),
    )
