"""Deepzzzz: a personal AI companion with memory, reminders and notes, built with LangGraph."""

from .agent import ChatResult, Deepzzzz, build_default
from .store import Store, StoreError

__all__ = ["ChatResult", "Deepzzzz", "build_default", "Store", "StoreError"]
