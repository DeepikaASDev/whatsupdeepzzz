"""Long-term storage for Deepzzzz: memories about you, reminders and notes (SQLite).

The chat history itself is stored separately by LangGraph's SQLite checkpointer.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

MAX_MEMORIES_IN_PROMPT = 60
MAX_TEXT = 500


class StoreError(Exception):
    """Invalid input. The message is safe to show the user."""


@dataclass
class Memory:
    id: int
    content: str
    created_at: datetime


@dataclass
class Reminder:
    id: int
    text: str
    due: datetime | None
    done: bool


@dataclass
class Note:
    id: int
    title: str
    content: str
    created_at: datetime


def _clean(text: str, what: str) -> str:
    text = " ".join((text or "").split())
    if not text:
        raise StoreError(f"The {what} is empty.")
    if len(text) > MAX_TEXT:
        raise StoreError(f"The {what} is too long (max {MAX_TEXT} characters).")
    return text


class Store:
    def __init__(self, db_path: str = ":memory:", now_fn: Callable[[], datetime] | None = None):
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self.now = now_fn or datetime.now
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                due TEXT,
                done INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    # ---------- memories ----------

    def add_memory(self, content: str) -> Memory:
        content = _clean(content, "memory")
        with self._lock:
            existing = self._conn.execute(
                "SELECT id FROM memories WHERE lower(content) = lower(?)", (content,)
            ).fetchone()
            if existing:
                return self.get_memory(existing["id"])  # already known; don't duplicate
            cur = self._conn.execute(
                "INSERT INTO memories (content, created_at) VALUES (?, ?)", (content, self.now().isoformat())
            )
            self._conn.commit()
        return self.get_memory(cur.lastrowid)

    def get_memory(self, memory_id: int) -> Memory:
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            raise StoreError(f"No memory with ID {memory_id}.")
        return Memory(row["id"], row["content"], datetime.fromisoformat(row["created_at"]))

    def list_memories(self, limit: int = MAX_MEMORIES_IN_PROMPT) -> list[Memory]:
        rows = self._conn.execute(
            "SELECT * FROM (SELECT * FROM memories ORDER BY id DESC LIMIT ?) ORDER BY id", (limit,)
        ).fetchall()
        return [Memory(r["id"], r["content"], datetime.fromisoformat(r["created_at"])) for r in rows]

    def delete_memory(self, memory_id: int) -> Memory:
        with self._lock:
            memory = self.get_memory(memory_id)
            self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._conn.commit()
        return memory

    # ---------- reminders ----------

    def add_reminder(self, text: str, due: datetime | None = None) -> Reminder:
        text = _clean(text, "reminder")
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO reminders (text, due, created_at) VALUES (?, ?, ?)",
                (text, due.isoformat(timespec="minutes") if due else None, self.now().isoformat()),
            )
            self._conn.commit()
        return self.get_reminder(cur.lastrowid)

    def get_reminder(self, reminder_id: int) -> Reminder:
        row = self._conn.execute("SELECT * FROM reminders WHERE id = ?", (reminder_id,)).fetchone()
        if row is None:
            raise StoreError(f"No reminder with ID {reminder_id}.")
        return self._row_to_reminder(row)

    def list_reminders(self, include_done: bool = False) -> list[Reminder]:
        sql = "SELECT * FROM reminders" + ("" if include_done else " WHERE done = 0")
        # Reminders with a due date first (soonest first), then undated ones.
        sql += " ORDER BY due IS NULL, due, id"
        return [self._row_to_reminder(r) for r in self._conn.execute(sql).fetchall()]

    def due_reminders(self) -> list[Reminder]:
        """Open reminders that are due by the end of today (including overdue)."""
        end_of_today = self.now().replace(hour=23, minute=59, second=59, microsecond=0)
        return [r for r in self.list_reminders() if r.due is not None and r.due <= end_of_today]

    def complete_reminder(self, reminder_id: int) -> Reminder:
        with self._lock:
            reminder = self.get_reminder(reminder_id)
            if reminder.done:
                raise StoreError(f"Reminder {reminder_id} is already done.")
            self._conn.execute("UPDATE reminders SET done = 1 WHERE id = ?", (reminder_id,))
            self._conn.commit()
        reminder.done = True
        return reminder

    # ---------- notes ----------

    def add_note(self, title: str, content: str) -> Note:
        title = _clean(title, "note title")[:100]
        content = (content or "").strip()
        if not content:
            raise StoreError("The note is empty.")
        if len(content) > 5000:
            raise StoreError("The note is too long (max 5000 characters).")
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO notes (title, content, created_at) VALUES (?, ?, ?)",
                (title, content, self.now().isoformat()),
            )
            self._conn.commit()
        row = self._conn.execute("SELECT * FROM notes WHERE id = ?", (cur.lastrowid,)).fetchone()
        return self._row_to_note(row)

    def search_notes(self, query: str = "", limit: int = 10) -> list[Note]:
        query = (query or "").strip()
        if query:
            like = f"%{query}%"
            rows = self._conn.execute(
                "SELECT * FROM notes WHERE title LIKE ? OR content LIKE ? ORDER BY id DESC LIMIT ?",
                (like, like, limit),
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM notes ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._row_to_note(r) for r in rows]

    # ---------- helpers ----------

    @staticmethod
    def _row_to_reminder(row: sqlite3.Row) -> Reminder:
        return Reminder(row["id"], row["text"], datetime.fromisoformat(row["due"]) if row["due"] else None,
                        bool(row["done"]))

    @staticmethod
    def _row_to_note(row: sqlite3.Row) -> Note:
        return Note(row["id"], row["title"], row["content"], datetime.fromisoformat(row["created_at"]))
