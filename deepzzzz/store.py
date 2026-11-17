"""Long-term storage for Deepzzzz: memories about you, reminders and notes (SQLite).

The chat history itself is stored separately by LangGraph's SQLite checkpointer.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
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
class Mood:
    id: int
    score: int  # 1 = very low, 2 = low, 3 = okay, 4 = good, 5 = great
    label: str  # a word or two, e.g. "stressed", "excited"
    note: str
    created_at: datetime


@dataclass
class DayMood:
    day: date
    average: float
    entries: list[Mood]


@dataclass
class WeekSummary:
    start: date
    end: date
    days: list[DayMood]  # only days with at least one check-in
    check_ins: int
    average: float | None
    top_labels: list[tuple[str, int]]
    best_day: DayMood | None
    hardest_day: DayMood | None
    reminders_completed: list[str] = field(default_factory=list)
    reminders_overdue: list[str] = field(default_factory=list)
    memories_added: list[str] = field(default_factory=list)
    notes_added: list[str] = field(default_factory=list)


MOOD_WORDS = {1: "very low", 2: "low", 3: "okay", 4: "good", 5: "great"}


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
            CREATE TABLE IF NOT EXISTS moods (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                score INTEGER NOT NULL CHECK (score BETWEEN 1 AND 5),
                label TEXT NOT NULL,
                note TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_moods_created ON moods(created_at);
            """
        )
        # Databases created before mood tracking have no completed_at column on reminders.
        columns = {r["name"] for r in self._conn.execute("PRAGMA table_info(reminders)")}
        if "completed_at" not in columns:
            self._conn.execute("ALTER TABLE reminders ADD COLUMN completed_at TEXT")
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
            self._conn.execute("UPDATE reminders SET done = 1, completed_at = ? WHERE id = ?",
                               (self.now().isoformat(), reminder_id))
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

    # ---------- moods ----------

    def log_mood(self, score: int, label: str, note: str = "") -> Mood:
        try:
            score = int(score)
        except (TypeError, ValueError):
            raise StoreError("Mood score must be a number from 1 to 5.")
        if not 1 <= score <= 5:
            raise StoreError("Mood score must be from 1 (very low) to 5 (great).")
        label = _clean(label, "mood label")[:40].lower()
        note = " ".join((note or "").split())[:300]
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO moods (score, label, note, created_at) VALUES (?, ?, ?, ?)",
                (score, label, note, self.now().isoformat()),
            )
            self._conn.commit()
        return self._row_to_mood(self._conn.execute("SELECT * FROM moods WHERE id = ?", (cur.lastrowid,)).fetchone())

    def moods_between(self, start: datetime, end: datetime) -> list[Mood]:
        rows = self._conn.execute(
            "SELECT * FROM moods WHERE created_at >= ? AND created_at < ? ORDER BY created_at",
            (start.isoformat(), end.isoformat()),
        ).fetchall()
        return [self._row_to_mood(r) for r in rows]

    def mood_days(self, days: int = 14) -> list[DayMood]:
        """Check-ins grouped by day for the last `days` days (including today), oldest first."""
        today = self.now().date()
        start = datetime.combine(today - timedelta(days=days - 1), datetime.min.time())
        end = datetime.combine(today + timedelta(days=1), datetime.min.time())
        by_day: dict[date, list[Mood]] = {}
        for mood in self.moods_between(start, end):
            by_day.setdefault(mood.created_at.date(), []).append(mood)
        return [DayMood(d, round(sum(m.score for m in ms) / len(ms), 1), ms) for d, ms in sorted(by_day.items())]

    def logged_mood_today(self) -> bool:
        return bool(self.mood_days(1))  # mood_days(1) covers only today

    def week_summary(self, days: int = 7) -> WeekSummary:
        now = self.now()
        start_day = now.date() - timedelta(days=days - 1)
        start = datetime.combine(start_day, datetime.min.time()).isoformat()
        mood_days = self.mood_days(days)
        moods = [m for d in mood_days for m in d.entries]

        label_counts: dict[str, int] = {}
        for m in moods:
            label_counts[m.label] = label_counts.get(m.label, 0) + 1
        top = sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]

        def texts(sql: str, *args) -> list[str]:
            return [r[0] for r in self._conn.execute(sql, args).fetchall()]

        return WeekSummary(
            start=start_day,
            end=now.date(),
            days=mood_days,
            check_ins=len(moods),
            average=round(sum(m.score for m in moods) / len(moods), 1) if moods else None,
            top_labels=top,
            best_day=max(mood_days, key=lambda d: (d.average, d.day)) if mood_days else None,
            hardest_day=min(mood_days, key=lambda d: (d.average, d.day)) if mood_days else None,
            reminders_completed=texts("SELECT text FROM reminders WHERE done = 1 AND completed_at >= ? ORDER BY completed_at", start),
            reminders_overdue=texts("SELECT text FROM reminders WHERE done = 0 AND due IS NOT NULL AND due < ? ORDER BY due",
                                    now.isoformat()),
            memories_added=texts("SELECT content FROM memories WHERE created_at >= ? ORDER BY id", start),
            notes_added=texts("SELECT title FROM notes WHERE created_at >= ? ORDER BY id", start),
        )

    # ---------- helpers ----------

    @staticmethod
    def _row_to_mood(row: sqlite3.Row) -> Mood:
        return Mood(row["id"], row["score"], row["label"], row["note"], datetime.fromisoformat(row["created_at"]))


    @staticmethod
    def _row_to_reminder(row: sqlite3.Row) -> Reminder:
        return Reminder(row["id"], row["text"], datetime.fromisoformat(row["due"]) if row["due"] else None,
                        bool(row["done"]))

    @staticmethod
    def _row_to_note(row: sqlite3.Row) -> Note:
        return Note(row["id"], row["title"], row["content"], datetime.fromisoformat(row["created_at"]))
