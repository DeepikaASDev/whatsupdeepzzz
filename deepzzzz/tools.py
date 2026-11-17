"""Tools Deepzzzz can call. Each returns plain text the model reads.

Success results start with a fixed word ("Saved memory", "Reminder added", ...)
so the output guardrail can check that what Deepzzzz *says* it did really happened.
"""

from __future__ import annotations

from datetime import datetime

from langchain_core.tools import BaseTool, tool

from .store import MOOD_WORDS, Store, StoreError, WeekSummary


def _fmt(dt: datetime) -> str:
    return dt.strftime("%a %b %d, %I:%M %p").replace(" 0", " ")


def _parse_due(value: str | None) -> datetime | None:
    if value is None or not str(value).strip():
        return None
    text = str(value).strip().replace(" ", "T", 1)
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        raise StoreError("Due times must look like 2026-11-18T09:00 (or 2026-11-18 for a whole day).")
    if len(text) == 10:  # date only -> 9 AM that day
        dt = dt.replace(hour=9)
    return dt.replace(tzinfo=None, second=0, microsecond=0)


def make_tools(store: Store) -> list[BaseTool]:
    @tool
    def remember_fact(fact: str) -> str:
        """Save a lasting fact about the user so you remember it in future chats.

        Use for things worth remembering long-term: their name, people in their life, goals,
        likes and dislikes, important dates, ongoing situations. Write it as a short sentence
        in third person, e.g. "Deepika is preparing for AI engineer interviews."

        Args:
            fact: The fact to remember, as one short sentence.
        """
        try:
            memory = store.add_memory(fact)
        except StoreError as e:
            return f"Error: {e}"
        return f"Saved memory #{memory.id}: {memory.content}"

    @tool
    def forget_fact(memory_id: int) -> str:
        """Delete a saved memory when the user asks you to forget it or it is no longer true.

        Args:
            memory_id: The memory's number, shown as #N in your memory list.
        """
        try:
            memory = store.delete_memory(int(memory_id))
        except (StoreError, ValueError) as e:
            return f"Error: {e}"
        return f"Forgot memory #{memory.id}: {memory.content}"

    @tool
    def add_reminder(text: str, due: str = "") -> str:
        """Add a reminder or to-do for the user.

        Args:
            text: What to remind them about, e.g. "Call mom".
            due: Optional due time as YYYY-MM-DDTHH:MM (or YYYY-MM-DD). Leave empty for no due date.
        """
        try:
            reminder = store.add_reminder(text, _parse_due(due))
        except StoreError as e:
            return f"Error: {e}"
        when = f" (due {_fmt(reminder.due)})" if reminder.due else ""
        return f"Reminder added #{reminder.id}: {reminder.text}{when}"

    @tool
    def list_reminders(include_done: bool = False) -> str:
        """List the user's reminders and to-dos.

        Args:
            include_done: Also show completed reminders.
        """
        reminders = store.list_reminders(include_done=include_done)
        if not reminders:
            return "No reminders."
        lines = []
        for r in reminders:
            when = f" (due {_fmt(r.due)})" if r.due else ""
            lines.append(f"- #{r.id} {r.text}{when}{' [done]' if r.done else ''}")
        return "Reminders:\n" + "\n".join(lines)

    @tool
    def complete_reminder(reminder_id: int) -> str:
        """Mark a reminder as done.

        Args:
            reminder_id: The reminder's number (#N).
        """
        try:
            reminder = store.complete_reminder(int(reminder_id))
        except (StoreError, ValueError) as e:
            return f"Error: {e}"
        return f"Completed reminder #{reminder.id}: {reminder.text}"

    @tool
    def save_note(title: str, content: str) -> str:
        """Save a longer note for the user, like a list, an idea, or interview prep.

        Args:
            title: A short title.
            content: The note text.
        """
        try:
            note = store.add_note(title, content)
        except StoreError as e:
            return f"Error: {e}"
        return f"Note saved #{note.id}: {note.title}"

    @tool
    def search_notes(query: str = "") -> str:
        """Find the user's saved notes by keyword. Leave query empty to list recent notes.

        Args:
            query: Words to look for in note titles and text.
        """
        notes = store.search_notes(query)
        if not notes:
            return "No matching notes."
        return "Notes:\n" + "\n\n".join(f"#{n.id} {n.title} ({n.created_at:%b %d})\n{n.content}" for n in notes)

    @tool
    def log_mood(score: int, label: str, note: str = "") -> str:
        """Record how the user is feeling, for their mood tracker and weekly recap.

        Call this when the user tells you how they feel (in a check-in or naturally in chat).
        Log at most once per clear change in mood; don't log every message.

        Args:
            score: 1 = very low, 2 = low, 3 = okay, 4 = good, 5 = great.
            label: One or two words for the feeling, e.g. "stressed", "excited", "tired".
            note: Optional short reason in their words, e.g. "mock interview went well".
        """
        try:
            mood = store.log_mood(score, label, note)
        except StoreError as e:
            return f"Error: {e}"
        return f"Mood logged #{mood.id}: {mood.label} ({mood.score}/5, {MOOD_WORDS[mood.score]})"

    @tool
    def get_weekly_recap(days: int = 7) -> str:
        """Get the user's mood check-ins and activity for the past days, to write their recap.

        Use when they ask how their week went, ask for a recap, or on Sundays when you offer one.
        Write the recap yourself from this data: warm, specific, short, and end with encouragement.

        Args:
            days: How many days to cover (default 7).
        """
        days = max(1, min(int(days or 7), 31))
        return format_summary(store.week_summary(days))

    return [remember_fact, forget_fact, add_reminder, list_reminders, complete_reminder, save_note, search_notes,
            log_mood, get_weekly_recap]


def format_summary(w: WeekSummary) -> str:
    lines = [f"Recap data for {w.start:%a %b %d} - {w.end:%a %b %d}:"]
    if w.check_ins:
        lines.append(f"Mood check-ins: {w.check_ins}, average {w.average}/5 ({MOOD_WORDS[round(w.average)]}).")
        lines.append("Most common feelings: " + ", ".join(f"{label} x{n}" for label, n in w.top_labels) + ".")
        lines.append(f"Best day: {w.best_day.day:%A} (avg {w.best_day.average}).")
        lines.append(f"Hardest day: {w.hardest_day.day:%A} (avg {w.hardest_day.average}).")
        lines.append("Day by day:")
        for d in w.days:
            entries = "; ".join(f"{m.label} {m.score}/5" + (f" ({m.note})" if m.note else "") for m in d.entries)
            lines.append(f"- {d.day:%a %b %d}: {entries}")
    else:
        lines.append("No mood check-ins in this period.")
    lines.append("Reminders completed: " + (", ".join(w.reminders_completed) or "none") + ".")
    if w.reminders_overdue:
        lines.append("Still overdue: " + ", ".join(w.reminders_overdue) + ".")
    if w.memories_added:
        lines.append("New things they shared: " + "; ".join(m.rstrip(".") for m in w.memories_added) + ".")
    if w.notes_added:
        lines.append("Notes saved: " + ", ".join(w.notes_added) + ".")
    return "\n".join(lines)
