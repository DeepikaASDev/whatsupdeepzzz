"""Deterministic checks that run before and after the language model.

- Input: message length; possible self-harm crisis -> the model is told to respond with care
  and 988 resources are guaranteed in the reply.
- Output: a canary token detects prompt leaks; the action-claim check makes sure Deepzzzz
  only says it saved/added/completed something when a tool really did it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PROMPT_CANARY = "DPZ-INTERNAL-91C4"

CRISIS_PATTERN = re.compile(
    r"\b("
    r"kill(ing)? myself|end(ing)? (my|it all)( life)?|take my (own )?life|"
    r"(want|wanna|going) to die|suicid\w*|"
    r"self[- ]harm\w*|hurt(ing)? myself|cut(ting)? myself|"
    r"(don'?t|do not) want to (live|be alive|exist)|no reason to live|better off dead"
    r")\b",
    re.IGNORECASE,
)

CRISIS_NOTE = (
    "The user may be in crisis or thinking about self-harm. Respond with warmth and care: acknowledge what "
    "they said, don't lecture, ask how they're doing right now, and encourage reaching out to someone they "
    "trust. Include that they can call or text 988 (US Suicide & Crisis Lifeline) any time, or call 911 if "
    "they're in immediate danger."
)
CRISIS_RESOURCES = (
    "\n\nIf you're thinking about hurting yourself, please reach out right now: call or text 988 "
    "(Suicide & Crisis Lifeline, US) any time, or call 911 if you're in danger. You don't have to go through it alone."
)

LEAK_REPLY = "Ha, nice try 😄 That's my secret recipe. So, what's going on with you today?"


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    reply: str | None = None
    reason: str | None = None
    crisis: bool = False


def check_input(text: str, max_chars: int = 4000) -> GuardResult:
    if not text or not text.strip():
        return GuardResult(False, "You there? Type something and I'm all ears 👂", "empty")
    if len(text) > max_chars:
        return GuardResult(False, f"Whoa, that's a lot! Can you keep it under {max_chars} characters?", "too_long")
    if CRISIS_PATTERN.search(text):
        return GuardResult(True, reason="crisis_support", crisis=True)
    return GuardResult(True)


def ensure_crisis_resources(reply: str) -> str:
    return reply if "988" in reply else reply.rstrip() + CRISIS_RESOURCES


def check_output(text: str) -> tuple[str, str | None]:
    if PROMPT_CANARY in text:
        return LEAK_REPLY, "prompt_leak"
    return text, None


# --- Action-claim check ---------------------------------------------------------------
# (what was claimed, pattern for the claim, tool that must have succeeded, success prefix)
_ACTION_CLAIMS = [
    ("saved that memory",
     re.compile(r"\b(i'?ll|i will) (always )?remember (that|this)\b|\b(i'?ve|i have) (saved|noted|remembered|stored)"
                r"( that| this| it)?\b|\b(saved|stored|added) (that|this|it) (to|in) (my )?memory\b|"
                r"\bgot it,? (i'?ll|i will) remember\b", re.IGNORECASE),
     "remember_fact", "Saved memory"),
    ("set that reminder",
     re.compile(r"\b(i'?ve|i have|i) (just )?(set|added|created|made) (a |the |your |that |this )?(reminder|to-?do)\b|"
                r"\breminder (is |has been )?(set|added|created|saved)\b", re.IGNORECASE),
     "add_reminder", "Reminder added"),
    ("saved that note",
     re.compile(r"\b(i'?ve|i have|i) (just )?(saved|added|written|made) (a |the |your |that |this )?note\b|"
                r"\bnote (is |has been )?saved\b", re.IGNORECASE),
     "save_note", "Note saved"),
    ("marked that done",
     re.compile(r"\b(i'?ve|i have) (marked|checked|ticked) (it|that|this|the reminder)( off| as done| done)?\b|"
                r"\bmarked (it |that |this )?(as )?(done|complete)\b", re.IGNORECASE),
     "complete_reminder", "Completed reminder"),
    ("logged your mood",
     re.compile(r"\b(i'?ve|i have|i) (just )?(logged|tracked|recorded|noted down|saved) (your |that |this |the )?"
                r"(mood|feeling|check-?in)\b|\b(mood|check-?in) (is |has been )?(logged|recorded|saved)\b", re.IGNORECASE),
     "log_mood", "Mood logged"),
    ("forgot that",
     re.compile(r"\b(i'?ve|i have) (forgotten|deleted|removed) (that|it|this)( memory)?\b", re.IGNORECASE),
     "forget_fact", "Forgot memory"),
]


def check_action_claims(text: str, turn_tool_results: list[tuple[str, str]]) -> tuple[str, str | None]:
    """Fix a reply that claims an action no tool performed in this turn."""
    for claim, pattern, tool_name, success in _ACTION_CLAIMS:
        if not pattern.search(text):
            continue
        if any(name == tool_name and content.startswith(success) for name, content in turn_tool_results):
            continue
        errors = [c for n, c in turn_tool_results if n == tool_name and c.startswith("Error:")]
        if errors:
            reason = errors[-1].removeprefix("Error:").strip()
            return f"Oops, I tried but it didn't work: {reason} Want to try again?", "unverified_action"
        return (f"Wait, I said I {claim}, but I didn't actually do it yet 😅 "
                "Want me to do it now?"), "unverified_action"
    return text, None
