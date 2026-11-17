"""Deterministic guardrails that run before and after the language model.

These do not depend on the model behaving well:
- input checks block prompt-injection attempts, oversized messages, and
  route possible medical emergencies to a fixed safety message;
- output checks stop the system prompt from leaking.
The system prompt adds softer rules (stay on topic, no medical advice,
confirm before booking), and the booking tools enforce confirmation in code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .clinic import CLINIC_PHONE

# Marker placed in the system prompt; if it ever shows up in a reply, the
# model is leaking its instructions and the reply is replaced.
PROMPT_CANARY = "BSD-INTERNAL-7F3A"

EMERGENCY_PATTERN = re.compile(
    r"\b("
    r"can'?t breathe|cannot breathe|(difficulty|trouble) breathing|"
    r"(bleeding|blood)\b[^.?!]{0,30}\b(won'?t|will not|doesn'?t|does not|can'?t get it to) stop|"
    r"uncontrolled bleeding|"
    r"swelling (in|of) (my )?(throat|neck|eye)|(throat|neck) is swelling|"
    r"(broken|fractured) jaw|jaw (is )?(broken|fractured)|"
    r"knocked[- ]out tooth|tooth (got )?knocked out|"
    r"chest pain|passed out|unconscious"
    r")\b",
    re.IGNORECASE,
)

INJECTION_PATTERN = re.compile(
    r"("
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|prompts?)|"
    r"disregard (all |the |your )?(previous |prior )?(instructions|rules|system prompt)|"
    r"(reveal|show|print|repeat) (me )?(your|the) (system )?(prompt|instructions)|"
    r"you are now (a|an|in)\b|developer mode|jailbreak"
    r")",
    re.IGNORECASE,
)

EMERGENCY_REPLY = (
    "This sounds like it could be an emergency. If you have trouble breathing, heavy bleeding "
    "that won't stop, or swelling in your throat or face, call 911 or go to the nearest emergency "
    f"room now. For an urgent dental problem, call us at {CLINIC_PHONE}."
)
INJECTION_REPLY = (
    "I can only help with Bright Smile Dental: our services, hours, prices, and booking, "
    "rescheduling or cancelling appointments. How can I help with that?"
)
LEAK_REPLY = "Sorry, I can't share that. I can help you with appointments or questions about the clinic."


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    reply: str | None = None
    reason: str | None = None


def check_input(text: str, max_chars: int = 1000) -> GuardResult:
    if not text or not text.strip():
        return GuardResult(False, "Could you type your question again?", "empty")
    if len(text) > max_chars:
        return GuardResult(False, f"That message is too long. Please keep it under {max_chars} characters.", "too_long")
    if EMERGENCY_PATTERN.search(text):
        return GuardResult(False, EMERGENCY_REPLY, "emergency")
    if INJECTION_PATTERN.search(text):
        return GuardResult(False, INJECTION_REPLY, "prompt_injection")
    return GuardResult(True)


# --- Action-claim check: the reply may only say "booked" if a tool really booked. ---
# Each action: (pattern that claims it happened, tool name, prefix of that tool's success result).
_ACTION_CLAIMS = {
    "booking": (
        re.compile(r"\b(is|are|has been|have been|you'?re|you are|i'?ve|i have|all|now)\s+(now\s+|all\s+)?"
                   r"(booked|scheduled|confirmed)\b|\b(booked|scheduled) (you|your)\b|\bbooking (is )?confirmed\b",
                   re.IGNORECASE),
        ("book_appointment", "reschedule_appointment"), ("Booked.", "Rescheduled."),
    ),
    "reschedule": (
        re.compile(r"\b(is|has been|have|was|i'?ve)\s+(now\s+)?(rescheduled|moved)\b|\b(rescheduled|moved) (your|the) appointment\b",
                   re.IGNORECASE),
        ("reschedule_appointment",), ("Rescheduled.",),
    ),
    "cancellation": (
        re.compile(r"\b(is|has been|have|was|i'?ve)\s+(now\s+)?cancell?ed\b|\bcancell?ed (your|the) appointment\b",
                   re.IGNORECASE),
        ("cancel_appointment",), ("Cancelled.",),
    ),
}


def check_action_claims(text: str, tool_results: list[tuple[str, str]]) -> tuple[str, str | None]:
    """Replace a reply that claims a booking, reschedule or cancellation no tool actually performed.

    `tool_results` is every (tool name, result text) pair in the conversation so far.
    Small models sometimes say "you're booked!" after the booking tool returned an error.
    """
    for action, (pattern, tools, success_prefixes) in _ACTION_CLAIMS.items():
        if not pattern.search(text):
            continue
        succeeded = any(name in tools and content.startswith(success_prefixes) for name, content in tool_results)
        if succeeded:
            continue
        failures = [content for name, content in tool_results if name in tools]
        last = failures[-1] if failures else ""
        if last.startswith("Error:"):
            detail = " " + last.removeprefix("Error:").strip()
        elif last.startswith("Not done"):
            detail = " I need you to confirm the details first."
        else:
            detail = ""
        return (f"Sorry, I haven't completed the {action} yet.{detail} "
                "To book, I need the service, day and time, your full name and your phone number."
                if action == "booking" else
                f"Sorry, I haven't completed the {action} yet.{detail} "
                "Please share your appointment ID and phone number so I can look it up."), "unverified_action"
    return text, None


def check_output(text: str) -> tuple[str, str | None]:
    """Return (safe_text, reason_if_replaced)."""
    if PROMPT_CANARY in text:
        return LEAK_REPLY, "prompt_leak"
    return text, None
