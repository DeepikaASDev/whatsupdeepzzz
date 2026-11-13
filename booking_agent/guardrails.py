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


def check_output(text: str) -> tuple[str, str | None]:
    """Return (safe_text, reason_if_replaced)."""
    if PROMPT_CANARY in text:
        return LEAK_REPLY, "prompt_leak"
    return text, None
