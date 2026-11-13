"""Static information about the demo clinic the agent works for.

Everything the agent says about the business (hours, services, prices, policies)
comes from here through the `get_clinic_info` tool, so the model never has to
guess facts about the clinic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time


@dataclass(frozen=True)
class Service:
    key: str
    name: str
    duration_min: int
    price_usd: int
    description: str
    aliases: tuple[str, ...] = ()


SERVICES: dict[str, Service] = {
    s.key: s
    for s in [
        Service("cleaning", "Teeth cleaning", 60, 120,
                "Routine cleaning and polish by a hygienist.",
                ("clean", "cleaning", "teeth cleaning", "hygiene")),
        Service("checkup", "Checkup and exam", 30, 80,
                "Dental exam with the dentist, including a quick screening.",
                ("checkup", "check-up", "exam", "examination")),
        Service("filling", "Filling", 60, 180,
                "Tooth-colored filling for a cavity.",
                ("filling", "cavity")),
        Service("whitening", "Teeth whitening", 90, 350,
                "In-office whitening session.",
                ("whitening", "whiten", "bleaching")),
        Service("consultation", "New-patient consultation", 30, 0,
                "Free first visit to discuss your needs and plan treatment.",
                ("consultation", "consult", "new patient", "first visit")),
    ]
}

CLINIC_NAME = "Bright Smile Dental"
CLINIC_ADDRESS = "123 Main Street, San Jose, CA 95112"
CLINIC_PHONE = "(555) 010-2030"

# weekday() -> (open, close). Missing weekday = closed.
OPENING_HOURS: dict[int, tuple[time, time]] = {
    0: (time(9), time(17)),  # Monday
    1: (time(9), time(17)),
    2: (time(9), time(17)),
    3: (time(9), time(17)),
    4: (time(9), time(17)),  # Friday
    5: (time(9), time(13)),  # Saturday
}

INSURANCE = ["Delta Dental", "Aetna", "Cigna", "MetLife", "Guardian"]

POLICIES = [
    "Please arrive 10 minutes early for your appointment.",
    "Cancel or reschedule at least 24 hours ahead to avoid a $25 late-cancellation fee.",
    "Payment is due at the time of service. We accept cards, HSA and FSA.",
]

_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def get_service(name: str) -> Service | None:
    """Find a service by key, display name or common alias (case-insensitive)."""
    needle = name.strip().lower()
    if not needle:
        return None
    for service in SERVICES.values():
        if needle in (service.key, service.name.lower(), *service.aliases):
            return service
    # Loose match: "a cleaning please" -> cleaning
    for service in SERVICES.values():
        if any(alias in needle for alias in service.aliases):
            return service
    return None


def opening_hours_for(day: date) -> tuple[time, time] | None:
    return OPENING_HOURS.get(day.weekday())


def clinic_info_text() -> str:
    lines = [f"{CLINIC_NAME}", f"Address: {CLINIC_ADDRESS}", f"Phone: {CLINIC_PHONE}", "", "Hours:"]
    for i, day_name in enumerate(_DAY_NAMES):
        hours = OPENING_HOURS.get(i)
        if hours:
            lines.append(f"- {day_name}: {_fmt_time(hours[0])} - {_fmt_time(hours[1])}")
        else:
            lines.append(f"- {day_name}: closed")
    lines += ["", "Services (duration, price):"]
    for s in SERVICES.values():
        price = "free" if s.price_usd == 0 else f"${s.price_usd}"
        lines.append(f"- {s.name}: {s.duration_min} min, {price}. {s.description}")
    lines += ["", "Insurance accepted: " + ", ".join(INSURANCE), "", "Policies:"]
    lines += [f"- {p}" for p in POLICIES]
    return "\n".join(lines)


def _fmt_time(t: time) -> str:
    return t.strftime("%I:%M %p").lstrip("0")
