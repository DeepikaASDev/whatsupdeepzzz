"""SQLite-backed appointment calendar.

All booking rules live here (opening hours, no double-booking, no past slots),
so they hold no matter what the language model asks for.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

from .clinic import Service, get_service, opening_hours_for

SLOT_STEP = timedelta(minutes=30)
MIN_LEAD_TIME = timedelta(hours=1)  # can't book a slot starting within the next hour
MAX_DAYS_AHEAD = 60


class BookingError(Exception):
    """A booking request broke a business rule. The message is safe to show the patient."""


@dataclass
class Appointment:
    id: str
    patient_name: str
    phone: str
    service: str
    start: datetime
    end: datetime
    status: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "patient_name": self.patient_name,
            "phone": self.phone,
            "service": self.service,
            "start": self.start.isoformat(timespec="minutes"),
            "end": self.end.isoformat(timespec="minutes"),
            "status": self.status,
        }


def normalize_phone(phone: str) -> str:
    digits = re.sub(r"\D", "", phone or "")
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        raise BookingError("Please provide a valid 10-digit US phone number.")
    return digits


class AppointmentStore:
    def __init__(self, db_path: str = ":memory:", now_fn: Callable[[], datetime] | None = None):
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self.now = now_fn or datetime.now
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS appointments (
                id TEXT PRIMARY KEY,
                patient_name TEXT NOT NULL,
                phone TEXT NOT NULL,
                service TEXT NOT NULL,
                start TEXT NOT NULL,
                end TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'booked',
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_appt_start ON appointments(start)")
        self._conn.commit()

    # ---------- queries ----------

    def available_slots(self, day: date, service_name: str, exclude_id: str | None = None) -> list[datetime]:
        service = self._require_service(service_name)
        hours = opening_hours_for(day)
        if hours is None:
            return []
        now = self.now()
        if day < now.date() or day > now.date() + timedelta(days=MAX_DAYS_AHEAD):
            return []

        duration = timedelta(minutes=service.duration_min)
        open_at = datetime.combine(day, hours[0])
        close_at = datetime.combine(day, hours[1])
        busy = [(a.start, a.end) for a in self._appointments_on(day) if a.id != exclude_id]

        slots = []
        start = open_at
        while start + duration <= close_at:
            end = start + duration
            if start >= now + MIN_LEAD_TIME and not any(s < end and start < e for s, e in busy):
                slots.append(start)
            start += SLOT_STEP
        return slots

    def find_upcoming(self, phone: str) -> list[Appointment]:
        phone = normalize_phone(phone)
        rows = self._conn.execute(
            "SELECT * FROM appointments WHERE phone = ? AND status = 'booked' AND start >= ? ORDER BY start",
            (phone, self.now().isoformat()),
        ).fetchall()
        return [self._row_to_appt(r) for r in rows]

    def get(self, appointment_id: str) -> Appointment | None:
        row = self._conn.execute("SELECT * FROM appointments WHERE id = ?", (appointment_id,)).fetchone()
        return self._row_to_appt(row) if row else None

    # ---------- commands ----------

    def book(self, patient_name: str, phone: str, service_name: str, start: datetime) -> Appointment:
        name = (patient_name or "").strip()
        if len(name) < 2:
            raise BookingError("Please provide the patient's full name.")
        phone = normalize_phone(phone)
        service = self._require_service(service_name)
        with self._lock:
            if start not in self.available_slots(start.date(), service.key):
                raise BookingError("That time is not available. Please check availability and pick another slot.")
            appt = Appointment(
                id=uuid.uuid4().hex[:8].upper(),
                patient_name=name,
                phone=phone,
                service=service.name,
                start=start,
                end=start + timedelta(minutes=service.duration_min),
                status="booked",
            )
            self._conn.execute(
                "INSERT INTO appointments (id, patient_name, phone, service, start, end, status, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (appt.id, appt.patient_name, appt.phone, appt.service,
                 appt.start.isoformat(), appt.end.isoformat(), appt.status, self.now().isoformat()),
            )
            self._conn.commit()
        return appt

    def reschedule(self, appointment_id: str, phone: str, new_start: datetime) -> Appointment:
        with self._lock:
            appt = self._require_owned(appointment_id, phone)
            service = self._require_service(appt.service)
            if new_start not in self.available_slots(new_start.date(), service.key, exclude_id=appt.id):
                raise BookingError("That new time is not available. Please check availability and pick another slot.")
            appt.start = new_start
            appt.end = new_start + timedelta(minutes=service.duration_min)
            self._conn.execute(
                "UPDATE appointments SET start = ?, end = ? WHERE id = ?",
                (appt.start.isoformat(), appt.end.isoformat(), appt.id),
            )
            self._conn.commit()
        return appt

    def cancel(self, appointment_id: str, phone: str) -> Appointment:
        with self._lock:
            appt = self._require_owned(appointment_id, phone)
            appt.status = "cancelled"
            self._conn.execute("UPDATE appointments SET status = 'cancelled' WHERE id = ?", (appt.id,))
            self._conn.commit()
        return appt

    # ---------- helpers ----------

    def _require_service(self, name: str) -> Service:
        service = get_service(name)
        if service is None:
            raise BookingError(f"Unknown service '{name}'. Use one of the services listed in the clinic info.")
        return service

    def _require_owned(self, appointment_id: str, phone: str) -> Appointment:
        phone = normalize_phone(phone)
        appt = self.get((appointment_id or "").strip().upper())
        # Same error for "not found" and "wrong phone" so IDs can't be probed.
        if appt is None or appt.phone != phone:
            raise BookingError("No appointment found with that ID and phone number.")
        if appt.status != "booked":
            raise BookingError("That appointment is already cancelled.")
        if appt.start < self.now():
            raise BookingError("That appointment is in the past and can't be changed.")
        return appt

    def _appointments_on(self, day: date) -> list[Appointment]:
        rows = self._conn.execute(
            "SELECT * FROM appointments WHERE status = 'booked' AND start >= ? AND start < ?",
            (datetime.combine(day, datetime.min.time()).isoformat(),
             datetime.combine(day + timedelta(days=1), datetime.min.time()).isoformat()),
        ).fetchall()
        return [self._row_to_appt(r) for r in rows]

    @staticmethod
    def _row_to_appt(row: sqlite3.Row) -> Appointment:
        return Appointment(
            id=row["id"],
            patient_name=row["patient_name"],
            phone=row["phone"],
            service=row["service"],
            start=datetime.fromisoformat(row["start"]),
            end=datetime.fromisoformat(row["end"]),
            status=row["status"],
        )
