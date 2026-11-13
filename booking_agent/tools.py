"""Tools the agent can call. Each one returns plain text the model can read.

Business errors come back as "Error: ..." strings instead of exceptions, so the
agent can explain the problem to the patient and try again.
Actions that change the calendar require `patient_confirmed=True`, which the
system prompt only allows after the patient has said yes to a summary.
"""

from __future__ import annotations

from datetime import date, datetime

from langchain_core.tools import BaseTool, tool

from .clinic import clinic_info_text
from .store import Appointment, AppointmentStore, BookingError

MAX_SLOTS_SHOWN = 8


def _fmt(dt: datetime) -> str:
    return dt.strftime("%a %b %d, %I:%M %p").replace(" 0", " ")


def _describe(appt: Appointment) -> str:
    return (f"ID {appt.id}: {appt.service} for {appt.patient_name} on {_fmt(appt.start)} "
            f"(until {appt.end.strftime('%I:%M %p').lstrip('0')}), status {appt.status}")


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value.strip())
    except (ValueError, AttributeError):
        raise BookingError("Dates must be in YYYY-MM-DD format.")


def _parse_datetime(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value.strip())
    except (ValueError, AttributeError):
        raise BookingError("Times must be in YYYY-MM-DDTHH:MM format, e.g. 2026-11-17T10:30.")
    return dt.replace(tzinfo=None, second=0, microsecond=0)


NOT_CONFIRMED = ("Not done: the patient has not confirmed yet. Summarize the details, ask the patient "
                 "to confirm, and call again with patient_confirmed=true only after they say yes.")


def make_tools(store: AppointmentStore) -> list[BaseTool]:
    @tool
    def get_clinic_info() -> str:
        """Get the clinic's address, phone, opening hours, services with durations and prices, accepted insurance and policies."""
        return clinic_info_text()

    @tool
    def check_availability(day: str, service: str) -> str:
        """List open appointment start times for a service on one day.

        Args:
            day: The date in YYYY-MM-DD format.
            service: The service name, e.g. "cleaning", "checkup", "filling", "whitening", "consultation".
        """
        try:
            slots = store.available_slots(_parse_date(day), service)
        except BookingError as e:
            return f"Error: {e}"
        if not slots:
            return f"No open times for {service} on {day}. The clinic may be closed or fully booked; try another day."
        shown = slots[:MAX_SLOTS_SHOWN]
        lines = [f"- {_fmt(s)} (start_time={s.isoformat(timespec='minutes')})" for s in shown]
        more = f"\n...and {len(slots) - len(shown)} more later that day." if len(slots) > len(shown) else ""
        return f"Open times for {service} on {day}:\n" + "\n".join(lines) + more

    @tool
    def book_appointment(patient_name: str, phone: str, service: str, start_time: str,
                         patient_confirmed: bool = False) -> str:
        """Book a new appointment. Only call with patient_confirmed=true after the patient said yes to a summary.

        Args:
            patient_name: The patient's full name.
            phone: The patient's 10-digit phone number.
            service: The service name.
            start_time: Start time from check_availability, in YYYY-MM-DDTHH:MM format.
            patient_confirmed: True only if the patient explicitly confirmed these exact details.
        """
        if not patient_confirmed:
            return NOT_CONFIRMED
        try:
            appt = store.book(patient_name, phone, service, _parse_datetime(start_time))
        except BookingError as e:
            return f"Error: {e}"
        return "Booked. " + _describe(appt)

    @tool
    def find_appointments(phone: str) -> str:
        """Find a patient's upcoming appointments by their phone number.

        Args:
            phone: The patient's 10-digit phone number.
        """
        try:
            appts = store.find_upcoming(phone)
        except BookingError as e:
            return f"Error: {e}"
        if not appts:
            return "No upcoming appointments found for that phone number."
        return "Upcoming appointments:\n" + "\n".join(f"- {_describe(a)}" for a in appts)

    @tool
    def reschedule_appointment(appointment_id: str, phone: str, new_start_time: str,
                               patient_confirmed: bool = False) -> str:
        """Move an existing appointment to a new time. Only call with patient_confirmed=true after the patient said yes.

        Args:
            appointment_id: The appointment ID from find_appointments.
            phone: The phone number on the appointment.
            new_start_time: New start time from check_availability, in YYYY-MM-DDTHH:MM format.
            patient_confirmed: True only if the patient explicitly confirmed the new time.
        """
        if not patient_confirmed:
            return NOT_CONFIRMED
        try:
            appt = store.reschedule(appointment_id, phone, _parse_datetime(new_start_time))
        except BookingError as e:
            return f"Error: {e}"
        return "Rescheduled. " + _describe(appt)

    @tool
    def cancel_appointment(appointment_id: str, phone: str, patient_confirmed: bool = False) -> str:
        """Cancel an existing appointment. Only call with patient_confirmed=true after the patient said yes.

        Args:
            appointment_id: The appointment ID from find_appointments.
            phone: The phone number on the appointment.
            patient_confirmed: True only if the patient explicitly confirmed the cancellation.
        """
        if not patient_confirmed:
            return NOT_CONFIRMED
        try:
            appt = store.cancel(appointment_id, phone)
        except BookingError as e:
            return f"Error: {e}"
        hours_left = (appt.start - store.now()).total_seconds() / 3600
        note = " Note: this was within 24 hours, so a $25 late-cancellation fee may apply." if hours_left < 24 else ""
        return "Cancelled. " + _describe(appt) + note

    return [get_clinic_info, check_availability, book_appointment, find_appointments,
            reschedule_appointment, cancel_appointment]
