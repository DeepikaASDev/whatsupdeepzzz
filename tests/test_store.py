from datetime import date, datetime

import pytest

from booking_agent.store import AppointmentStore, BookingError, normalize_phone

from .conftest import FIXED_NOW

MONDAY = date(2026, 11, 16)
SATURDAY = date(2026, 11, 21)
SUNDAY = date(2026, 11, 22)


def test_slots_follow_opening_hours(store):
    slots = store.available_slots(MONDAY, "cleaning")  # 60 min, 9-17
    assert slots[0] == datetime(2026, 11, 16, 9, 0)
    assert slots[-1] == datetime(2026, 11, 16, 16, 0)  # last start that ends by 17:00


def test_saturday_short_day_and_sunday_closed(store):
    assert store.available_slots(SATURDAY, "whitening")[-1] == datetime(2026, 11, 21, 11, 30)
    assert store.available_slots(SUNDAY, "checkup") == []


def test_no_slots_in_the_past_or_within_lead_time():
    store = AppointmentStore(":memory:", now_fn=lambda: datetime(2026, 11, 16, 10, 15))
    slots = store.available_slots(MONDAY, "checkup")
    assert slots[0] == datetime(2026, 11, 16, 11, 30)  # now + 1h lead, rounded to the next slot
    assert store.available_slots(date(2026, 11, 13), "checkup") == []


def test_booking_blocks_overlapping_slots(store):
    store.book("Ana Lopez", "(408) 555-0101", "cleaning", datetime(2026, 11, 16, 10, 0))
    slots = store.available_slots(MONDAY, "checkup")
    assert datetime(2026, 11, 16, 10, 0) not in slots
    assert datetime(2026, 11, 16, 10, 30) not in slots
    assert datetime(2026, 11, 16, 9, 30) in slots      # 9:30-10:00 ends exactly when the cleaning starts
    assert datetime(2026, 11, 16, 11, 0) in slots
    with pytest.raises(BookingError, match="not available"):
        store.book("Ben Wu", "4085550102", "checkup", datetime(2026, 11, 16, 10, 30))


def test_book_validates_inputs(store):
    start = datetime(2026, 11, 16, 9, 0)
    with pytest.raises(BookingError, match="phone"):
        store.book("Ana Lopez", "12345", "cleaning", start)
    with pytest.raises(BookingError, match="Unknown service"):
        store.book("Ana Lopez", "4085550101", "root canal", start)
    with pytest.raises(BookingError, match="full name"):
        store.book(" ", "4085550101", "cleaning", start)
    with pytest.raises(BookingError, match="not available"):
        store.book("Ana Lopez", "4085550101", "cleaning", datetime(2026, 11, 22, 10, 0))  # Sunday


def test_find_reschedule_cancel(store):
    appt = store.book("Ana Lopez", "408-555-0101", "cleaning", datetime(2026, 11, 17, 9, 0))
    assert [a.id for a in store.find_upcoming("+1 408 555 0101")] == [appt.id]

    moved = store.reschedule(appt.id, "4085550101", datetime(2026, 11, 17, 9, 30))  # overlaps itself: allowed
    assert moved.start == datetime(2026, 11, 17, 9, 30)
    assert moved.end == datetime(2026, 11, 17, 10, 30)

    with pytest.raises(BookingError, match="No appointment found"):
        store.cancel(appt.id, "4085559999")  # wrong phone

    store.cancel(appt.id.lower(), "4085550101")
    assert store.find_upcoming("4085550101") == []
    with pytest.raises(BookingError, match="already cancelled"):
        store.cancel(appt.id, "4085550101")


def test_normalize_phone():
    assert normalize_phone("+1 (408) 555-0101") == "4085550101"
    with pytest.raises(BookingError):
        normalize_phone("555-0101")


def test_fixed_now_is_monday_morning():
    assert FIXED_NOW.weekday() == 0
