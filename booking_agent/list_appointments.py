"""Show what is really in the appointment database:  python -m booking_agent.list_appointments

Use it to check the agent's work: a booking only counts if it appears here.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

from .config import get_settings


def main() -> int:
    db_path = get_settings().db_path
    if not Path(db_path).is_file():
        print(f"No database yet at {db_path}. It is created when the app first runs.")
        return 0
    rows = sqlite3.connect(db_path).execute(
        "SELECT id, start, service, patient_name, phone, status FROM appointments ORDER BY start"
    ).fetchall()
    if not rows:
        print("No appointments in the database.")
        return 0
    print(f"{'ID':<10}{'Start':<18}{'Service':<26}{'Patient':<20}{'Phone':<12}Status")
    for appt_id, start, service, name, phone, status in rows:
        print(f"{appt_id:<10}{start[:16].replace('T', ' '):<18}{service:<26}{name:<20}{phone:<12}{status}")
    print(f"\n{len(rows)} appointment(s); {sum(r[5] == 'booked' for r in rows)} booked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
