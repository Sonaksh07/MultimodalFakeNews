"""
Single source of truth for timestamps across the entire project.

Every timestamp shown to the user -- database records, PDF reports,
dashboard displays, audit history -- must come from `now_ist()` /
`now_ist_str()` so they're guaranteed consistent, rather than mixing
`datetime.now()` (server-local time, which may not be IST depending on the
host) and `datetime.utcnow()` (UTC) in different places.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def now_ist() -> datetime:
    """Returns the current time as a timezone-aware datetime in IST."""
    return datetime.now(IST)


def now_ist_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Returns the current IST time formatted as a string."""
    return now_ist().strftime(fmt)


def now_ist_naive() -> datetime:
    """Returns the current IST time as a naive datetime (no tzinfo), for
    storage in database columns that don't preserve timezone info."""
    return now_ist().replace(tzinfo=None)
