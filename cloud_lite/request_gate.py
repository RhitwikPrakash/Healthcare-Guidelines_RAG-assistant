from __future__ import annotations

import random
import threading
import time
from typing import Callable, Optional


_GATE_LOCK = threading.Lock()
_NEXT_PROVIDER_TIME = 0.0


def reserve_request_slot(
    *,
    minimum_delay: float = 20.0,
    maximum_delay: float = 30.0,
    minimum_spacing: float = 25.0,
) -> float:
    """Reserve a provider-call time and return the number of seconds to wait.

    The random delay slows each submitted question. The shared in-process slot also
    spaces calls from multiple Streamlit sessions that use the same worker process.
    """
    if minimum_delay < 0 or maximum_delay < minimum_delay:
        raise ValueError("Invalid request-delay range")

    global _NEXT_PROVIDER_TIME

    now = time.monotonic()
    requested_delay = random.uniform(minimum_delay, maximum_delay)

    with _GATE_LOCK:
        earliest_for_spacing = _NEXT_PROVIDER_TIME
        scheduled_time = max(now + requested_delay, earliest_for_spacing)
        _NEXT_PROVIDER_TIME = scheduled_time + minimum_spacing

    return max(0.0, scheduled_time - now)


def wait_for_request_slot(
    progress_callback: Optional[Callable[[float, int, int], None]] = None,
    *,
    minimum_delay: float = 20.0,
    maximum_delay: float = 30.0,
    minimum_spacing: float = 25.0,
) -> int:
    """Wait for a reserved provider-call slot and report countdown progress.

    Returns the rounded total wait time.
    """
    wait_seconds = reserve_request_slot(
        minimum_delay=minimum_delay,
        maximum_delay=maximum_delay,
        minimum_spacing=minimum_spacing,
    )
    total_seconds = max(1, int(round(wait_seconds)))
    started = time.monotonic()

    while True:
        elapsed = time.monotonic() - started
        remaining = max(0.0, wait_seconds - elapsed)
        completed_fraction = 1.0 if wait_seconds <= 0 else min(1.0, elapsed / wait_seconds)

        if progress_callback:
            progress_callback(completed_fraction, int(round(remaining)), total_seconds)

        if remaining <= 0:
            break
        time.sleep(min(1.0, remaining))

    return total_seconds