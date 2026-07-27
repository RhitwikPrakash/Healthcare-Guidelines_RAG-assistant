from __future__ import annotations

import logging
import threading

from app.config import get_settings
from app.services.chat_store import chat_store


logger = logging.getLogger("healthcare-rag.history-cleanup")


class HistoryCleanupService:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            try:
                result = chat_store.cleanup_expired_history()
                logger.info("Chat-retention cleanup completed: %s", result)
            except Exception:  # noqa: BLE001
                logger.exception("Initial chat-retention cleanup failed")
            self._thread = threading.Thread(
                target=self._run,
                name="chat-history-cleanup",
                daemon=True,
            )
            self._thread.start()

    def _run(self) -> None:
        interval_seconds = get_settings().chat_cleanup_interval_hours * 3600
        while not self._stop.wait(interval_seconds):
            try:
                result = chat_store.cleanup_expired_history()
                logger.info("Scheduled chat-retention cleanup completed: %s", result)
            except Exception:  # noqa: BLE001
                logger.exception("Scheduled chat-retention cleanup failed")

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2)


history_cleanup = HistoryCleanupService()