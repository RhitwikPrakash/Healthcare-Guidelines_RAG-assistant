from __future__ import annotations
import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from uuid import uuid4

_BINDING_KEYS = ("request_id", "conversation_id", "user_message_id", "question_hash", "document_set_hash")

class JobManager:
    def __init__(self, max_workers: int = 2, retention_hours: int = 6, maximum_jobs: int = 500) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._request_index: dict[tuple[str, str], str] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rag-job")
        self._retention = timedelta(hours=retention_hours)
        self._maximum_jobs = maximum_jobs

    def _remove_job(self, job_id: str) -> None:
        job = self._jobs.pop(job_id, None)
        if not job:
            return
        owner_id = str(job.get("owner_id") or "")
        request_id = str(job.get("request_id") or "")
        if owner_id and request_id:
            self._request_index.pop((owner_id, request_id), None)

    def _prune(self) -> None:
        now = datetime.now(timezone.utc)
        removable: list[tuple[str, datetime]] = []
        for job_id, job in self._jobs.items():
            if job.get("status") not in {"complete", "failed"}:
                continue
            try:
                updated = datetime.fromisoformat(str(job["updated_at"]))
            except (ValueError, TypeError, KeyError):
                updated = now
            if now - updated > self._retention:
                removable.append((job_id, updated))
        for job_id, _ in removable:
            self._remove_job(job_id)

        overflow = len(self._jobs) - self._maximum_jobs
        if overflow > 0:
            completed = sorted(
                (
                    (job_id, str(job.get("updated_at", "")))
                    for job_id, job in self._jobs.items()
                    if job.get("status") in {"complete", "failed"}
                ),
                key=lambda item: item[1],
            )
            for job_id, _ in completed[:overflow]:
                self._remove_job(job_id)

    def create(
        self,
        kind: str,
        steps: list[str],
        owner_id: str | None = None,
        *,
        binding: dict[str, str | None] | None = None,
    ) -> str:
        job_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        clean_binding = {key: (binding or {}).get(key) for key in _BINDING_KEYS}
        with self._lock:
            self._prune()
            request_id = str(clean_binding.get("request_id") or "")
            if owner_id and request_id:
                indexed = self._request_index.get((owner_id, request_id))
                if indexed and indexed in self._jobs:
                    raise ValueError("This request_id is already registered")
            self._jobs[job_id] = {
                "job_id": job_id,
                "owner_id": owner_id,
                "status": "queued",
                "kind": kind,
                "progress": 0.0,
                "phase": "Queued",
                "detail": "",
                "steps": [{"name": name, "status": "pending"} for name in steps],
                "result": None,
                "error": None,
                "created_at": now,
                "updated_at": now,
                **clean_binding,
            }
            if owner_id and request_id:
                self._request_index[(owner_id, request_id)] = job_id
        return job_id

    def find_by_request(self, owner_id: str, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            job_id = self._request_index.get((owner_id, request_id))
            if not job_id:
                return None
            return self.get(job_id, owner_id=owner_id)

    def submit(self, job_id: str, fn: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any) -> None:
        def runner() -> None:
            self.mark_running(job_id)
            try:
                result = fn(job_id, *args, **kwargs)
                self.complete(job_id, result)
            except Exception as exc:  # noqa: BLE001
                self.fail(job_id, f"{type(exc).__name__}: {exc}")

        self._executor.submit(runner)

    def get(self, job_id: str, owner_id: str | None = None) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            stored_owner = job.get("owner_id")
            if owner_id is not None and stored_owner is not None and stored_owner != owner_id:
                return None
            clean_job = copy.deepcopy(job)
            clean_job.pop("owner_id", None)
            return clean_job

    def assert_binding(self, job_id: str, **expected: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise RuntimeError("The background job no longer exists")
            for key, value in expected.items():
                if key not in _BINDING_KEYS:
                    continue
                if str(job.get(key) or "") != str(value or ""):
                    raise RuntimeError(f"Background job binding mismatch for {key}")

    def mark_running(self, job_id: str) -> None:
        self._patch(job_id, status="running", phase="Starting")

    def update(
        self,
        job_id: str,
        *,
        step: int | None = None,
        phase: str | None = None,
        detail: str | None = None,
        progress: float | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if step is not None:
                for index, item in enumerate(job["steps"]):
                    if index < step:
                        item["status"] = "complete"
                    elif index == step:
                        item["status"] = "running"
                    elif item["status"] != "complete":
                        item["status"] = "pending"
            if phase is not None:
                job["phase"] = phase
            if detail is not None:
                job["detail"] = detail
            if progress is not None:
                job["progress"] = max(0.0, min(1.0, float(progress)))
            job["updated_at"] = datetime.now(timezone.utc).isoformat()

    def complete(self, job_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            bound_result = dict(result)
            for key in _BINDING_KEYS:
                bound_value = job.get(key)
                supplied_value = bound_result.get(key)
                if bound_value and supplied_value and str(bound_value) != str(supplied_value):
                    raise RuntimeError(f"Result binding mismatch for {key}")
                if bound_value:
                    bound_result[key] = bound_value
            job["status"] = "complete"
            job["progress"] = 1.0
            job["phase"] = "Complete"
            for item in job["steps"]:
                item["status"] = "complete"
            job["result"] = bound_result
            job["updated_at"] = datetime.now(timezone.utc).isoformat()

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job["status"] = "failed"
            job["phase"] = "Failed"
            job["error"] = error
            for item in job["steps"]:
                if item["status"] == "running":
                    item["status"] = "failed"
            job["updated_at"] = datetime.now(timezone.utc).isoformat()

    def _patch(self, job_id: str, **values: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(values)
            self._jobs[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

jobs = JobManager()