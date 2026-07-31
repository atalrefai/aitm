"""Background job tracker for long-running engine runs from the web UI."""

from __future__ import annotations

import inspect
import threading
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable


class JobCancelledError(RuntimeError):
    """Raised when a background job is cancelled by the user."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    id: str
    name: str
    status: str = "queued"  # queued | running | success | error | cancelled
    created_at: str = field(default_factory=_utc)
    started_at: str | None = None
    finished_at: str | None = None
    result: Any = None
    error: str | None = None
    progress: float = 0.0  # 0..100
    message: str = ""
    cancel_requested: bool = False
    logs: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    # Structured live status for Engine4 (per-TF stages/metrics) — not log text.
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, name: str, fn: Callable[..., Any]) -> Job:
        job = Job(id=str(uuid.uuid4())[:8], name=name)
        with self._lock:
            self._jobs[job.id] = job

        def _run() -> None:
            with self._lock:
                job.status = "running"
                job.started_at = _utc()
                job.progress = 0.0
                job.message = "بدء…"
            try:
                params = inspect.signature(fn).parameters
                result = fn(job) if len(params) >= 1 else fn()
                with self._lock:
                    if job.cancel_requested:
                        job.status = "cancelled"
                        job.message = "تم إيقاف المهمة"
                        job.finished_at = _utc()
                        return
                    job.result = result
                    job.status = "success"
                    job.progress = 100.0
                    if not job.message or job.message == "بدء…":
                        job.message = "اكتمل"
                    job.finished_at = _utc()
            except JobCancelledError:
                with self._lock:
                    job.status = "cancelled"
                    job.error = None
                    job.message = "تم إيقاف المهمة"
                    job.finished_at = _utc()
            except Exception as exc:
                with self._lock:
                    job.status = "error"
                    job.error = f"{exc}\n{traceback.format_exc()}"
                    job.message = str(exc)
                    job.finished_at = _utc()

        threading.Thread(target=_run, daemon=True).start()
        return job

    def set_progress(self, job_id: str, progress: float, message: str = "") -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.progress = max(0.0, min(100.0, float(progress)))
            if message:
                job.message = message
                job.history.append(
                    {
                        "ts": _utc(),
                        "progress": job.progress,
                        "message": message,
                    }
                )
                job.history = job.history[-300:]

    def append_log(self, job_id: str, line: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            text = str(line).strip()
            if not text:
                return
            job.logs.append(f"{_utc()} {text}")
            job.logs = job.logs[-500:]

    def set_details(self, job_id: str, details: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.details = dict(details or {})

    def update_details(self, job_id: str, patch: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            cur = dict(job.details or {})
            for k, v in (patch or {}).items():
                if k == "timeframes" and isinstance(v, dict) and isinstance(cur.get("timeframes"), dict):
                    merged = dict(cur["timeframes"])
                    merged.update(v)
                    cur["timeframes"] = merged
                else:
                    cur[k] = v
            job.details = cur

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.cancel_requested = True
            if job.status == "queued":
                job.status = "cancelled"
                job.message = "تم إيقاف المهمة"
                job.finished_at = _utc()
            elif job.status == "running":
                job.message = "جارٍ إيقاف المهمة…"
            return job

    def raise_if_cancelled(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and job.cancel_requested:
                raise JobCancelledError("تم إيقاف المهمة")

    def list(self, limit: int = 50) -> list[Job]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return jobs[:limit]


jobs = JobManager()
