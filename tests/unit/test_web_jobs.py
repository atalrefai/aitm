from __future__ import annotations

import time

from atis.web.jobs import jobs


def test_job_cancel_marks_running_job_cancelled() -> None:
    def long_job(job: object) -> str:
        job_id = getattr(job, "id")
        for _ in range(50):
            jobs.raise_if_cancelled(job_id)
            time.sleep(0.01)
        return "done"

    created = jobs.submit("cancel_me", long_job)

    for _ in range(50):
        current = jobs.get(created.id)
        if current and current.status == "running":
            break
        time.sleep(0.01)

    jobs.cancel(created.id)

    for _ in range(100):
        current = jobs.get(created.id)
        if current and current.status == "cancelled":
            break
        time.sleep(0.01)

    final = jobs.get(created.id)
    assert final is not None
    assert final.status == "cancelled"
    assert final.cancel_requested is True
