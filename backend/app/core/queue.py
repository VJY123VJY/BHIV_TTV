import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from app.core.exceptions import JobNotFoundError

class JobManager:
    """Manages asynchronous generation jobs and their execution states."""
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def create_job(
        self,
        job_id: str,
        request_data: Dict[str, Any],
        settings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        async with self._lock:
            job = {
                "job_id": job_id,
                "status": "queued",
                "progress": 0,
                "stage": "queued",
                "request": request_data,
                "settings": settings,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "result": None,
                "error": None
            }
            self._jobs[job_id] = job
            return job

    async def update_progress(self, job_id: str, stage: str, progress: int):
        async with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "processing"
                self._jobs[job_id]["stage"] = stage
                self._jobs[job_id]["progress"] = progress
                self._jobs[job_id]["updated_at"] = datetime.utcnow().isoformat() + "Z"

    async def complete_job(self, job_id: str, result_data: Dict[str, Any]):
        async with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "completed"
                self._jobs[job_id]["stage"] = "completed"
                self._jobs[job_id]["progress"] = 100
                self._jobs[job_id]["result"] = result_data
                self._jobs[job_id]["updated_at"] = datetime.utcnow().isoformat() + "Z"

    async def fail_job(self, job_id: str, error_message: str):
        async with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "failed"
                self._jobs[job_id]["stage"] = "failed"
                self._jobs[job_id]["error"] = error_message
                self._jobs[job_id]["updated_at"] = datetime.utcnow().isoformat() + "Z"

    def get_job(self, job_id: str) -> Dict[str, Any]:
        if job_id not in self._jobs:
            raise JobNotFoundError(f"Job '{job_id}' not found.")
        return self._jobs[job_id]

    def list_jobs(self) -> Dict[str, Dict[str, Any]]:
        return self._jobs

job_manager = JobManager()
