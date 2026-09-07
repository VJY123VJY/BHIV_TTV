from fastapi import APIRouter, HTTPException
from app.schemas.response import JobResponse
from app.core.queue import job_manager
from app.core.exceptions import JobNotFoundError

router = APIRouter()

@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Retrieve execution status, current progress, and output metadata for a job."""
    try:
        job = job_manager.get_job(job_id)
        return JobResponse(
            job_id=job["job_id"],
            status=job["status"],
            progress=job["progress"],
            stage=job["stage"],
            created_at=job["created_at"],
            updated_at=job["updated_at"],
            result=job.get("result"),
            error=job.get("error")
        )
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
