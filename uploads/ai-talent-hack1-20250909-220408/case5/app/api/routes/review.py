from fastapi import APIRouter, UploadFile, File, Form
from fastapi import HTTPException
from pathlib import Path
import tempfile
import shutil
import json

from app.schemas.review import ReviewResponse
from app.services.review_service import run_autoreview_pipeline, run_autoreview_from_source


router = APIRouter(prefix="/review", tags=["review"])


@router.post("/run", response_model=ReviewResponse)
async def run_review(
    desc: UploadFile = File(..., description="HTML с описанием проекта"),
    checklist: UploadFile = File(..., description="HTML чеклиста"),
    project_zip: UploadFile = File(..., description="ZIP архива проекта"),
) -> ReviewResponse:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            desc_path = tmp / "desc.html"
            checklist_path = tmp / "checklist.html"
            zip_path = tmp / "project.zip"

            desc_path.write_bytes(await desc.read())
            checklist_path.write_bytes(await checklist.read())
            zip_path.write_bytes(await project_zip.read())

            result = run_autoreview_pipeline(
                desc_path=str(desc_path),
                checklist_path=str(checklist_path),
                zip_path=str(zip_path),
            )

            return ReviewResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run_from_github")
def run_review_from_github(repo_url: str, branch: str = "main") -> dict:
    try:
        result = run_autoreview_from_source(repo_url, branch=branch)
        return {
            'ok': True,
            'html': result.get('report_html') or '',
            'meta': result.get('project_meta') or {},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

