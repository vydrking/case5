from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pathlib import Path
import tempfile
import json
import os

from app.schemas.review import ReviewResponse
from app.services.review_service import run_autoreview_pipeline, run_autoreview_from_source


router = APIRouter(prefix="/review", tags=["review"])

# File size limits (in bytes)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_HTML_SIZE = 10 * 1024 * 1024  # 10MB for HTML files

# Allowed content types
ALLOWED_HTML_TYPES = {"text/html", "application/xhtml+xml"}
ALLOWED_ZIP_TYPES = {"application/zip", "application/x-zip-compressed"}


def validate_file(file: UploadFile, max_size: int, allowed_types: set[str], file_type: str) -> None:
    """Validate file size and content type."""
    if file.size and file.size > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"{file_type} file too large. Maximum size: {max_size // (1024*1024)}MB"
        )

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {file_type} file type: {file.content_type}. Allowed: {', '.join(allowed_types)}"
        )


@router.post("/run", response_model=ReviewResponse)
async def run_review(
    desc: UploadFile = File(..., description="HTML с описанием проекта"),
    checklist: UploadFile = File(..., description="HTML чеклиста"),
    project_zip: UploadFile = File(..., description="ZIP архива проекта"),
) -> ReviewResponse:
    try:
        # Validate file sizes and types
        validate_file(desc, MAX_HTML_SIZE, ALLOWED_HTML_TYPES, "description HTML")
        validate_file(checklist, MAX_HTML_SIZE, ALLOWED_HTML_TYPES, "checklist HTML")
        validate_file(project_zip, MAX_FILE_SIZE, ALLOWED_ZIP_TYPES, "project ZIP")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            desc_path = tmp / "desc.html"
            checklist_path = tmp / "checklist.html"
            zip_path = tmp / "project.zip"

            # Read and write files
            desc_content = await desc.read()
            checklist_content = await checklist.read()
            zip_content = await project_zip.read()

            desc_path.write_bytes(desc_content)
            checklist_path.write_bytes(checklist_content)
            zip_path.write_bytes(zip_content)

            result = run_autoreview_pipeline(
                desc_path=str(desc_path),
                checklist_path=str(checklist_path),
                zip_path=str(zip_path),
            )

            return ReviewResponse(**result)

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {str(e)}")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"File system error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/run_from_github")
def run_review_from_github(repo_url: str, branch: str = "main") -> dict:
    try:
        # Basic URL validation
        if not repo_url or not isinstance(repo_url, str):
            raise HTTPException(status_code=400, detail="Invalid repository URL")

        if not repo_url.startswith(('http://', 'https://')) or 'github.com' not in repo_url:
            raise HTTPException(status_code=400, detail="Only GitHub repository URLs are supported")

        result = run_autoreview_from_source(repo_url, branch=branch)
        return {
            'ok': True,
            'html': result.get('report_html') or '',
            'meta': result.get('project_meta') or {},
        }
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid repository data: {str(e)}")
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Unable to connect to repository: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

