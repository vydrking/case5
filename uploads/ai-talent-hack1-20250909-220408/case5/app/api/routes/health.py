from fastapi import APIRouter, HTTPException
import psutil
import os
from datetime import datetime
from app.core.config import settings


router = APIRouter(tags=["health"])


def check_system_resources() -> dict:
    """Check basic system resources."""
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        return {
            "memory_percent": memory.percent,
            "disk_percent": disk.percent,
            "cpu_count": psutil.cpu_count(),
        }
    except Exception as e:
        return {"error": f"Failed to check system resources: {str(e)}"}


def check_yandex_credentials() -> dict:
    """Check if Yandex credentials are configured."""
    try:
        # Validate credentials without exposing sensitive data
        settings.validate_yandex_credentials()
        return {"yandex_configured": True}
    except ValueError:
        return {"yandex_configured": False}
    except Exception as e:
        return {"yandex_configured": False, "error": str(e)}


@router.get("/health")
def health() -> dict:
    """Basic health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "AutoReview Service"
    }


@router.get("/health/detailed")
def detailed_health() -> dict:
    """Comprehensive health check with system information."""
    system_info = check_system_resources()
    yandex_info = check_yandex_credentials()

    # Check if any critical issues exist
    critical_issues = []
    if system_info.get("memory_percent", 0) > 90:
        critical_issues.append("High memory usage")
    if system_info.get("disk_percent", 0) > 95:
        critical_issues.append("Low disk space")

    return {
        "status": "ok" if not critical_issues else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "AutoReview Service",
        "version": "0.1.0",
        "system": system_info,
        "yandex": yandex_info,
        "critical_issues": critical_issues,
    }


