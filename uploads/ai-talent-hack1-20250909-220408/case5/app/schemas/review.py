from typing import Any, List, Dict, Optional
from pydantic import BaseModel


class AutotestResult(BaseModel):
    id: Optional[str] = None
    type: Optional[str] = None
    ok: Optional[bool] = None
    explanation: Optional[str] = None
    detail: Optional[str] = None


class ReviewResponse(BaseModel):
    project_meta: Dict[str, Any]
    checklist: Dict[str, Any]
    project_overview: Dict[str, Any]
    project_samples: Dict[str, Any]
    issues: List[str]
    review: Optional[str] = None
    validation: Optional[str] = None
    requirements: Optional[str] = None
    autotests: Optional[Dict[str, Any]] = None
    autotest_results: Optional[Dict[str, Any]] = None
    rule_issues: Optional[List[Dict[str, Any]]] = None


