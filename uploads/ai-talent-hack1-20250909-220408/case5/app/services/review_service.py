from pathlib import Path
import json
from typing import Any, Dict, Optional

from case5.autoreview.parsers import read_html, parse_project_description, parse_checklist, extract_zip, project_overview
from case5.autoreview.analyzer import collect_text_samples, naive_quality_checks
from case5.autoreview.yandex_client import YandexGPTClient
from case5.autoreview.graph import build_graph
from case5.autoreview.github_reader import materialize_repo_into_dir
from case5.autoreview.cli import render_markdown
import logging


def run_autoreview_pipeline(desc_path: str, checklist_path: str, zip_path: str) -> Dict[str, Any]:
    workdir = Path.cwd() / "workdir"
    project_dir = workdir / "project"
    workdir.mkdir(parents=True, exist_ok=True)
    project_dir.mkdir(parents=True, exist_ok=True)

    desc_html = read_html(desc_path)
    check_html = read_html(checklist_path)

    meta = parse_project_description(desc_html)
    check = parse_checklist(check_html)

    extract_zip(zip_path, str(project_dir))

    overview = project_overview(str(project_dir))
    samples = collect_text_samples(str(project_dir))
    issues = naive_quality_checks(str(project_dir))

    state = {
        'project_meta': meta,
        'checklist': check,
        'project_overview': overview,
        'project_samples': samples,
        'issues': issues,
    }

    client = YandexGPTClient()
    graph = build_graph(client)
    result = graph.invoke(state)

    # объединяем state и result для полноты
    return {**state, **result}
def run_autoreview_from_source(source: str, branch: str = "main") -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    workdir = Path.cwd() / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    if 'github.com' in source:
        logger.info("review_service: materialize GitHub source=%s branch=%s", source, branch)
        root_path = materialize_repo_into_dir(source, branch=branch, workdir=str(workdir))
    else:
        root_path = source
    project_dir = Path(root_path)
    logger.info("review_service: project_dir=%s", project_dir)

    desc_text = (_read_first_readme_text(project_dir) if 'github.com' in source else '') or ''
    meta = parse_project_description(desc_text)
    check = {'title': 'Heuristic checklist', 'items': [
        'README присутствует',
        'Определены зависимости (requirements.txt/pyproject.toml)',
        'Наличие тестов (tests/)',
    ]}

    overview = project_overview(str(project_dir))
    logger.info("review_service: files_count=%d", len(overview.get('files') or []))
    samples = collect_text_samples(str(project_dir))
    issues = naive_quality_checks(str(project_dir))

    state = {
        'project_meta': meta,
        'checklist': check,
        'project_overview': overview,
        'project_samples': samples,
        'issues': issues,
    }

    client = YandexGPTClient()
    graph = build_graph(client)
    logger.info("review_service: graph compiled, invoking...")
    result = graph.invoke(state)
    logger.info("review_service: graph finished")
    combined = {**state, **result}
    # markdown/html
    md = render_markdown(combined)
    try:
        import markdown  # type: ignore
        html = markdown.markdown(md)
    except Exception:
        html = f"<pre>{md}</pre>"
    combined['report_markdown'] = md
    combined['report_html'] = html
    return combined

def _read_first_readme_text(root: Path) -> Optional[str]:
    for name in ["README.md", "README.MD", "Readme.md", "readme.md", "README.txt", "README.rst"]:
        p = root / name
        if p.exists():
            try:
                return p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
    return None



