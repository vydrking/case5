from pathlib import Path
import json
import tempfile
import shutil
from typing import Any, Dict, Optional
from contextlib import contextmanager

from case5.autoreview.parsers import read_html, parse_project_description, parse_checklist, extract_zip, project_overview
from case5.autoreview.analyzer import collect_text_samples, naive_quality_checks
from case5.autoreview.yandex_client import YandexGPTClient
from case5.autoreview.graph import build_graph
from case5.autoreview.github_reader import materialize_repo_into_dir
from case5.autoreview.cli import render_markdown
import logging


@contextmanager
def temporary_work_directory():
    """Context manager for creating and cleaning up temporary work directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        project_dir = workdir / "project"
        project_dir.mkdir(parents=True, exist_ok=True)
        try:
            yield workdir, project_dir
        finally:
            # Cleanup is handled automatically by TemporaryDirectory
            pass


def run_autoreview_pipeline(desc_path: str, checklist_path: str, zip_path: str) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)

    try:
        # Validate input files exist
        desc_file = Path(desc_path)
        checklist_file = Path(checklist_path)
        zip_file = Path(zip_path)

        if not desc_file.exists():
            raise FileNotFoundError(f"Description file not found: {desc_path}")
        if not checklist_file.exists():
            raise FileNotFoundError(f"Checklist file not found: {checklist_path}")
        if not zip_file.exists():
            raise FileNotFoundError(f"ZIP file not found: {zip_path}")

        # Read and parse HTML files
        desc_html = read_html(desc_path)
        check_html = read_html(checklist_path)

        meta = parse_project_description(desc_html)
        check = parse_checklist(check_html)

        # Use temporary directory for processing
        with temporary_work_directory() as (workdir, project_dir):
            logger.info(f"Extracting ZIP to temporary directory: {project_dir}")
            extract_zip(zip_path, str(project_dir))

            logger.info("Analyzing project structure...")
            overview = project_overview(str(project_dir))

            logger.info("Collecting text samples...")
            samples = collect_text_samples(str(project_dir))

            logger.info("Running quality checks...")
            issues = naive_quality_checks(str(project_dir))

        state = {
            'project_meta': meta,
            'checklist': check,
            'project_overview': overview,
            'project_samples': samples,
            'issues': issues,
        }

        logger.info("Initializing AI client...")
        client = YandexGPTClient()

        logger.info("Building analysis graph...")
        graph = build_graph(client)

        logger.info("Running AI-powered analysis...")
        result = graph.invoke(state)

        logger.info("Analysis complete")
        # объединяем state и result для полноты
        return {**state, **result}

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except PermissionError as e:
        logger.error(f"Permission denied: {e}")
        raise
    except OSError as e:
        logger.error(f"OS error during file processing: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in autoreview pipeline: {e}")
        raise
def run_autoreview_from_source(source: str, branch: str = "main") -> Dict[str, Any]:
    logger = logging.getLogger(__name__)

    try:
        if 'github.com' in source:
            logger.info("review_service: materialize GitHub source=%s branch=%s", source, branch)
            with temporary_work_directory() as (workdir, project_dir):
                root_path = materialize_repo_into_dir(source, branch=branch, workdir=str(workdir))
                actual_project_dir = Path(root_path)
        else:
            actual_project_dir = Path(source)
            if not actual_project_dir.exists():
                raise FileNotFoundError(f"Source directory not found: {source}")

        logger.info("review_service: project_dir=%s", actual_project_dir)

        desc_text = (_read_first_readme_text(actual_project_dir) if 'github.com' in source else '') or ''
        meta = parse_project_description(desc_text)
        check = {'title': 'Heuristic checklist', 'items': [
            'README присутствует',
            'Определены зависимости (requirements.txt/pyproject.toml)',
            'Наличие тестов (tests/)',
        ]}

        overview = project_overview(str(actual_project_dir))
        logger.info("review_service: files_count=%d", len(overview.get('files') or []))
        samples = collect_text_samples(str(actual_project_dir))
        issues = naive_quality_checks(str(actual_project_dir))

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
        except Exception as e:
            logger.warning(f"Failed to convert markdown to HTML: {e}")
            html = f"<pre>{md}</pre>"

        combined['report_markdown'] = md
        combined['report_html'] = html
        return combined

    except FileNotFoundError as e:
        logger.error(f"File or directory not found: {e}")
        raise
    except PermissionError as e:
        logger.error(f"Permission denied: {e}")
        raise
    except OSError as e:
        logger.error(f"OS error during processing: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in autoreview from source: {e}")
        raise

def _read_first_readme_text(root: Path) -> Optional[str]:
    for name in ["README.md", "README.MD", "Readme.md", "readme.md", "README.txt", "README.rst"]:
        p = root / name
        if p.exists():
            try:
                return p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
    return None



