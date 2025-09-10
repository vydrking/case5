import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

from .parsers import read_html, parse_project_description, parse_checklist, extract_zip, project_overview
from .analyzer import collect_text_samples, naive_quality_checks
from .yandex_client import YandexGPTClient
from .graph import build_graph


def render_project_header(meta: dict) -> list[str]:
	"""Render the project header section."""
	lines = []
	title = meta.get('title', '')
	lines.append(f"# Автоматическое первичное ревью: {title}")
	lines.append('')
	return lines


def render_project_description(meta: dict) -> list[str]:
	"""Render the project description section."""
	lines = []
	lines.append('## Краткое описание проекта')
	content = meta.get('content', '')[:1200]
	lines.append(content)
	lines.append('')
	return lines


def render_checklist(check: dict) -> list[str]:
	"""Render the checklist section."""
	lines = []
	lines.append('## Чеклист (извлеченные пункты)')
	for item in check.get('items', [])[:50]:
		lines.append(f"- {item}")
	lines.append('')
	return lines


def render_reviewer_report(review: str) -> list[str]:
	"""Render the reviewer report section."""
	lines = []
	lines.append('## Отчет ревьюера')
	lines.append(review)
	lines.append('')
	return lines


def render_issues(issues: list) -> list[str]:
	"""Render the issues section."""
	lines = []
	lines.append('## Проблемы по правилам (агрегированные)')
	for issue in issues:
		location = ', '.join(str(x) for x in issue.get('lines') or [])
		rule = issue.get('rule', '')
		file_path = issue.get('file', '')
		description = issue.get('description', '')
		suggestion = issue.get('suggestion', '')
		lines.append(f"- [{rule}] {file_path}:{location} — {description}\n  Совет: {suggestion}")
	lines.append('')
	return lines


def render_autotest_results(autores: dict) -> list[str]:
	"""Render the autotest results section."""
	lines = []
	lines.append('## Результаты автотестов')
	for result in (autores.get('results') or [])[:200]:
		status = 'PASS' if result.get('ok') else 'FAIL'
		test_id = result.get('id', '')
		explanation = result.get('explanation', '')
		test_type = result.get('type', '')
		detail = result.get('detail', '')
		lines.append(f"- [{status}] {test_id} — {explanation} ({test_type}) {detail}")
	lines.append('')
	return lines


def render_validation(validation: str) -> list[str]:
	"""Render the validation section."""
	lines = []
	lines.append('## Валидатор: итоговая оценка')
	lines.append(validation)
	return lines


def render_markdown(result: dict) -> str:
	"""Render the complete markdown report from result data."""
	meta = result.get('project_meta') or {}
	check = result.get('checklist') or {}
	review = result.get('review') or ''
	validation = result.get('validation') or ''
	autores = result.get('autotest_results') or {}
	issues = result.get('rule_issues') or []

	lines = []
	lines.extend(render_project_header(meta))
	lines.extend(render_project_description(meta))
	lines.extend(render_checklist(check))
	lines.extend(render_reviewer_report(review))
	lines.extend(render_issues(issues))
	lines.extend(render_autotest_results(autores))
	lines.extend(render_validation(validation))

	return '\n'.join(lines)


def main():
	parser = argparse.ArgumentParser(prog='autoreview')
	parser.add_argument('--desc', required=True, help='path to project description HTML')
	parser.add_argument('--checklist', required=True, help='path to checklist HTML')
	parser.add_argument('--zip', dest='zip_path', required=True, help='path to project zip')
	parser.add_argument('--workdir', default=str(Path('.').resolve() / 'workdir'), help='where to extract and work')
	parser.add_argument('--out', default='review.json', help='output review json path')
	parser.add_argument('--issues', dest='issues_out', default=None, help='optional issues JSON path')
	parser.add_argument('--md', dest='md_out', default=None, help='optional markdown report path')
	args = parser.parse_args()

	workdir = Path(args.workdir)
	workdir.mkdir(parents=True, exist_ok=True)

	desc_html = read_html(args.desc)
	check_html = read_html(args.checklist)

	meta = parse_project_description(desc_html)
	check = parse_checklist(check_html)

	extract_dir = Path(workdir) / 'project'
	extract_dir.mkdir(parents=True, exist_ok=True)
	extract_zip(args.zip_path, str(extract_dir))

	overview = project_overview(str(extract_dir))
	samples = collect_text_samples(str(extract_dir))
	issues = naive_quality_checks(str(extract_dir))

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

	# persist json
	Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
	print('Saved ->', args.out)

	# optional issues
	if args.issues_out:
		Path(args.issues_out).write_text(json.dumps(result.get('rule_issues') or [], ensure_ascii=False, indent=2), encoding='utf-8')
		print('Saved ->', args.issues_out)

	# optional markdown
	if args.md_out:
		md = render_markdown({**state, **result})
		Path(args.md_out).write_text(md, encoding='utf-8')
		print('Saved ->', args.md_out)
