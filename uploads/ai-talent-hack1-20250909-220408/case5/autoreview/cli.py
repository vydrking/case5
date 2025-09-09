import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

from .parsers import read_html, parse_project_description, parse_checklist, extract_zip, project_overview
from .analyzer import collect_text_samples, naive_quality_checks
from .yandex_client import YandexGPTClient
from .graph import build_graph


def render_markdown(result: dict) -> str:
	meta = result.get('project_meta') or {}
	check = result.get('checklist') or {}
	review = result.get('review') or ''
	validation = result.get('validation') or ''
	autotests = result.get('autotests') or {}
	autores = result.get('autotest_results') or {}
	issues = result.get('rule_issues') or []
	lines = []
	lines.append(f"# Автоматическое первичное ревью: {meta.get('title','')}")
	lines.append('')
	lines.append('## Краткое описание проекта')
	lines.append(meta.get('content','')[:1200])
	lines.append('')
	lines.append('## Чеклист (извлеченные пункты)')
	for it in check.get('items', [])[:50]:
		lines.append(f"- {it}")
	lines.append('')
	lines.append('## Отчет ревьюера')
	lines.append(review)
	lines.append('')
	lines.append('## Проблемы по правилам (агрегированные)')
	for it in issues:
		loc = ', '.join(str(x) for x in it.get('lines') or [])
		lines.append(f"- [{it.get('rule')}] {it.get('file')}:{loc} — {it.get('description')}\n  Совет: {it.get('suggestion')}")
	lines.append('')
	lines.append('## Результаты автотестов')
	for r in (autores.get('results') or [])[:200]:
		status = 'PASS' if r.get('ok') else 'FAIL'
		lines.append(f"- [{status}] {r.get('id')} — {r.get('explanation')} ({r.get('type')}) {r.get('detail')}")
	lines.append('')
	lines.append('## Валидатор: итоговая оценка')
	lines.append(validation)
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
