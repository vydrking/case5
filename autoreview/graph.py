from langgraph.graph import StateGraph, END
from typing import TypedDict
import json

from .yandex_client import YandexGPTClient
from .prompts import EXTRACTOR_SYSTEM_PROMPT, REVIEWER_SYSTEM_PROMPT, VALIDATOR_SYSTEM_PROMPT, AUTOTEST_EXTRACTOR_PROMPT, RULE_CHECKER_PROMPT, REVIEW_FILTER_PROMPT
from .analyzer import run_autotests
from .indexer import CodeIndex
from .aggregator import dedupe_and_group


class ReviewState(TypedDict, total=False):
	project_meta: dict
	checklist: dict
	requirements: str
	autotests: dict
	autotest_results: dict
	project_overview: dict
	project_samples: dict
	issues: list[str]
	review: str
	validation: str
	index_ready: bool
	per_rule_context: list[dict]
	rule_issues_raw: list[dict]
	rule_issues_filtered: list[dict]
	rule_issues: list[dict]


def build_graph(client: YandexGPTClient):
	g = StateGraph(ReviewState)

	def extract_requirements(state: ReviewState):
		meta = state.get('project_meta') or {}
		check = state.get('checklist') or {}
		prompt = (
			f"{EXTRACTOR_SYSTEM_PROMPT}\n\n"
			f"Описание проекта:\n{meta.get('title', '')}\n\n{meta.get('content', '')}\n\n"
			f"Чеклист (пункты):\n- " + '\n- '.join(check.get('items', []))
		)
		text = client.complete(prompt)
		return {'requirements': text}

	def extract_autotests(state: ReviewState):
		meta = state.get('project_meta') or {}
		check = state.get('checklist') or {}
		prompt = (
			f"{AUTOTEST_EXTRACTOR_PROMPT}\n\n"
			f"Описание проекта:\n{meta.get('title', '')}\n\n{meta.get('content', '')}\n\n"
			f"Чеклист (пункты):\n- " + '\n- '.join(check.get('items', []))
		)
		text = client.complete(prompt)
		try:
			obj = json.loads(text)
		except Exception:
			obj = {'tests': []}
		return {'autotests': obj}

	def run_tests(state: ReviewState):
		root = (state.get('project_overview') or {}).get('root')
		if not root:
			return {'autotest_results': {'results': []}}
		return {'autotest_results': run_autotests(root, state.get('autotests') or {'tests': []})}

	def build_index(state: ReviewState):
		root = (state.get('project_overview') or {}).get('root')
		if not root:
			return {'index_ready': False}
		idx = CodeIndex(root)
		idx.build()
		state['_index'] = idx  # memoize in state (not serialized)
		return {'index_ready': True}

	def prepare_per_rule_context(state: ReviewState):
		idx: CodeIndex = state.get('_index')  # type: ignore
		check = state.get('checklist') or {}
		contexts = []
		for item in check.get('items', [])[:50]:
			query = item
			chunks = idx.retrieve(query, top_k=8) if idx else []
			contexts.append({'rule': item, 'chunks': chunks})
		return {'per_rule_context': contexts}

	def rule_checkers(state: ReviewState):
		contexts = state.get('per_rule_context') or []
		findings: list[dict] = []
		for ctx in contexts:
			rule = ctx.get('rule')
			chunks = ctx.get('chunks') or []
			payload = []
			for c in chunks:
				payload.append(f"FILE: {c.get('file')}\nLINES: {c.get('lines')}\n{c.get('text')}")
			prompt = (
				f"{RULE_CHECKER_PROMPT}\n\n"
				f"Правило чек-листа: {rule}\n\n"
				f"Контекст кода:\n\n" + '\n\n---\n\n'.join(payload)
			)
			text = client.complete(prompt)
			try:
				arr = json.loads(text)
			except Exception:
				arr = []
			for it in arr:
				if it:
					findings.append(it)
		return {'rule_issues_raw': findings}

	def review_filter(state: ReviewState):
		raw = state.get('rule_issues_raw') or []
		kept: list[dict] = []
		for it in raw:
			prompt = (
				f"{REVIEW_FILTER_PROMPT}\n\n"
				f"Замечание: {json.dumps(it, ensure_ascii=False)}"
			)
			text = client.complete(prompt)
			try:
				obj = json.loads(text)
			except Exception:
				obj = {'keep': True, 'reason': 'offline'}
			if obj.get('keep'):
				kept.append(it)
		return {'rule_issues_filtered': kept}

	def aggregate(state: ReviewState):
		return {'rule_issues': dedupe_and_group(state.get('rule_issues_filtered') or [])}

	def review_project(state: ReviewState):
		reqs = state.get('requirements', '')
		overview = state.get('project_overview', {})
		samples = state.get('project_samples', {})
		issues = state.get('issues', [])
		prompt = (
			f"{REVIEWER_SYSTEM_PROMPT}\n\n"
			f"Требования и критерии:\n{reqs}\n\n"
			f"Структура проекта:\n{overview}\n\n"
			f"Фрагменты файлов (усечено):\n" + '\n\n'.join(
				f"### {name}\n{content[:2000]}" for name, content in list(samples.items())[:20]
			)
			+ ("\n\nПредварительно найденные проблемы:\n- " + '\n- '.join(issues) if issues else '')
		)
		text = client.complete(prompt)
		return {'review': text}

	def validate(state: ReviewState):
		reqs = state.get('requirements', '')
		review = state.get('review', '')
		results = state.get('autotest_results', {})
		prompt = (
			f"{VALIDATOR_SYSTEM_PROMPT}\n\n"
			f"Требования:\n{reqs}\n\n"
			f"Отчет ревьюера:\n{review}\n\n"
			f"Результаты автотестов (JSON):\n{json.dumps(results, ensure_ascii=False)}"
		)
		text = client.complete(prompt)
		return {'validation': text}

	g.add_node('extract_requirements', extract_requirements)
	g.add_node('extract_autotests', extract_autotests)
	g.add_node('run_tests', run_tests)
	g.add_node('build_index', build_index)
	g.add_node('prepare_per_rule_context', prepare_per_rule_context)
	g.add_node('rule_checkers', rule_checkers)
	g.add_node('review_filter', review_filter)
	g.add_node('aggregate', aggregate)
	g.add_node('review_project', review_project)
	g.add_node('validate', validate)

	g.set_entry_point('extract_requirements')
	g.add_edge('extract_requirements', 'extract_autotests')
	g.add_edge('extract_autotests', 'run_tests')
	g.add_edge('run_tests', 'build_index')
	g.add_edge('build_index', 'prepare_per_rule_context')
	g.add_edge('prepare_per_rule_context', 'rule_checkers')
	g.add_edge('rule_checkers', 'review_filter')
	g.add_edge('review_filter', 'aggregate')
	g.add_edge('aggregate', 'review_project')
	g.add_edge('review_project', 'validate')
	g.add_edge('validate', END)

	return g.compile()
