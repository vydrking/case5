from langgraph.graph import StateGraph, END
from typing import TypedDict
import json

from .yandex_client import YandexGPTClient
from .prompts import EXTRACTOR_SYSTEM_PROMPT, REVIEWER_SYSTEM_PROMPT, VALIDATOR_SYSTEM_PROMPT, AUTOTEST_EXTRACTOR_PROMPT
from .analyzer import run_autotests


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
	g.add_node('review_project', review_project)
	g.add_node('validate', validate)

	g.set_entry_point('extract_requirements')
	g.add_edge('extract_requirements', 'extract_autotests')
	g.add_edge('extract_autotests', 'run_tests')
	g.add_edge('run_tests', 'review_project')
	g.add_edge('review_project', 'validate')
	g.add_edge('validate', END)

	return g.compile()
