import os
from pathlib import Path
import shutil
import subprocess
import logging


def _parse_github_url(url: str) -> tuple[str, str]:
	"""Return (owner, repo) from a GitHub URL.
	Supports variants like:
	- https://github.com/owner/repo
	- https://github.com/owner/repo.git
	- https://github.com/owner/repo/tree/branch
	- with query strings
	"""
	try:
		clean = url.split('github.com/', 1)[1]
		clean = clean.split('?', 1)[0]
		parts = [p for p in clean.split('/') if p]
		owner = parts[0]
		repo = parts[1]
		if repo.endswith('.git'):
			repo = repo[:-4]
		return owner, repo
	except Exception:
		raise ValueError(f"Invalid GitHub URL: {url}")


def materialize_repo_into_dir(repo_url: str, branch: str = 'main', workdir: str | None = None) -> str:
	"""Materialize repository code locally for analysis.

	Strategy:
	1) Try LlamaIndex GithubRepositoryReader with token and include-only .py (как в notebooks).
	2) If reader is unavailable/empty or token отсутствует — fallback: git clone --depth 1 -b <branch>, затем скопировать все .py в целевую папку.
	"""
	logger = logging.getLogger(__name__)
	owner, repo = _parse_github_url(repo_url)
	wd = Path(workdir or (Path.cwd() / 'workdir')).resolve()
	out = wd / f'github_{owner}_{repo}'
	out.mkdir(parents=True, exist_ok=True)

	def _clone_then_copy_py() -> str:
		logger.info("github_reader: fallback to git clone owner=%s repo=%s branch=%s", owner, repo, branch)
		tmp = wd / f"clone_{owner}_{repo}"
		if tmp.exists():
			shutil.rmtree(tmp, ignore_errors=True)
		subprocess.check_call(['git', 'clone', '--depth', '1', '-b', branch, f'https://github.com/{owner}/{repo}.git', str(tmp)])
		for p in tmp.rglob('*.py'):
			rel = p.relative_to(tmp)
			dst = out / rel
			dst.parent.mkdir(parents=True, exist_ok=True)
			shutil.copy2(p, dst)
		count = len(list(out.rglob('*.py')))
		logger.info("github_reader: clone materialized .py files=%d into %s", count, out)
		return str(out)

	# Попытка через LlamaIndex reader (как в ноутбуках)
	try:
		from llama_index.readers.github import GithubClient, GithubRepositoryReader
		token = os.getenv('GITHUB_TOKEN') or os.getenv('GITHUB_PAT') or ''
		client = GithubClient(github_token=token)
		reader = GithubRepositoryReader(
			github_client=client,
			owner=owner,
			repo=repo,
			filter_file_extensions=(
				".py",
				GithubRepositoryReader.FilterType.INCLUDE,
			),
		)
		logger.info("github_reader: using LlamaIndex reader owner=%s repo=%s branch=%s token=%s", owner, repo, branch, 'set' if token else 'empty')
		documents = reader.load_data(branch=branch)
		wrote_any = False
		for i, d in enumerate(documents):
			md = getattr(d, 'metadata', None) or {}
			file_name = md.get('file_name') or f'doc_{i}.py'
			if not str(file_name).endswith('.py'):
				continue
			text = getattr(d, 'text', '') or ''
			rel_path = md.get('file_path') or md.get('path') or file_name
			rel_path = str(rel_path).lstrip('/')
			p = out / rel_path
			p.parent.mkdir(parents=True, exist_ok=True)
			p.write_text(text, encoding='utf-8')
			wrote_any = True
		count = len(list(out.rglob('*.py')))
		logger.info("github_reader: reader materialized .py files=%d into %s", count, out)
		if not wrote_any:
			return _clone_then_copy_py()
		return str(out)
	except Exception as e:
		logger.exception("github_reader: reader failed, switching to clone: %s", e)
		# Фолбэк, если ридер недоступен или ошибка
		return _clone_then_copy_py()



