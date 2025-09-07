import os
import re
import glob
from pathlib import Path


def collect_text_samples(root_dir: str, limit_bytes: int = 80000) -> dict:
	root = Path(root_dir)
	acc = {}
	remaining = limit_bytes
	for p in sorted(root.rglob('*')):
		if p.is_file() and p.suffix.lower() in {'.md', '.txt', '.py', '.js', '.ts', '.html', '.css'}:
			try:
				data = p.read_text(encoding='utf-8', errors='ignore')
			except Exception:
				continue
			if remaining <= 0:
				break
			chunk = data[: min(len(data), max(0, remaining))]
			acc[str(p.relative_to(root))] = chunk
			remaining -= len(chunk)
	return acc


def naive_quality_checks(root_dir: str) -> list[str]:
	issues = []
	for p in Path(root_dir).rglob('*'):
		if p.is_file():
			if p.suffix.lower() in {'.py'}:
				try:
					code = p.read_text(encoding='utf-8', errors='ignore')
					if 'print(' in code and 'if __name__' not in code:
						issues.append(f'Possible stray prints in {p.name}')
				except Exception:
					pass
	return issues


def run_autotests(root_dir: str, suite: dict) -> dict:
	root = Path(root_dir)
	results = []
	for t in suite.get('tests', []) or []:
		type_ = t.get('type')
		ok = False
		detail = ''
		if type_ == 'file_exists':
			p = root / (t.get('path') or '')
			ok = p.exists()
			detail = str(p)
		elif type_ == 'glob_exists':
			pat = str(root / (t.get('glob') or ''))
			matches = glob.glob(pat, recursive=True)
			ok = len(matches) > 0
			detail = ','.join(matches[:5])
		elif type_ == 'file_contains':
			p = root / (t.get('path') or '')
			try:
				data = p.read_text(encoding='utf-8', errors='ignore')
				ok = (t.get('pattern') or '') in data
				detail = f"found={ok}"
			except Exception:
				ok = False
		elif type_ == 'grep_count':
			p = root / (t.get('path') or '')
			count_min = int(t.get('count_min') or 1)
			try:
				data = p.read_text(encoding='utf-8', errors='ignore')
				cnt = len(re.findall(t.get('pattern') or '', data))
				ok = cnt >= count_min
				detail = f"count={cnt}"
			except Exception:
				ok = False
		results.append({
			'id': t.get('id'),
			'type': type_,
			'ok': bool(ok),
			'explanation': t.get('explanation') or '',
			'detail': detail,
		})
	return {'results': results}
