from typing import Any


def normalize_line_range(lines: list[int]) -> tuple[int, int]:
	if not lines:
		return (0, 0)
	if len(lines) == 1:
		return (int(lines[0]), int(lines[0]))
	return (int(min(lines[0], lines[1])), int(max(lines[0], lines[1])))


def dedupe_and_group(issues: list[dict]) -> list[dict]:
	seen: set[tuple[str, int, int, str]] = set()
	result: list[dict] = []
	for it in issues:
		file = it.get('file') or ''
		start, end = normalize_line_range(it.get('lines') or [])
		rule = it.get('rule') or ''
		key = (file, start, end, rule)
		if key in seen:
			continue
		seen.add(key)
		result.append({
			'file': file,
			'lines': [start] if start == end else [start, end],
			'rule': rule,
			'description': it.get('description') or '',
			'suggestion': it.get('suggestion') or '',
		})
	result.sort(key=lambda x: (x['file'], (x['lines'][0] if x['lines'] else 0)))
	return result
