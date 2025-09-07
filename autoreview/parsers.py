import re
import zipfile
from pathlib import Path
from bs4 import BeautifulSoup


def read_html(path: str) -> str:
	p = Path(path)
	return p.read_text(encoding='utf-8', errors='ignore')


def parse_project_description(html_text: str) -> dict:
	soup = BeautifulSoup(html_text, 'lxml')
	title = soup.title.get_text(strip=True) if soup.title else ''
	headers = [h.get_text(' ', strip=True) for h in soup.select('h1, h2, h3')]
	paras = [p.get_text(' ', strip=True) for p in soup.select('p, li')]
	return {
		'title': title,
		'headers': headers,
		'content': '\n'.join(paras),
	}


def parse_checklist(html_text: str) -> dict:
	soup = BeautifulSoup(html_text, 'lxml')
	title = soup.title.get_text(strip=True) if soup.title else ''
	items = []
	for li in soup.select('li'):
		text = li.get_text(' ', strip=True)
		if text:
			items.append(text)
	return {'title': title, 'items': items}


def extract_zip(zip_path: str, out_dir: str) -> str:
	out = Path(out_dir)
	out.mkdir(parents=True, exist_ok=True)
	with zipfile.ZipFile(zip_path, 'r') as zf:
		zf.extractall(out)
	return str(out)


def project_overview(root_dir: str) -> dict:
	root = Path(root_dir)
	files = []
	for p in root.rglob('*'):
		if p.is_file():
			files.append(str(p.relative_to(root)))
	return {'root': str(root), 'files': files}

