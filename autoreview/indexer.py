import os
import re
from pathlib import Path


class CodeChunk(dict):
	pass


class CodeIndex:
	def __init__(self, root_dir: str):
		self.root_dir = str(Path(root_dir).resolve())
		self.chunks: list[CodeChunk] = []
		self._bm25 = None

	def build(self, max_file_bytes: int = 200_000, chunk_size: int = 400, overlap: int = 50) -> None:
		root = Path(self.root_dir)
		ex_c = {'.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip', '.lock', '.ico', '.ttf', '.woff', '.woff2'}
		for p in sorted(root.rglob('*')):
			if not p.is_file() or p.suffix.lower() in ex_c:
				continue
			try:
				text = p.read_text(encoding='utf-8', errors='ignore')
			except Exception:
				continue
			if len(text) > max_file_bytes:
				text = text[:max_file_bytes]
			lines = text.splitlines()
			start = 0
			while start < len(lines):
				end = min(len(lines), start + chunk_size)
				snippet = '\n'.join(lines[start:end])
				self.chunks.append(CodeChunk({
					'file': str(p.relative_to(root)),
					'lines': [start + 1, end],
					'text': snippet,
				}))
				start = end - overlap
				if start < 0:
					start = 0

		try:
			from llama_index.core import SimpleDirectoryReader
			from llama_index.core import Document
			from llama_index.core import get_response_synthesizer
			from llama_index.core.indices.keyword_table import KeywordTableIndex
			from llama_index.core.retrievers import BM25Retriever
			docs = [Document(text=c['text'], metadata={'file': c['file'], 'lines': c['lines']}) for c in self.chunks]
			index = KeywordTableIndex.from_documents(docs)
			self._bm25 = BM25Retriever.from_defaults(documents=docs, similarity_top_k=10)
		except Exception:
			self._bm25 = None

	def retrieve(self, query: str, top_k: int = 8) -> list[CodeChunk]:
		if self._bm25 is not None:
			try:
				nodes = self._bm25.retrieve(query)
				hits = []
				for n in nodes[:top_k]:
					md = n.node.metadata or {}
					hits.append(CodeChunk({'file': md.get('file'), 'lines': md.get('lines'), 'text': n.node.get_content()}))
				return hits
			except Exception:
				pass
		tokens = set(re.findall(r'[A-Za-zА-Яа-я0-9_\-]{3,}', query.lower()))
		scored = []
		for c in self.chunks:
			text_l = c['text'].lower()
			score = sum(text_l.count(t) for t in tokens)
			if score:
				scored.append((score, c))
		scored.sort(key=lambda x: x[0], reverse=True)
		return [c for _, c in scored[:top_k]]
