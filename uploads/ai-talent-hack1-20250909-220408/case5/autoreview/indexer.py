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
		self._vec_retriever = None
		self._use_rrf = True

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
			from llama_index.core import Document
			from llama_index.core.indices.keyword_table import KeywordTableIndex
			from llama_index.core.retrievers import BM25Retriever
			from llama_index.core import VectorStoreIndex
			from case5.autoreview.yandex_embeddings import YandexEmbedding
			docs = [Document(text=c['text'], metadata={'file': c['file'], 'lines': c['lines']}) for c in self.chunks]
			KeywordTableIndex.from_documents(docs)
			self._bm25 = BM25Retriever.from_defaults(documents=docs, similarity_top_k=10)
			embed = YandexEmbedding()
			vindex = VectorStoreIndex.from_documents(docs, embed_model=embed)
			self._vec_retriever = vindex.as_retriever(similarity_top_k=10)
		except Exception:
			self._bm25 = None
			self._vec_retriever = None

	def retrieve(self, query: str, top_k: int = 8) -> list[CodeChunk]:
		# Гибридное извлечение с RRF (если доступно)
		bm25_nodes = []
		vec_nodes = []
		if self._bm25 is not None:
			try:
				bm25_nodes = list(self._bm25.retrieve(query) or [])
			except Exception:
				bm25_nodes = []
		if self._vec_retriever is not None:
			try:
				vec_nodes = list(self._vec_retriever.retrieve(query) or [])
			except Exception:
				vec_nodes = []
		if bm25_nodes or vec_nodes:
			# Reciprocal Rank Fusion
			k_rrf = 60
			scores: dict[str, float] = {}
			payload: dict[str, tuple[str, list[int], str]] = {}
			def add(nodes, weight: float = 1.0):
				for rank, n in enumerate(nodes):
					# формируем ключ по файлу+линиям+хэшу содержимого для дедуп
					md = getattr(n, 'node', None).metadata if hasattr(n, 'node') else (getattr(n, 'metadata', None) or {})
					text = getattr(n, 'node', None).get_content() if hasattr(n, 'node') else getattr(n, 'text', '')
					file = (md or {}).get('file') or ''
					lines = (md or {}).get('lines') or []
					key = f"{file}:{lines}:{hash(text)}"
					scores[key] = scores.get(key, 0.0) + weight * (1.0 / (k_rrf + rank + 1))
					payload[key] = (file, lines, text)
			add(bm25_nodes, weight=1.0)
			add(vec_nodes, weight=1.0)
			ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
			res: list[CodeChunk] = []
			for key, _ in ordered:
				file, lines, text = payload[key]
				res.append(CodeChunk({'file': file, 'lines': lines, 'text': text}))
			return res

		tokens = set(re.findall(r'[A-Za-zА-Яа-я0-9_\-]{3,}', query.lower()))
		scored = []
		for c in self.chunks:
			text_l = c['text'].lower()
			score = sum(text_l.count(t) for t in tokens)
			if score:
				scored.append((score, c))
		scored.sort(key=lambda x: x[0], reverse=True)
		return [c for _, c in scored[:top_k]]
