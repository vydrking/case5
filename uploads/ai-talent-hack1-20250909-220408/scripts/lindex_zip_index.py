#!/usr/bin/env python3
"""
Индексация проекта из ZIP с использованием LlamaIndex:
 - безопасная распаковка ZIP
 - сбор документов по расширениям
 - BM25 (KeywordTableIndex) + VectorStoreIndex (эмбеддинги: Yandex или FastEmbed)
 - опционально KnowledgeGraphIndex
 - сохранение артефактов top_hits.json и index_summary.json
"""

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Sequence
import sys

from llama_index.core import VectorStoreIndex
from llama_index.core.indices.keyword_table import KeywordTableIndex
from llama_index.core.schema import Document
from llama_index.core.settings import Settings
from llama_index.core.llms.mock import MockLLM


# Добавляем корень проекта в sys.path, чтобы работал импорт case5.autoreview
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

USE_YANDEX = bool(os.getenv("YANDEX_API_KEY") and os.getenv("YANDEX_FOLDER_ID"))
if USE_YANDEX:
    # адаптер эмбеддингов Yandex под LlamaIndex
    from case5.autoreview.yandex_embeddings import YandexEmbedding  # type: ignore
else:
    from llama_index.embeddings.fastembed import FastEmbedEmbedding  # type: ignore


def _normalize_exts(exts_arg: str | None) -> Sequence[str]:
    if not exts_arg:
        exts = [".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".md"]
    else:
        exts = []
        for e in exts_arg.split(","):
            e = e.strip()
            if not e:
                continue
            if not e.startswith("."):
                e = "." + e
            exts.append(e.lower())
    return tuple(sorted(set(exts)))


def _safe_extract(zip_path: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = (dest_dir / member.filename).resolve()
            if not str(member_path).startswith(str(dest_dir.resolve())):
                raise RuntimeError(f"Небезопасный путь в zip: {member.filename}")
        zf.extractall(dest_dir)


def _collect_documents(root: Path, allow_exts: Sequence[str]) -> List[Document]:
    docs: List[Document] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in allow_exts:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        rel = str(p.relative_to(root))
        docs.append(Document(text=text, metadata={"file_path": rel, "file_name": p.name}))
    return docs


def _build_retrievers(docs: List[Document], top_k: int):
    # Убираем зависимость от openai-llm: подставим MockLLM как дефолт
    Settings.llm = MockLLM()
    kw_index = KeywordTableIndex.from_documents(docs)
    if USE_YANDEX:
        embed_model = YandexEmbedding()
    else:
        embed_model = FastEmbedEmbedding()
    vec_index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    bm25 = kw_index.as_retriever(retriever_mode="bm25", similarity_top_k=top_k)
    vret = vec_index.as_retriever(similarity_top_k=top_k)
    return bm25, vret


def _run_retrieval(bm25, vret, out_dir: Path) -> None:
    queries = [
        "инструкции запуска (README, как запустить)",
        "основная точка входа (main, app, server)",
        "инициализация приложения",
        "маршруты/роутеры",
        "конфигурация/настройки",
    ]
    results = []
    for q in queries:
        item = {"query": q, "bm25": [], "vec": []}
        try:
            bm = bm25.retrieve(q)
        except Exception as e:
            bm = []
            item["bm25_error"] = str(e)
        try:
            vr = vret.retrieve(q)
        except Exception as e:
            vr = []
            item["vec_error"] = str(e)

        for n in bm[:5]:
            md = getattr(n, "node", None).metadata if hasattr(n, "node") else {}
            item["bm25"].append({
                "file": md.get("file_path") or md.get("file_name"),
                "lines": md.get("lines"),
            })

        for n in vr[:5]:
            md = getattr(n, "node", None).metadata if hasattr(n, "node") else {}
            item["vec"].append({
                "file": md.get("file_path") or md.get("file_name"),
                "lines": md.get("lines"),
            })

        results.append(item)

    (out_dir / "top_hits.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def _try_build_kg(docs: List[Document]) -> bool:
    try:
        from llama_index.core import KnowledgeGraphIndex
        kg_index = KnowledgeGraphIndex.from_documents(docs, max_triplets_per_chunk=5)
        engine = kg_index.as_query_engine(include_text=True)
        _ = engine.query("Опиши основные связи между файлами проекта.")
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser("lindex_zip_index")
    ap.add_argument("--zip", required=True, help="Путь к ZIP-файлу с проектом")
    ap.add_argument("--out", default=str(Path.cwd() / "workdir" / "lindex_zip"), help="Директория для результатов")
    ap.add_argument("--ext", default=None, help="Расширения через запятую (py,js,ts,html,css,…)")
    ap.add_argument("--topk", type=int, default=8, help="Top-K для ретриверов")
    ap.add_argument("--kg", action="store_true", help="Пытаться строить KnowledgeGraphIndex")
    ap.add_argument("--keep-extracted", action="store_true", help="Не удалять распакованный проект")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    allow_exts = _normalize_exts(args.ext)
    zip_path = Path(args.zip).resolve()
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP не найден: {zip_path}")

    extract_dir = Path(tempfile.mkdtemp(prefix="zipidx_"))
    print(f"[i] Extracting: {zip_path} -> {extract_dir}")
    _safe_extract(zip_path, extract_dir)

    print(f"[i] Collecting documents (ext: {', '.join(allow_exts)}) …")
    docs = _collect_documents(extract_dir, allow_exts)
    print(f"[i] Documents collected: {len(docs)}")

    if not docs:
        summary = {"ok": False, "reason": "no_docs", "docs_count": 0}
        (out_dir / "index_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if not args.keep_extracted:
            shutil.rmtree(extract_dir, ignore_errors=True)
        return 0

    print("[i] Building retrievers (BM25, Vector)…")
    bm25, vret = _build_retrievers(docs, args.topk)

    print("[i] Running retrieval smoke-tests…")
    _run_retrieval(bm25, vret, out_dir)

    kg_enabled = False
    if args.kg:
        kg_enabled = _try_build_kg(docs)

    summary = {
        "ok": True,
        "docs_count": len(docs),
        "embed_model": "YANDEX (yandexgpt-embedding-lite)" if USE_YANDEX else "FastEmbed",
        "kg_enabled": bool(kg_enabled),
        "out_dir": str(out_dir),
        "exts": list(allow_exts),
        "topk": args.topk,
    }
    (out_dir / "index_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.keep_extracted:
        shutil.rmtree(extract_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())


