import os
import re

# Простой графовый RAG над исходниками: строим граф файлов/классов/функций,
# извлекаем связи import/contains, режем текст на чанки и отбираем
# релевантные узлы под бюджет контекста.


def _supported(path: str) -> bool:
    # Поддерживаемые расширения исходников
    exts = ('.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rb', '.php', '.rs', '.kt', '.m', '.swift', '.html', '.css')
    return path.endswith(exts)


def _read(path: str) -> str:
    # Толерантное чтение файла (пропускаем ошибки кодировки)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return ''


def _chunk(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    # Нестрогое разбиение на перекрывающиеся чанки для устойчивости к границам
    if size <= 0:
        return [text]
    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        chunks.append(text[i:i + size])
        i += max(1, size - overlap)
    return chunks


def _module_name(path: str) -> str:
    # Имя модуля для сопоставления import → файл
    base = os.path.basename(path)
    return os.path.splitext(base)[0].lower()


def _extract_imports(code: str) -> list[str]:
    # Простейшее извлечение импортов: import x / from x import y
    results: list[str] = []
    for m in re.finditer(r'\bimport\s+([a-zA-Z0-9_\.]+)', code):
        results.append(m.group(1).split('.')[0].lower())
    for m in re.finditer(r'\bfrom\s+([a-zA-Z0-9_\.]+)\s+import\s+', code):
        results.append(m.group(1).split('.')[0].lower())
    return results


def _extract_units(code: str) -> list[dict]:
    # Сигнатурное извлечение классов и функций (без парсера AST)
    units: list[dict] = []
    for m in re.finditer(r'\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\b', code):
        name = m.group(1)
        start = m.start()
        end = min(len(code), start + 2000)
        units.append({'type': 'class', 'name': name, 'start': start, 'end': end})
    for m in re.finditer(r'\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', code):
        name = m.group(1)
        start = m.start()
        end = min(len(code), start + 1200)
        units.append({'type': 'function', 'name': name, 'start': start, 'end': end})
    return units


def build_code_graph(root_dir: str) -> dict:
    # Обход каталога, построение узлов и рёбер графа
    nodes: dict[str, dict] = {}
    edges: list[tuple[str, str, str]] = []
    file_paths: list[str] = []
    for r, _, files in os.walk(root_dir):
        for f in files:
            p = os.path.join(r, f)
            if _supported(p):
                file_paths.append(p)
    name_to_path: dict[str, str] = {}
    for p in file_paths:
        mn = _module_name(p)
        if mn not in name_to_path:
            name_to_path[mn] = p
    for p in file_paths:
        code = _read(p)
        fid = f'file::{p}'
        nodes[fid] = {
            'id': fid,
            'type': 'file',
            'name': os.path.basename(p),
            'path': p,
            'text': code,
            'chunks': _chunk(code, 900, 150)
        }
        for imp in _extract_imports(code):
            if imp in name_to_path:
                tgt = f'file::{name_to_path[imp]}'
                edges.append((fid, tgt, 'imports'))
        for u in _extract_units(code):
            uid = f"{fid}::{u['type']}::{u['name']}"
            segment = code[u['start']:u['end']]
            nodes[uid] = {
                'id': uid,
                'type': u['type'],
                'name': u['name'],
                'path': p,
                'text': segment,
                'chunks': _chunk(segment, 700, 120)
            }
            edges.append((fid, uid, 'contains'))
    graph = {'nodes': nodes, 'edges': edges}
    return graph


def _tokenize(text: str) -> list[str]:
    # Грубый токенайзер по небуквенным символам
    return [t for t in re.split(r'[^a-zA-Z0-9_]+', (text or '').lower()) if t]


def _score_node(node: dict, checklist: str | None, indeg: dict[str, int], outdeg: dict[str, int]) -> float:
    # Эвристика ранжирования: семантические совпадения, центральность, длина
    s = 0.0
    name = (node.get('name') or '').lower()
    tks = set(_tokenize(node.get('text') or ''))
    if checklist:
        q = _tokenize(checklist)
        for qtok in q:
            if qtok in tks or qtok in name:
                s += 3.0
    if any(k in name for k in ['main', 'init', 'app', 'server', 'bot', 'handler']):
        s += 2.5
    if node['type'] == 'file':
        s += 1.0
    s += 0.5 * float(indeg.get(node['id'], 0))
    s += 0.3 * float(outdeg.get(node['id'], 0))
    if len(node.get('text') or '') > 0:
        s += min(2.0, len(node['text']) / 5000.0)
    return s


def _degrees(edges: list[tuple[str, str, str]]) -> tuple[dict[str, int], dict[str, int]]:
    # Быстрый расчёт in/out степеней для эвристик
    indeg: dict[str, int] = {}
    outdeg: dict[str, int] = {}
    for a, b, _ in edges:
        outdeg[a] = outdeg.get(a, 0) + 1
        indeg[b] = indeg.get(b, 0) + 1
    return indeg, outdeg


def select_context_nodes(graph: dict, checklist: str | None, budget_chars: int = 6000) -> str:
    # Выбор верхних узлов по скору и сборка контекста под бюджет
    nodes = graph['nodes']
    edges = graph['edges']
    indeg, outdeg = _degrees(edges)
    scored = []
    for nid, node in nodes.items():
        scored.append((nid, _score_node(node, checklist, indeg, outdeg)))
    scored.sort(key=lambda x: x[1], reverse=True)
    picked: list[str] = []
    total = 0
    for nid, _ in scored:
        node = nodes[nid]
        header = f"=== {node['type']} | {node.get('name') or ''} | {node['path']} ===\n"
        content = header
        for ch in node['chunks'][:2]:
            content += ch + '\n'
        if total + len(content) > budget_chars:
            break
        picked.append(content)
        total += len(content)
        if total >= budget_chars:
            break
    rels: dict[str, list[str]] = {}
    for a, b, t in edges:
        rels.setdefault(a, []).append(f'{t}->{b}')
    rel_lines: list[str] = []
    for a, lst in list(rels.items())[:40]:
        rel_lines.append(f'{a} : ' + ', '.join(lst[:6]))
    graph_summary = 'Граф связей (сокращенно):\n' + '\n'.join(rel_lines) + '\n\n'
    return graph_summary + '\n'.join(picked)


def prepare_graph_rag_context(root_dir: str, checklist: str | None) -> str:
    # Однопроходная сборка контекста для промпта
    g = build_code_graph(root_dir)
    return select_context_nodes(g, checklist, 6000)


def prepare_graph_rag_batches(root_dir: str, checklist: str | None, batch_budget: int = 3500, max_batches: int = 4) -> list[str]:
    # Формирование нескольких батчей для многопроходного анализа
    g = build_code_graph(root_dir)
    nodes = g['nodes']
    edges = g['edges']
    indeg, outdeg = _degrees(edges)
    scored = []
    for nid, node in nodes.items():
        scored.append((nid, _score_node(node, checklist, indeg, outdeg)))
    scored.sort(key=lambda x: x[1], reverse=True)
    batches: list[str] = []
    used: set[str] = set()
    idx = 0
    while idx < len(scored) and len(batches) < max_batches:
        total = 0
        parts: list[str] = []
        while idx < len(scored) and total < batch_budget:
            nid, _ = scored[idx]
            idx += 1
            if nid in used:
                continue
            node = nodes[nid]
            header = f"=== {node['type']} | {node.get('name') or ''} | {node['path']} ===\n"
            content = header
            for ch in node['chunks'][:2]:
                content += ch + '\n'
            if total + len(content) > batch_budget and parts:
                break
            parts.append(content)
            used.add(nid)
            total += len(content)
        if parts:
            batches.append('\n'.join(parts))
        else:
            break
    return batches


