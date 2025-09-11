import os
import tempfile
import requests
import subprocess
import certifi
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from dotenv import load_dotenv
from botapp.graph_rag import prepare_graph_rag_context
from botapp.graph_rag import prepare_graph_rag_batches
from patoolib import extract_archive
import zipfile
import tarfile
from html.parser import HTMLParser
import json
import shutil
from collections.abc import Callable

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'), override=True)

_COMBINED_CA_PATH = None

def _get_ca_bundle_path() -> str:
    global _COMBINED_CA_PATH
    base = certifi.where()
    extra = os.getenv('EXTRA_CA_BUNDLE')
    if not extra or not os.path.isfile(extra):
        return base
    if _COMBINED_CA_PATH:
        return _COMBINED_CA_PATH
    try:
        combined = os.path.join(tempfile.gettempdir(), 'combined_cacert.pem')
        with open(base, 'rb') as b, open(extra, 'rb') as e, open(combined, 'wb') as out:
            out.write(b.read())
            out.write(b'\n')
            out.write(e.read())
        _COMBINED_CA_PATH = combined
        return combined
    except Exception:
        return base

os.environ['SSL_CERT_FILE'] = _get_ca_bundle_path()
os.environ['REQUESTS_CA_BUNDLE'] = _get_ca_bundle_path()
os.environ['CURL_CA_BUNDLE'] = _get_ca_bundle_path()

def _assert_ca_bundle() -> None:
    ca = _get_ca_bundle_path()
    try:
        with open(ca, 'rb') as _:
            pass
        print(f'CA bundle OK: {ca}')
    except Exception as e:
        print(f'CA bundle check failed: {ca} -> {e}')


def post_json(url: str, data: dict, headers: dict, timeout: int = 60) -> requests.Response:
    s = requests.Session()
    s.trust_env = False
    return s.post(url, json=data, headers=headers, timeout=timeout, verify=_get_ca_bundle_path())


def get_stream(url: str, timeout: int = 120) -> requests.Response:
    s = requests.Session()
    s.trust_env = False
    return s.get(url, stream=True, timeout=timeout, verify=_get_ca_bundle_path())

def _clean_env(val: str | None) -> str | None:
    if not isinstance(val, str):
        return None
    v = val.strip()
    if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
        v = v[1:-1]
    return v or None

def _clip(text: str | None, max_len: int) -> str:
    t = text or ''
    if len(t) > max_len:
        return t[:max_len]
    return t

MAX_TOKENS = 20000

def _clip_tokens(text: str | None, max_tokens: int = MAX_TOKENS) -> str:
    t = text or ''
    max_chars = max(0, int(max_tokens * 2 - 1000))
    if len(t) > max_chars:
        return t[:max_chars]
    return t

def _split_text(text: str, size: int) -> list[str]:
    t = text or ''
    if not t:
        return []
    parts: list[str] = []
    i = 0
    n = len(t)
    while i < n:
        parts.append(t[i:i + size])
        i += size
    return parts

def _mine_rules(text: str, per_chunk_chars: int = 4000, partial_limit: int = 600, final_limit: int = 1200) -> str:
    chunks = (_split_text(text, per_chunk_chars) or [''])[:3]
    partials: list[str] = []
    for idx, ch in enumerate(chunks, 1):
        prompt = (
            'Извлеки краткие проверочные правила из текста. '
            'Строго выведи нумерованный список, по одному правилу в строке, без пояснений. '
            f'Не более {partial_limit} символов.\n\nТекст:\n' + _clip(ch, per_chunk_chars)
        )
        out = call_yandex('yandexgpt-lite', prompt, temperature=0.1, max_tokens=450)
        partials.append(_clip(out, partial_limit))
    joined = '\n'.join(partials)
    final_prompt = (
        'Объедини правила ниже, убери повторы, нормализуй формулировки. '
        'Строго выведи нумерованный список без пояснений. '
        f'Общий объём не более {final_limit} символов.\n\n' + _clip(joined, 12000)
    )
    return _clip(call_yandex('yandexgpt-lite', final_prompt, temperature=0.1, max_tokens=600), final_limit)

def _build_rules(description: str | None, checklist: str | None) -> str | None:
    text = ''
    if description:
        text += _clip(description, 120000)
    if checklist:
        if text:
            text += '\n\n'
        text += _clip(checklist, 120000)
    if not text:
        return None
    return _mine_rules(text, 6000, 900, 2500)

def _split_rules_text(rules: str | None, max_lines: int = 18, max_chars: int = 900) -> list[str]:
    if not rules:
        return []
    lines = [ln.strip() for ln in (rules or '').splitlines() if ln.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for ln in lines:
        ln2 = ln[:max_chars]
        if len(cur) >= max_lines or cur_len + len(ln2) + 1 > max_chars:
            if cur:
                chunks.append('\n'.join(cur))
            cur = [ln2]
            cur_len = len(ln2)
        else:
            cur.append(ln2)
            cur_len += len(ln2) + 1
    if cur:
        chunks.append('\n'.join(cur))
    return chunks

TELEGRAM_TOKEN = _clean_env(os.getenv('TELEGRAM_BOT_TOKEN'))
YANDEX_FOLDER_ID = _clean_env(os.getenv('YANDEX_FOLDER_ID'))
YANDEX_API_KEY = _clean_env(os.getenv('YANDEX_API_KEY'))
YANDEXGPT_URL = 'https://llm.api.cloud.yandex.net/foundationModels/v1/completion'
from botapp.prompts import SYSTEM_PROMPT, REVIEWER_PROMPT

SESSIONS: dict[int, dict] = {}

INFERENCE_URL = os.getenv('INFERENCE_URL') or 'http://127.0.0.1:8080'

def _service_analyze(context_text: str, description: str | None, checklist: str | None, mode: str | None = None) -> tuple[str, str]:
    url = INFERENCE_URL.rstrip('/') + '/analyze'
    payload = {
        'context': context_text or '',
        'description': description,
        'checklist': checklist,
        'mode': mode
    }
    try:
        resp = post_json(url, payload, headers={}, timeout=180)
        if resp.status_code != 200:
            return (f"Ошибка сервиса ({resp.status_code}): {resp.text}", '')
        data = resp.json()
        return (data.get('report') or '', data.get('reviewer') or '')
    except Exception as e:
        return (f'Ошибка сервиса: {str(e)}', '')

def _service_multipass(batches: list[str], description: str | None, checklist: str | None) -> tuple[str, str]:
    url = INFERENCE_URL.rstrip('/') + '/multipass'
    payload = {
        'batches': batches or [],
        'description': description,
        'checklist': checklist
    }
    try:
        resp = post_json(url, payload, headers={}, timeout=300)
        if resp.status_code != 200:
            return (f"Ошибка сервиса ({resp.status_code}): {resp.text}", '')
        data = resp.json()
        return (data.get('report') or '', data.get('reviewer') or '')
    except Exception as e:
        return (f'Ошибка сервиса: {str(e)}', '')

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        'Привет! Я проверю ваш проект и сформирую два отчёта: для студента и для ревьюера.\n\n'
        'Что делать:\n'
        '- Пришлите архив проекта (.zip/.tar/.7z/.rar) или ссылку на GitHub\n'
        '- При необходимости приложите описание/чеклист (.txt/.md/.yaml/.html)\n\n'
        'Команды:\n'
        '- /start — это сообщение\n'
        '- /rag — статус RAG и размер контекста\n'
        '- /cancel — отменить текущую операцию'
    )
    chat_id = update.effective_chat.id
    sess = SESSIONS.setdefault(chat_id, {'description': None, 'checklist': None, 'multi_pass': True, 'gen': 0})
    sess['awaiting_extras'] = True
    await update.message.reply_text('Пришлите файл чеклиста (.txt/.md/.yaml/.html) или нажмите «Без доп. файлов».', reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton('Без доп. файлов', callback_data='skip_extras')]]))

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    sess = SESSIONS.setdefault(chat_id, {'description': None, 'checklist': None, 'multi_pass': True, 'gen': 0})
    sess['gen'] = (sess.get('gen') or 0) + 1
    await update.message.reply_text('Текущая операция отменена.')

async def on_skip_extras(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    chat_id = q.message.chat.id
    sess = SESSIONS.setdefault(chat_id, {'description': None, 'checklist': None, 'multi_pass': True, 'gen': 0})
    sess['awaiting_extras'] = False
    if sess.get('checklist'):
        await q.edit_message_text('Описание пропущено. Проверка по чеклисту будет выполнена. Пришлите архив проекта или ссылку на GitHub.')
    else:
        await q.edit_message_text('Пропущены доп. файлы. Пришлите архив проекта или ссылку на GitHub.')

async def reset_extras(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    sess = SESSIONS.setdefault(chat_id, {'description': None, 'checklist': None, 'multi_pass': True, 'gen': 0})
    sess['description'] = None
    sess['checklist'] = None
    sess['awaiting_extras'] = True
    await update.message.reply_text('Чеклист и описание сброшены. Пришлите файл чеклиста или нажмите «Без доп. файлов».', reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton('Без доп. файлов', callback_data='skip_extras')]]))

async def rag_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    sess = SESSIONS.get(chat_id) or {}
    used = sess.get('rag_used')
    size = sess.get('rag_context_size')
    checklist = bool(sess.get('checklist'))
    description = bool(sess.get('description'))
    mp = sess.get('multi_pass')
    await update.message.reply_text(
        f"RAG: {'on' if used else 'off'} | ctx: {size or 0} symbols | чеклист: {'да' if checklist else 'нет'} | описание: {'да' if description else 'нет'} | режим: {'multi' if mp else 'single'}"
    )

def _extract_graph_section(ctx: str) -> str:
    if not ctx:
        return ''
    lines = ctx.splitlines()
    out = []
    capture = False
    for ln in lines:
        if ln.strip().startswith('Граф связей (сокращенно):'):
            capture = True
            out.append(ln)
            continue
        if capture:
            if not ln.strip():
                break
            out.append(ln)
    return '\n'.join(out)

def _summary_to_mermaid(summary: str) -> str:
    edges: list[tuple[str,str,str]] = []
    nodes: set[str] = set()
    for ln in summary.splitlines():
        if ' :' not in ln:
            continue
        left, right = ln.split(' : ', 1)
        src = left.strip()
        for part in right.split(','):
            part = part.strip()
            if '->' not in part:
                continue
            rel, tgt = part.split('->', 1)
            rel = rel.strip()
            tgt = tgt.strip()
            edges.append((src, tgt, rel))
            nodes.add(src)
            nodes.add(tgt)
    def sid(x: str) -> str:
        import re
        return re.sub(r'[^A-Za-z0-9_]', '_', x)[:80]
    def label(x: str) -> str:
        import os
        if '::' in x:
            base = x.rsplit('::', 1)[-1]
        else:
            base = os.path.basename(x)
        return base[:60]
    lines = ['graph LR']
    if not nodes and not edges:
        lines.append('N0["Граф пуст — пришлите архив или ссылку, затем вызовите /graph"]')
        return '\n'.join(lines)
    for n in list(nodes)[:200]:
        lines.append(f"{sid(n)}[\"{label(n)}\"]")
    for s,t,r in edges[:400]:
        arrow = '-->' if r != 'imports' else '---'
        lines.append(f"{sid(s)} {arrow} {sid(t)}:::rel")
    lines.append('classDef rel stroke:#888,stroke-width:1px,color:#444')
    return '\n'.join(lines)



def clone_github_repo(url: str, path: str) -> None:
    subprocess.run(['git', 'clone', '--depth', '1', url, path], check=True)


def extract_any_archive(src_path: str, out_dir: str) -> None:
    lower = src_path.lower()
    if lower.endswith('.zip'):
        with zipfile.ZipFile(src_path) as zf:
            zf.extractall(out_dir)
        return
    if lower.endswith(('.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz', '.txz')):
        mode = 'r'
        if lower.endswith(('.tar.gz', '.tgz')):
            mode = 'r:gz'
        elif lower.endswith(('.tar.bz2', '.tbz2')):
            mode = 'r:bz2'
        elif lower.endswith(('.tar.xz', '.txz')):
            mode = 'r:xz'
        with tarfile.open(src_path, mode) as tf:
            tf.extractall(out_dir)
        return
    if lower.endswith('.7z') and shutil.which('7z'):
        subprocess.run(['7z', 'x', '-y', f'-o{out_dir}', src_path], check=True)
        return
    if lower.endswith('.rar'):
        if shutil.which('unrar'):
            subprocess.run(['unrar', 'x', '-y', src_path, out_dir], check=True)
            return
        if shutil.which('unar'):
            subprocess.run(['unar', '-quiet', '-output-directory', out_dir, src_path], check=True)
            return
    try:
        extract_archive(src_path, outdir=out_dir)
    except Exception as e:
        hint = ''
        if lower.endswith('.7z'):
            hint = " Установите p7zip (например: brew install p7zip)."
        elif lower.endswith('.rar'):
            hint = " Установите unrar или unar (например: brew install unar)."
        raise RuntimeError(f'Не удалось распаковать архив: {e}.{hint}')

def analyze_code(code_text: str, description: str | None = None, checklist: str | None = None) -> str:
    try:
        prompt = SYSTEM_PROMPT + '\n\n'
        rules = _build_rules(description, checklist)
        if rules:
            prompt += f'Свод правил для проверки (сжато):\n{rules}\n\n'
        prompt += f'Контекст кода (Graph RAG):\n```\n{_clip(code_text, 12000)}\n```'
        prompt = _clip_tokens(prompt, MAX_TOKENS)

        headers = {
            'Authorization': f'Api-Key {YANDEX_API_KEY}',
            'Content-Type': 'application/json',
            'x-folder-id': YANDEX_FOLDER_ID
        }
        
        data = {
            'modelUri': f'gpt://{YANDEX_FOLDER_ID}/yandexgpt-lite',
            'completionOptions': {
                'stream': False,
                'temperature': 0.2,
                'maxTokens': 2000
            },
            'messages': [
                {
                    'role': 'user',
                    'text': prompt
                }
            ]
        }

        print('Отправка запроса к YandexGPT...')
        response = post_json(
            YANDEXGPT_URL, 
            data,
            headers,
            60
        )
        
        print(f'Статус ответа: {response.status_code}')
        
        if response.status_code != 200:
            print(f'Текст ошибки: {response.text}')
            return f'Ошибка API ({response.status_code}): {response.text}'
        
        result = response.json()
        return result['result']['alternatives'][0]['message']['text']
        
    except Exception as e:
        error_msg = f'Ошибка при анализе кода: {str(e)}'
        print(error_msg)
        return error_msg

def analyze_code_simple(code_text: str, description: str | None = None, checklist: str | None = None) -> str:
    try:
        prompt = SYSTEM_PROMPT + '\n\n'
        rules = _build_rules(description, checklist)
        if rules:
            prompt += f'Свод правил для проверки (сжато):\n{rules}\n\n'

        headers = {
            'Authorization': f'Api-Key {YANDEX_API_KEY}',
            'Content-Type': 'application/json',
            'x-folder-id': YANDEX_FOLDER_ID
        }
        
        data = {
            'modelUri': f'gpt://{YANDEX_FOLDER_ID}/yandexgpt-lite',
            'completionOptions': {
                'stream': False,
                'temperature': 0.2,
                'maxTokens': 1500
            },
            'messages': [
                {
                    'role': 'user', 
                    'text': prompt
                }
            ]
        }

        prompt = _clip_tokens(prompt, MAX_TOKENS)
        response = post_json(YANDEXGPT_URL, data, headers, 60)
        
        if response.status_code != 200:
            return f'Ошибка API: {response.status_code}'
        
        result = response.json()
        return result['result']['alternatives'][0]['message']['text']
        
    except Exception as e:
        return f'Ошибка: {str(e)}'

def call_yandex(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    headers = {
        'Authorization': f'Api-Key {YANDEX_API_KEY}',
        'Content-Type': 'application/json',
        'x-folder-id': YANDEX_FOLDER_ID
    }
    data = {
        'modelUri': f'gpt://{YANDEX_FOLDER_ID}/{model}',
        'completionOptions': {
            'stream': False,
            'temperature': temperature,
            'maxTokens': max_tokens
        },
        'messages': [
            {
                'role': 'user',
                'text': prompt
            }
        ]
    }
    resp = post_json(YANDEXGPT_URL, data, headers, 60)
    if resp.status_code != 200:
        return f"Ошибка API ({resp.status_code}): {resp.text}"
    return resp.json()['result']['alternatives'][0]['message']['text']

def summarize_for_reviewer(student_report: str) -> str:
    prompt = REVIEWER_PROMPT + '\n\n' + _clip(student_report, 120000)
    prompt = _clip_tokens(prompt, MAX_TOKENS)
    return call_yandex('yandexgpt-lite', prompt, temperature=0.1, max_tokens=800)

def analyze_multipass(batches: list[str], description: str | None, checklist: str | None, is_cancelled: Callable | None = None) -> str:
    summaries: list[str] = []
    rules_full = _build_rules(description, checklist)
    rule_chunks = _split_rules_text(rules_full, max_lines=14, max_chars=800) or ['']
    for i, ctx in enumerate(batches, 1):
        for j, rule_chunk in enumerate(rule_chunks, 1):
            if is_cancelled and is_cancelled():
                return 'Операция отменена.'
            prompt = SYSTEM_PROMPT + '\n\n'
            if rule_chunk:
                prompt += f'Правила для проверки (часть {j}/{len(rule_chunks)}):\n{rule_chunk}\n\n'
            prompt += f'Контекст кода (Graph RAG, batch {i}/{len(batches)}):\n```\n{_clip(ctx, 6500)}\n```\n'
            prompt += 'Дай краткое резюме замечаний для этого батча по указанным правилам. Без повторов кода.'
            prompt = _clip_tokens(prompt, MAX_TOKENS)
            out = call_yandex('yandexgpt-lite', prompt, temperature=0.2, max_tokens=700)
            summaries.append(_clip(out, 8000))
    final_prompt = SYSTEM_PROMPT + '\n\n'
    if rules_full:
        final_prompt += f'Свод правил (для справки):\n{_clip(rules_full, 1200)}\n\n'
    joined = '\n\n'.join(summaries)
    if len(joined) > 90000:
        joined = joined[:90000]
    final_prompt += 'Ниже частичные отчёты, объедини их, убери повторы, сделай единый отчёт по требуемым разделам. В разделе про чеклист: не придумывай нумерацию; используй лейблы; для невыполненных/частичных пунктов процитируй текст требования перед статусом; если чеклист отсутствует — так и напиши.\n' + joined
    final_prompt = _clip_tokens(final_prompt, MAX_TOKENS)
    return call_yandex('yandexgpt', final_prompt, temperature=0.2, max_tokens=1600)

async def handle_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text or update.message.caption
    chat_id = update.effective_chat.id
    sess = SESSIONS.setdefault(chat_id, {'description': None, 'checklist': None, 'multi_pass': True, 'gen': 0})
    gen = sess.get('gen') or 0
    def _is_cancelled() -> bool:
        return (SESSIONS.get(chat_id) or {}).get('gen') != gen
    checklist = None
    processing_msg = None

    if update.message.document:
        file = await update.message.document.get_file()
        with tempfile.TemporaryDirectory() as tmp_dir:
            fpath = os.path.join(tmp_dir, update.message.document.file_name or 'file.bin')
            await asyncio.to_thread(download_telegram_file, file, fpath)
            if _is_cancelled():
                await update.message.reply_text('Отмена: обработка файла прервана.')
                return
            name_lower = fpath.lower()
            is_archive = name_lower.endswith(('.zip', '.tar', '.rar', '.7z', '.tar.gz', '.tgz'))
            is_textdoc = name_lower.endswith(('.txt', '.md', '.rst', '.markdown', '.yaml', '.yml'))
            is_htmldoc = name_lower.endswith(('.html', '.htm'))
            # Запрос доп. файлов (чеклист/описание) до проекта
            if sess.get('awaiting_extras') and (is_textdoc or is_htmldoc):
                try:
                    with open(fpath, 'r', encoding='utf-8') as tf:
                        raw = tf.read()
                    text = raw
                    if is_htmldoc:
                        class _Plain(HTMLParser):
                            def __init__(self):
                                super().__init__()
                                self.parts: list[str] = []
                            def handle_data(self, data: str):
                                self.parts.append(data)
                            def get_text(self) -> str:
                                return ' '.join(self.parts)
                        p = _Plain()
                        p.feed(raw)
                        text = p.get_text()
                    fname = os.path.basename(fpath)
                    if not sess.get('checklist'):
                        sess['checklist'] = text
                        await update.message.reply_text(f'Чеклист получен: {fname}')
                        await update.message.reply_text(
                            'Теперь пришлите файл описания проекта (.txt/.md/.yaml/.html) или нажмите «Без доп. файлов».',
                            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton('Без доп. файлов', callback_data='skip_extras')]])
                        )
                    elif not sess.get('description'):
                        sess['description'] = text
                        await update.message.reply_text(f'Описание проекта получено: {fname}')
                        sess['awaiting_extras'] = False
                        await update.message.reply_text('Теперь пришлите архив проекта или ссылку на GitHub.')
                    else:
                        await update.message.reply_text('Доп. файлы уже получены. Пришлите проект или нажмите /reset для сброса.')
                except Exception as e:
                    await update.message.reply_text(f'Не удалось прочитать файл: {str(e)}')
                return
            if sess.get('awaiting_extras') and is_archive:
                await update.message.reply_text(
                    'Сначала пришлите чеклист/описание или нажмите «Без доп. файлов».',
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton('Без доп. файлов', callback_data='skip_extras')]])
                )
                return
            if is_archive:
                try:
                    processing_msg = await update.message.reply_text('<i>Проект обрабатывается, подождите…</i>', parse_mode='HTML')
                except Exception:
                    processing_msg = None
                extract_path = os.path.join(tmp_dir, 'extracted')
                os.makedirs(extract_path, exist_ok=True)
                try:
                    extract_any_archive(fpath, extract_path)
                    if _is_cancelled():
                        await update.message.reply_text('Отмена: распаковка выполнена, дальнейшая обработка прервана.')
                        return
                    if sess.get('multi_pass'):
                        _ctx_once = build_context_from_directory(extract_path, sess.get('checklist'))
                        sess['last_graph_summary'] = _extract_graph_section(_ctx_once)
                        if _is_cancelled():
                            await update.message.reply_text('Отмена: подготовка контекста RAG прервана.')
                            return
                        batches = prepare_graph_rag_batches(extract_path, sess.get('checklist'), 2000, 2)
                        sess['rag_used'] = True
                        sess['rag_context_size'] = sum(len(b) for b in batches)
                        rep, rev = _service_multipass(batches, sess.get('description'), sess.get('checklist'))
                        if processing_msg:
                            try:
                                await processing_msg.delete()
                            except Exception:
                                pass
                        await update.message.reply_text(rep[:4096])
                        if rev:
                            await update.message.reply_text('Отчёт для ревьюера:\n' + rev[:4096])
                        return
                    code_text = build_context_from_directory(extract_path, None)
                    sess['rag_used'] = True
                    sess['rag_context_size'] = len(code_text)
                    sess['last_graph_summary'] = _extract_graph_section(code_text)
                    if _is_cancelled():
                        await update.message.reply_text('Отмена: подготовка контекста выполнена, анализ прерван.')
                        return
                    if user_input and 'чеклист' in user_input.lower():
                        sess['checklist'] = user_input
                        code_text = build_context_from_directory(extract_path, sess['checklist'])
                        sess['rag_context_size'] = len(code_text)
                        sess['last_graph_summary'] = _extract_graph_section(code_text)
                    if _is_cancelled():
                        await update.message.reply_text('Отмена: анализ прерван.')
                        return
                    rep, rev = _service_analyze(code_text, sess.get('description'), sess.get('checklist'))
                    if 'Ошибка' in rep or len(rep) < 50:
                        rep, rev = _service_analyze(code_text, sess.get('description'), sess.get('checklist'), mode='simple')
                    if processing_msg:
                        try:
                            await processing_msg.delete()
                        except Exception:
                            pass
                    await update.message.reply_text(rep[:4096])
                    if rev:
                        await update.message.reply_text('Отчёт для ревьюера:\n' + rev[:4096])
                except Exception as e:
                    if processing_msg:
                        try:
                            await processing_msg.delete()
                        except Exception:
                            pass
                    await update.message.reply_text(f'Ошибка при обработке архива: {str(e)}')
            elif is_textdoc or is_htmldoc:
                try:
                    with open(fpath, 'r', encoding='utf-8') as tf:
                        raw = tf.read()
                    text = raw
                    if is_htmldoc:
                        class _Plain(HTMLParser):
                            def __init__(self):
                                super().__init__()
                                self.parts: list[str] = []
                            def handle_data(self, data: str):
                                self.parts.append(data)
                            def get_text(self) -> str:
                                return ' '.join(self.parts)
                        p = _Plain()
                        p.feed(raw)
                        text = p.get_text()
                    sess['description'] = text
                    if not sess.get('checklist'):
                        sess['checklist'] = text
                    await update.message.reply_text('Описание/чеклист получены. Теперь отправьте архив проекта или ссылку на GitHub.')
                except Exception as e:
                    await update.message.reply_text(f'Не удалось прочитать файл описания: {str(e)}')
            else:
                await update.message.reply_text('Формат файла не поддерживается. Пришлите архив проекта (.zip/.tar) или текстовый файл описания (.txt/.md).')

    elif user_input and 'github.com' in user_input:
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                try:
                    processing_msg = await update.message.reply_text('<i>Проект обрабатывается, подождите…</i>', parse_mode='HTML')
                except Exception:
                    processing_msg = None
                repo_url = user_input.split()[0]
                clone_github_repo(repo_url, tmp_dir)
                if _is_cancelled():
                    await update.message.reply_text('Отмена: клонирование выполнено, дальнейшая обработка прервана.')
                    return
                if sess.get('multi_pass'):
                    _ctx_once = build_context_from_directory(tmp_dir, sess.get('checklist'))
                    sess['last_graph_summary'] = _extract_graph_section(_ctx_once)
                    if _is_cancelled():
                        await update.message.reply_text('Отмена: подготовка контекста RAG прервана.')
                        return
                    batches = prepare_graph_rag_batches(tmp_dir, sess.get('checklist'), 2000, 2)
                    sess['rag_used'] = True
                    sess['rag_context_size'] = sum(len(b) for b in batches)
                    rep, rev = _service_multipass(batches, sess.get('description'), sess.get('checklist'))
                    if processing_msg:
                        try:
                            await processing_msg.delete()
                        except Exception:
                            pass
                    await update.message.reply_text(rep[:4096])
                    if rev:
                        await update.message.reply_text('Отчёт для ревьюера:\n' + rev[:4096])
                    return
                code_text = build_context_from_directory(tmp_dir, None)
                sess['rag_used'] = True
                sess['rag_context_size'] = len(code_text)
                sess['last_graph_summary'] = _extract_graph_section(code_text)
                
                if len(user_input.split()) > 1:
                    checklist = ' '.join(user_input.split()[1:])
                    sess['checklist'] = checklist
                    code_text = build_context_from_directory(tmp_dir, sess['checklist'])
                    sess['rag_context_size'] = len(code_text)
                    sess['last_graph_summary'] = _extract_graph_section(code_text)
                
                rep, rev = _service_analyze(code_text, sess.get('description'), sess.get('checklist'))
                if 'Ошибка' in rep or len(rep) < 50:
                    rep, rev = _service_analyze(code_text, sess.get('description'), sess.get('checklist'), mode='simple')
                if processing_msg:
                    try:
                        await processing_msg.delete()
                    except Exception:
                        pass
                
                await update.message.reply_text(rep[:4096])
                if rev:
                    await update.message.reply_text('Отчёт для ревьюера:\n' + rev[:4096])
                
            except Exception as e:
                if processing_msg:
                    try:
                        await processing_msg.delete()
                    except Exception:
                        pass
                await update.message.reply_text(f'Ошибка при обработке GitHub: {str(e)}')

    else:
        await update.message.reply_text('Отправь архив, ссылку на GitHub или текстовый файл описания (.txt/.md).')

def build_context_from_directory(directory: str, checklist: str | None) -> str:
    ctx = prepare_graph_rag_context(directory, checklist)
    if not ctx or not ctx.strip():
        return 'Не найдено файлов с кодом для анализа.'
    if len(ctx) > 9000:
        ctx = ctx[:9000] + '\n\n... [контекст обрезан]'
    return ctx

def test_connection():
    try:
        headers = {
            'Authorization': f'Api-Key {YANDEX_API_KEY}',
            'Content-Type': 'application/json',
            'x-folder-id': YANDEX_FOLDER_ID
        }
        
        data = {
            'modelUri': f'gpt://{YANDEX_FOLDER_ID}/yandexgpt-lite',
            'completionOptions': {
                'stream': False,
                'temperature': 0.1,
                'maxTokens': 100
            },
            'messages': [
                {
                    'role': 'user', 
                    'text': 'Привет, это тест'
                }
            ]
        }

        response = post_json(YANDEXGPT_URL, data, headers, 30)
        print(f'Тест подключения: статус {response.status_code}')
        
        if response.status_code == 200:
            result = response.json()
            print('Тест успешен!')
            return True
        else:
            print(f'Ошибка теста: {response.text}')
            return False
            
    except Exception as e:
        print(f'Ошибка теста подключения: {e}')
        return False

def download_telegram_file(file_obj, dest_path: str) -> None:
    url = getattr(file_obj, 'file_path', None)
    if not url:
        raise ValueError('Нет пути к файлу')
    if not str(url).startswith('http'):
        url = f'https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{url}'
    with get_stream(str(url), 120) as resp:
        resp.raise_for_status()
        with open(dest_path, 'wb') as out:
            for chunk in resp.iter_content(8192):
                if chunk:
                    out.write(chunk)

def main():
    required_vars = ['TELEGRAM_BOT_TOKEN', 'YANDEX_FOLDER_ID', 'YANDEX_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Отсутствуют переменные: {', '.join(missing_vars)}")
        return
    
    _assert_ca_bundle()
    
    print("Тестируем подключение к YandexGPT...")
    if not test_connection():
        print("Не удалось подключиться к YandexGPT. Продолжаю запуск бота, проверки будут работать без теста.")
    
    print('Запуск бота...')
    async def _post_init(application: Application):
        await application.bot.delete_webhook(drop_pending_updates=True)
    app = Application.builder().token(TELEGRAM_TOKEN).post_init(_post_init).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('cancel', cancel))
    app.add_handler(CommandHandler('reset', reset_extras))
    app.add_handler(CommandHandler('rag', rag_status))
    app.add_handler(CallbackQueryHandler(on_skip_extras, pattern='^skip_extras$'))
    app.add_handler(MessageHandler((filters.TEXT & ~filters.COMMAND) | filters.Document.ALL, handle_input))
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()