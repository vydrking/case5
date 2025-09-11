import os
import subprocess
from html.parser import HTMLParser

from botapp.prompts import SYSTEM_PROMPT, COMMON_RULES
from botapp.graph_rag import prepare_graph_rag_context
from botapp.http import get_stream
from botapp.config import TELEGRAM_TOKEN, YANDEX_API_KEY, YANDEX_FOLDER_ID, YANDEXGPT_URL
from botapp.http import post_json

MAX_TOKENS = 20000

def _clip(text: str | None, max_len: int) -> str:
    t = text or ''
    if len(t) > max_len:
        return t[:max_len]
    return t

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
    from botapp.review import call_yandex
    chunks = (_split_text(text, per_chunk_chars) or [''])[:3]
    partials: list[str] = []
    for ch in chunks:
        prompt = 'Извлеки краткие проверочные правила из текста. Строго выведи нумерованный список, по одному правилу в строке, без пояснений. ' + f'Не более {partial_limit} символов.\n\nТекст:\n' + _clip(ch, per_chunk_chars)
        out = call_yandex('yandexgpt-lite', prompt, temperature=0.1, max_tokens=450)
        partials.append(_clip(out, partial_limit))
    joined = '\n'.join(partials)
    final_prompt = 'Объедини правила ниже, убери повторы, нормализуй формулировки. Строго выведи нумерованный список без пояснений. ' + f'Общий объём не более {final_limit} символов.\n\n' + _clip(joined, 12000)
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

def _build_common_rules_if_needed(description: str | None, checklist: str | None) -> str | None:
    if checklist or description:
        return None
    return COMMON_RULES

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

def analyze_code(code_text: str, description: str | None = None, checklist: str | None = None) -> str:
    try:
        prompt = SYSTEM_PROMPT + '\n\n'
        rules = _build_rules(description, checklist) or _build_common_rules_if_needed(description, checklist)
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
        resp = post_json(YANDEXGPT_URL, data, headers, 60)
        if resp.status_code != 200:
            return f'Ошибка API ({resp.status_code}): {resp.text}'
        return resp.json()['result']['alternatives'][0]['message']['text']
    except Exception as e:
        return f'Ошибка при анализе кода: {str(e)}'

def analyze_code_simple(code_text: str, description: str | None = None, checklist: str | None = None) -> str:
    try:
        prompt = SYSTEM_PROMPT + '\n\n'
        rules = _build_rules(description, checklist) or _build_common_rules_if_needed(description, checklist)
        if rules:
            prompt += f'Свод правил для проверки (сжато):\n{rules}\n\n'
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
                'maxTokens': 1500
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
            return f'Ошибка API: {resp.status_code}'
        return resp.json()['result']['alternatives'][0]['message']['text']
    except Exception as e:
        return f'Ошибка: {str(e)}'

def analyze_multipass(batches: list[str], description: str | None, checklist: str | None, is_cancelled: callable | None = None) -> str:
    summaries: list[str] = []
    rules_full = _build_rules(description, checklist) or _build_common_rules_if_needed(description, checklist)
    rule_chunks = (_split_text(rules_full or '', 800) or ['']) if rules_full else ['']
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

def build_context_from_directory(directory: str, checklist: str | None) -> str:
    ctx = prepare_graph_rag_context(directory, checklist)
    if not ctx or not ctx.strip():
        return 'Не найдено файлов с кодом для анализа.'
    if len(ctx) > 9000:
        ctx = ctx[:9000] + '\n\n... [контекст обрезан]'
    return ctx

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

def clone_github_repo(url: str, path: str) -> None:
    subprocess.run(['git', 'clone', '--depth', '1', url, path], check=True)


