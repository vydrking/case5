import os
import certifi
import tempfile
import requests
from dotenv import load_dotenv

from fastapi import FastAPI
from pydantic import BaseModel


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

# load .env from project root
try:
    root_env = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path=os.path.abspath(root_env), override=True)
except Exception:
    pass


def _clean_env(val: str | None) -> str | None:
    if not isinstance(val, str):
        return None
    v = val.strip()
    if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
        v = v[1:-1]
    return v or None


YANDEX_FOLDER_ID = _clean_env(os.getenv('YANDEX_FOLDER_ID'))
YANDEX_API_KEY = _clean_env(os.getenv('YANDEX_API_KEY'))
YANDEXGPT_URL = 'https://llm.api.cloud.yandex.net/foundationModels/v1/completion'


def post_json(url: str, data: dict, headers: dict, timeout: int = 60) -> requests.Response:
    s = requests.Session()
    s.trust_env = False
    return s.post(url, json=data, headers=headers, timeout=timeout, verify=_get_ca_bundle_path())


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
    resp = post_json(YANDEXGPT_URL, data, headers, 90)
    if resp.status_code != 200:
        return f"Ошибка API ({resp.status_code}): {resp.text}"
    return resp.json()['result']['alternatives'][0]['message']['text']


def _mine_rules(text: str, per_chunk_chars: int = 4000, partial_limit: int = 600, final_limit: int = 1200) -> str:
    chunks = (_split_text(text, per_chunk_chars) or [''])[:3]
    partials: list[str] = []
    for ch in chunks:
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


class AnalyzeBody(BaseModel):
    context: str
    description: str | None = None
    checklist: str | None = None
    mode: str | None = None


class MultipassBody(BaseModel):
    batches: list[str]
    description: str | None = None
    checklist: str | None = None

class SummarizeBody(BaseModel):
    report: str


SYSTEM_PROMPT = '''
Ты проводишь строгий код-ревью. Требования к формату вывода:

### Отчёт по результатам ревью кода

Общие замечания по коду
- Краткий список проблем по коду в целом.
- Для каждой проблемы приведи 1–3 ссылки вида file:line (репрезентативные примеры) и краткую причину. Если проблема повторяется по всему файлу — так и отметь и всё равно приведи 1–3 примера.
- Если нет существенных замечаний — напиши: «Нет существенных замечаний».

Замечания по чеклисту
- Используй только предоставленный чеклист/описание (или их сжатые правила). Ничего не выдумывай.
- Для каждого нарушенного/частично выполненного требования процитируй текст правила (человеческое название), без выдуманной нумерации.
- Затем укажи под-пункты:
  • Статус: нарушено | частично выполнено
  • Файлы/строки: перечисли 1–3 ссылки file:line (репрезентативные примеры). Если нарушение встречается массово — отметь это явно и приведи примеры.
  • Обоснование: что именно не соответствует требованию
  • Рекомендации: пошаговый план исправления без приведения полного кода
- Если чеклист отсутствует — напиши: «Чеклист не предоставлен, проверены только общие замечания».

Саммари рекомендаций
- Краткий перечень ключевых шагов по исправлению, без повторов.

Общие правила:
- Всегда указывай ссылки вида file:line там, где это уместно.
- Для сложных случаев давай рекомендации по шагам (а не готовый код).
- Не придумывай отсутствующие пункты чеклиста и не выдумывай нумерацию.
'''


def analyze_code(context: str, description: str | None, checklist: str | None, simple: bool = False) -> str:
    prompt = SYSTEM_PROMPT + '\n\n'
    rules = _build_rules(description, checklist)
    only_general = not bool(rules)
    if only_general:
        prompt += 'ВНИМАНИЕ: чеклист/описание отсутствуют — выведи только разделы 1 и 3; раздел 2 не выводи.\n\n'
    if rules:
        prompt += f'Свод правил для проверки (сжато):\n{rules}\n\n'
    if not simple:
        prompt += f'Контекст кода (Graph RAG):\n```\n{_clip(context, 12000)}\n```'
        prompt = _clip_tokens(prompt, MAX_TOKENS)
        return call_yandex('yandexgpt-lite', prompt, temperature=0.2, max_tokens=2000)
    prompt = _clip_tokens(prompt, MAX_TOKENS)
    return call_yandex('yandexgpt-lite', prompt, temperature=0.2, max_tokens=1500)


def summarize_for_reviewer(student_report: str) -> str:
    only_general = 'Замечания по чеклисту' not in (student_report or '')
    if only_general:
        prompt = (
            'Сделай краткий конспект для ревьюера ТОЛЬКО из разделов «Общие замечания по коду» и «Саммари рекомендаций» (без раздела «Замечания по чеклисту»), без лишних подробностей и без повторов кода.\n\n'
            '### Отчёт по результатам ревью кода\n\n'
            'Общие замечания по коду — 2-6 пунктов, с file:line.\n'
            'Саммари рекомендаций — 3-8 ключевых шагов.\n\n' + _clip_tokens(student_report, MAX_TOKENS)
        )
    else:
        prompt = (
            'На основе отчёта ниже сделай краткий конспект для ревьюера в той же структуре (3 раздела), без лишних подробностей и без повторов кода.\n\n'
            '### Отчёт по результатам ревью кода\n\n'
            'Общие замечания по коду — 2-6 пунктов, с file:line.\n'
            'Замечания по чеклисту — только нарушенные/частично выполненные, с цитатой правила, file:line и краткой рекомендацией.\n'
            'Саммари рекомендаций — 3-8 ключевых шагов.\n\n' + _clip_tokens(student_report, MAX_TOKENS)
        )
    return call_yandex('yandexgpt-lite', prompt, temperature=0.1, max_tokens=800)


def analyze_multipass(batches: list[str], description: str | None, checklist: str | None) -> str:
    summaries: list[str] = []
    rules_full = _build_rules(description, checklist)
    rule_chunks = _split_rules_text(rules_full, max_lines=14, max_chars=800) or ['']
    for ctx in batches:
        for rule_chunk in rule_chunks:
            prompt = SYSTEM_PROMPT + '\n\n'
            if not rules_full:
                prompt += 'ВНИМАНИЕ: чеклист/описание отсутствуют — дай краткие общие замечания по коду (без раздела чеклиста).\n\n'
            if rule_chunk:
                prompt += f'Правила для проверки:\n{rule_chunk}\n\n'
            prompt += f'Контекст кода:\n```\n{_clip(ctx, 6500)}\n```\n'
            prompt += 'Дай краткое резюме замечаний.'
            prompt = _clip_tokens(prompt, MAX_TOKENS)
            out = call_yandex('yandexgpt-lite', prompt, temperature=0.2, max_tokens=700)
            summaries.append(_clip(out, 8000))
    final_prompt = SYSTEM_PROMPT + '\n\n'
    if rules_full:
        final_prompt += f'Свод правил (для справки):\n{_clip(rules_full, 1200)}\n\n'
    joined = '\n\n'.join(summaries)
    if len(joined) > 90000:
        joined = joined[:90000]
    if not rules_full:
        final_prompt += (
            'Ниже частичные отчёты. Объедини их и выдай ТОЛЬКО разделы 1 и 3 (без раздела 2), как в требованиях выше. '
            'Не дублируй одно и то же; обязательно приводи file:line.\n\n' + joined
        )
    else:
        final_prompt += (
            'Ниже частичные отчёты. Объедини их и выдай ровно в 3 раздела как в требованиях выше. '
            'Не дублируй одно и то же, не выдумывай пункты чеклиста, обязательно приводи file:line, '
            'в разделе чеклиста — цитируй формулировку правила и дай рекомендации по шагам.\n\n' + joined
        )
    final_prompt = _clip_tokens(final_prompt, MAX_TOKENS)
    return call_yandex('yandexgpt', final_prompt, temperature=0.2, max_tokens=1600)


app = FastAPI()


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/analyze')
def api_analyze(body: AnalyzeBody):
    simple = (body.mode or '').lower() == 'simple'
    report = analyze_code(body.context or '', body.description, body.checklist, simple)
    reviewer = summarize_for_reviewer(report)
    return {'report': report, 'reviewer': reviewer}


@app.post('/multipass')
def api_multipass(body: MultipassBody):
    report = analyze_multipass(body.batches or [], body.description, body.checklist)
    reviewer = summarize_for_reviewer(report)
    return {'report': report, 'reviewer': reviewer}


@app.post('/summarize')
def api_summarize(body: SummarizeBody):
    reviewer = summarize_for_reviewer(body.report or '')
    return {'reviewer': reviewer}


