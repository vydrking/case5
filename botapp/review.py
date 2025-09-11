from botapp.prompts import SYSTEM_PROMPT, REVIEWER_PROMPT
from botapp.config import YANDEX_API_KEY, YANDEX_FOLDER_ID, YANDEXGPT_URL
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


