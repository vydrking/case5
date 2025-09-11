import os
from dotenv import load_dotenv

def _clean_env(val: str | None) -> str | None:
    if not isinstance(val, str):
        return None
    v = val.strip()
    if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
        v = v[1:-1]
    return v or None

def load_env() -> None:
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', 'main.py').replace('main.py','') + '.env', override=True)

TELEGRAM_TOKEN = _clean_env(os.getenv('TELEGRAM_BOT_TOKEN'))
YANDEX_FOLDER_ID = _clean_env(os.getenv('YANDEX_FOLDER_ID'))
YANDEX_API_KEY = _clean_env(os.getenv('YANDEX_API_KEY'))
YANDEXGPT_URL = 'https://llm.api.cloud.yandex.net/foundationModels/v1/completion'

REQUIRED_VARS = ['TELEGRAM_BOT_TOKEN', 'YANDEX_FOLDER_ID', 'YANDEX_API_KEY']


