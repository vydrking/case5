import os
import json
import time
import logging
import urllib3

from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class YandexGPTClient:
	def __init__(self, api_key: str = None, folder_id: str = None, model: str = None, endpoint: str = None):
		self.api_key = api_key or os.getenv('YANDEX_API_KEY')
		self.folder_id = folder_id or os.getenv('YANDEX_FOLDER_ID')
		self.model = model or os.getenv('YANDEX_GPT_MODEL', 'yandexgpt-lite')
		self.endpoint = endpoint or os.getenv('YANDEX_GPT_ENDPOINT', 'https://llm.api.cloud.yandex.net/foundationModels/v1/completion')
		self.http = urllib3.PoolManager()

	def _headers(self) -> dict:
		return {
			'Authorization': f'Api-Key {self.api_key}' if self.api_key else '',
			'Content-Type': 'application/json',
		}

	def is_configured(self) -> bool:
		return bool(self.api_key and self.folder_id)

	def complete(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1200) -> str:
		if not self.is_configured():
			return self._offline_reply(prompt)

		payload = {
			'modelUri': f'gpt://{self.folder_id}/{self.model}',
			'completionOptions': {
				'temperature': temperature,
				'maxTokens': max_tokens,
				'stream': False,
			},
			'messages': [
				{'role': 'user', 'text': prompt},
			],
		}

		try:
			resp = self.http.request('POST', self.endpoint, body=json.dumps(payload).encode('utf-8'), headers=self._headers(), timeout=urllib3.Timeout(total=60))
			if resp.status >= 400:
				logger.error('YandexGPT error %s: %s', resp.status, resp.data)
				return self._offline_reply(prompt)
			data = json.loads(resp.data.decode('utf-8'))
			choices = data.get('result', {}).get('alternatives') or []
			if not choices:
				return self._offline_reply(prompt)
			return choices[0].get('message', {}).get('text', '').strip() or self._offline_reply(prompt)
		except Exception as e:
			logger.exception('YandexGPT request failed: %s', e)
			return self._offline_reply(prompt)

	def _offline_reply(self, prompt: str) -> str:
		prefix = '[OFFLINE DUMMY]'
		return f"{prefix} {prompt[:400]}\n\n(summary unavailable; configure YANDEX_API_KEY and YANDEX_FOLDER_ID)"

