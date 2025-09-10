import os
import json
import asyncio
import requests
from typing import List, Optional
from pydantic import Field
from pydantic.config import ConfigDict

# Современная структура пакета (проверено в окружении):
from llama_index.core.embeddings import BaseEmbedding


class YandexEmbedding(BaseEmbedding):
    """Адаптер эмбеддингов Yandex Cloud для LlamaIndex.

    Требуются переменные окружения:
    - YANDEX_API_KEY
    - YANDEX_FOLDER_ID
    Опционально:
    - YANDEX_EMB_ENDPOINT (по умолчанию foundationModels/v1/embeddings)
    - YANDEX_EMB_MODEL (по умолчанию yandexgpt-embedding-lite)
    """

    # Разрешаем поля pydantic и задаём дефолты через env
    model_config = ConfigDict(extra="allow")

    # поля базовой модели
    model_name: str = Field(default="yandexgpt-embedding-lite")

    # собственные поля
    api_key: Optional[str] = Field(default=None, exclude=True)
    folder_id: Optional[str] = Field(default=None)
    endpoint: str = Field(default="https://llm.api.cloud.yandex.net/foundationModels/v1/embeddings")

    def __init__(self, api_key: Optional[str] = None, folder_id: Optional[str] = None, model: Optional[str] = None, endpoint: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("YANDEX_API_KEY")
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")
        self.model_name = model or os.getenv("YANDEX_EMB_MODEL", "yandexgpt-embedding-lite")
        self.endpoint = endpoint or os.getenv("YANDEX_EMB_ENDPOINT", self.endpoint)
        if not (self.api_key and self.folder_id):
            raise RuntimeError("YANDEX_API_KEY и/или YANDEX_FOLDER_ID не заданы")

    def _request(self, texts: List[str]) -> List[List[float]]:
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"modelUri": f"emb://{self.folder_id}/{self.model_name}", "texts": texts}
        r = requests.post(self.endpoint, headers=headers, data=json.dumps(payload), timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"Yandex embeddings HTTP {r.status_code}: {r.text}")
        data = r.json()
        embs: List[List[float]] = []
        for item in data.get("embeddings", []):
            embs.append(item.get("vector", []))
        if len(embs) != len(texts):
            # на всякий случай выравниваем
            while len(embs) < len(texts):
                embs.append([0.0])
        return embs

    # LlamaIndex API
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._request([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._request([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._request(texts)



    # Async API, требуемая BaseEmbedding в свежих версиях LlamaIndex
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await asyncio.to_thread(self._get_text_embedding, text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self._get_text_embeddings, texts)

