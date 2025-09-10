# AutoReview FastAPI Service

## Запуск локально

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Переменные окружения

- YANDEX_API_KEY, YANDEX_FOLDER_ID — для онлайн- режима YandexGPT (иначе оффлайн-ответы).
- LOG_LEVEL — уровень логирования (INFO по умолчанию).

## Эндпоинты

- GET /api/health — проверка состояния
- POST /api/review/run — загрузка desc.html, checklist.html, project.zip (multipart/form-data). Возвращает JSON с обзором, замечаниями и оценкой.

## Структура

- app/main.py — инициализация приложения
- app/api/routes — роутеры
- app/services — доменная логика и интеграция с case5/autoreview
- app/schemas — модели ответов
- app/core — конфиг/логирование

## Основано на документации

- FastAPI: https://fastapi.tiangolo.com/
- MDN Web APIs: https://developer.mozilla.org/
- LangGraph: https://langchain-ai.github.io/langgraph/
- BeautifulSoup/lxml: https://www.crummy.com/software/BeautifulSoup/bs4/doc/, https://lxml.de/
