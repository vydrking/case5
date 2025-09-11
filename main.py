import os
import tempfile
import requests
import subprocess
import certifi
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
from graph_rag import prepare_graph_rag_context
from graph_rag import prepare_graph_rag_batches
from patoolib import extract_archive
import zipfile
import tarfile
import json

load_dotenv()

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
    return val.strip() if isinstance(val, str) else None

# Конфигурация
TELEGRAM_TOKEN = _clean_env(os.getenv('TELEGRAM_BOT_TOKEN'))
YANDEX_FOLDER_ID = _clean_env(os.getenv('YANDEX_FOLDER_ID'))
YANDEX_API_KEY = _clean_env(os.getenv('YANDEX_API_KEY'))
YANDEXGPT_URL = 'https://llm.api.cloud.yandex.net/foundationModels/v1/completion'
SYSTEM_PROMPT = '''Ты — опытный ревьюер кода. Проанализируй проект на ошибки (синтаксис, логика, безопасность), стиль и архитектуру, а также соответствие описанию проекта и чеклисту требований. Не исправляй код — только укажи проблемы и объясни, как их устранить. Дай чёткие ссылки на файлы/фрагменты из контекста.

Сформируй ответ строго в разделах:
### Общие замечания по коду
- перечисли найденные ошибки, анти-паттерны, проблемы стиля, безопасности и архитектуры

### Замечания по чеклисту (если чеклист есть)
- сопоставь пункты чеклиста с проектом: что выполнено/не выполнено, где несоответствия

### Замечания по описанию проекта (если описание есть)
- проверь соответствие целям и ограничениям описания: что реализовано, что отсутствует или реализовано иначе

### Резюме
- приоритизированный список конкретных действий. Для каждого шага укажи: короткий план исправления (что поменять и где), почему это важно, на что обратить внимание (граничные случаи, риски), и как быстро проверить результат (команда/тест/чек). Избегай полного кода; давай точечные подсказки: имена файлов/функций, возможные сигнатуры, правила линтера, команды запуска тестов.'''

SESSIONS: dict[int, dict] = {}

# Обработчик команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Приветствие и инициализация сессии чата (multi-pass по умолчанию)
    await update.message.reply_text(
        "Привет! Отправь мне ZIP-архив с кодом или ссылку на GitHub-репозиторий. "
        "Можно добавить чеклист (текстом)."
    )
    chat_id = update.effective_chat.id
    SESSIONS.setdefault(chat_id, {'description': None, 'checklist': None, 'multi_pass': True})

async def rag_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Краткий статус RAG и режима для текущего чата
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
    # Извлекаем текстовую сводку графа из начала контекста
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
    # Преобразуем сводку графа в Mermaid-диаграмму (graph LR)
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
    for n in list(nodes)[:200]:
        lines.append(f"{sid(n)}[\"{label(n)}\"]")
    for s,t,r in edges[:400]:
        arrow = '-->' if r != 'imports' else '---'
        lines.append(f"{sid(s)} {arrow} {sid(t)}:::rel")
    lines.append('classDef rel stroke:#888,stroke-width:1px,color:#444')
    return '\n'.join(lines)

async def show_graph(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Отправляем пользователю .mmd файл для отображения графа в Mermaid
    chat_id = update.effective_chat.id
    sess = SESSIONS.get(chat_id) or {}
    summary = sess.get('last_graph_summary') or ''
    if not summary:
        await update.message.reply_text('Граф недоступен. Пришлите архив или ссылку, чтобы я построил граф.')
        return
    mermaid = _summary_to_mermaid(summary)
    import tempfile
    import io
    with tempfile.NamedTemporaryFile('w+', suffix='.mmd', delete=False, encoding='utf-8') as tf:
        tf.write(mermaid)
        tf.flush()
        path = tf.name
    try:
        with open(path, 'rb') as f:
            await update.message.reply_document(f, filename='graph.mmd', caption='Mermaid диаграмма графа. Открой в любом Mermaid-рендерере.')
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

async def mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Переключение режима single/multi для текущего чата
    chat_id = update.effective_chat.id
    sess = SESSIONS.setdefault(chat_id, {'description': None, 'checklist': None, 'multi_pass': True})
    arg = (update.message.text or '').strip().split()
    if len(arg) > 1 and arg[1].lower() in ('multi', 'single'):
        sess['multi_pass'] = (arg[1].lower() == 'multi')
        await update.message.reply_text(f"Режим установлен: {'multi' if sess['multi_pass'] else 'single'}")
    else:
        await update.message.reply_text("Использование: /mode multi|single")

def clone_github_repo(url: str, path: str) -> None:
    # Поверхностное клонирование репозитория
    subprocess.run(['git', 'clone', '--depth', '1', url, path], check=True)


def extract_any_archive(src_path: str, out_dir: str) -> None:
    # Распаковка стандартных архивов средствами stdlib, иначе патчим через patool
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
    extract_archive(src_path, outdir=out_dir)

# Анализ кода через YandexGPT
def analyze_code(code_text: str, description: str | None = None, checklist: str | None = None) -> str:
    # Однопроходный анализ: собираем промпт и вызываем yandexgpt-lite
    try:
        prompt = SYSTEM_PROMPT + '\n\n'
        
        if description:
            prompt += f'Описание проекта:\n{description}\n\n'
        if checklist:
            prompt += f'Чеклист требований:\n{checklist}\n\n'
        prompt += f'Контекст кода (Graph RAG):\n```\n{code_text}\n```'

        headers = {
            'Authorization': f'Api-Key {YANDEX_API_KEY}',
            'Content-Type': 'application/json',
            'x-folder-id': YANDEX_FOLDER_ID
        }
        
        # Правильный формат запроса для YandexGPT
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

# Альтернативная версия с более простым промптом
def analyze_code_simple(code_text: str, description: str | None = None, checklist: str | None = None) -> str:
    # Запасной формат промпта для стабильности
    try:
        prompt = SYSTEM_PROMPT + '\n\n'
        if description:
            prompt += f'Описание проекта:\n{description}\n\n'
        if checklist:
            prompt += f'Чеклист требований:\n{checklist}\n\n'

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

        response = post_json(YANDEXGPT_URL, data, headers, 60)
        
        if response.status_code != 200:
            return f'Ошибка API: {response.status_code}'
        
        result = response.json()
        return result['result']['alternatives'][0]['message']['text']
        
    except Exception as e:
        return f'Ошибка: {str(e)}'

def call_yandex(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    # Унифицированный вызов Yandex GPT с CA и отключением системных прокси
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

def analyze_multipass(batches: list[str], description: str | None, checklist: str | None) -> str:
    # Многопроходный анализ: lite по батчам → итог heavy-агрегация
    summaries: list[str] = []
    for i, ctx in enumerate(batches, 1):
        prompt = SYSTEM_PROMPT + '\n\n'
        if description:
            prompt += f'Описание проекта:\n{description}\n\n'
        if checklist:
            prompt += f'Чеклист требований:\n{checklist}\n\n'
        prompt += f'Контекст кода (Graph RAG, batch {i}/{len(batches)}):\n```\n{ctx}\n```\n'
        prompt += 'Дай краткое резюме замечаний для этого батча. Без повторов кода.'
        out = call_yandex('yandexgpt-lite', prompt, temperature=0.2, max_tokens=800)
        summaries.append(out)
    final_prompt = SYSTEM_PROMPT + '\n\n'
    if description:
        final_prompt += f'Описание проекта:\n{description}\n\n'
    if checklist:
        final_prompt += f'Чеклист требований:\n{checklist}\n\n'
    final_prompt += 'Ниже частичные отчёты по батчам, объедини их, убери повторы, сделай единый отчёт по требуемым разделам:\n' + '\n\n'.join(summaries)
    return call_yandex('yandexgpt', final_prompt, temperature=0.2, max_tokens=1800)

# Обработчик входящих сообщений
async def handle_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Универсальный обработчик: архивы, текст описания/чеклиста, ссылки GitHub
    user_input = update.message.text or update.message.caption
    chat_id = update.effective_chat.id
    sess = SESSIONS.setdefault(chat_id, {'description': None, 'checklist': None, 'multi_pass': True})
    checklist = None

    # Если есть документ (ZIP-архив)
    if update.message.document:
        file = await update.message.document.get_file()
        with tempfile.TemporaryDirectory() as tmp_dir:
            fpath = os.path.join(tmp_dir, update.message.document.file_name or 'file.bin')
            await asyncio.to_thread(download_telegram_file, file, fpath)
            name_lower = fpath.lower()
            is_archive = name_lower.endswith(('.zip', '.tar', '.rar', '.7z', '.tar.gz', '.tgz'))
            is_textdoc = name_lower.endswith(('.txt', '.md', '.rst', '.markdown', '.yaml', '.yml'))
            if is_archive:
                extract_path = os.path.join(tmp_dir, 'extracted')
            os.makedirs(extract_path)
            try:
                    extract_any_archive(fpath, extract_path)
                    if sess.get('multi_pass'):
                        _ctx_once = build_context_from_directory(extract_path, sess.get('checklist'))
                        sess['last_graph_summary'] = _extract_graph_section(_ctx_once)
                        batches = prepare_graph_rag_batches(extract_path, sess.get('checklist'), 3500, 4)
                        sess['rag_used'] = True
                        sess['rag_context_size'] = sum(len(b) for b in batches)
                        report = analyze_multipass(batches, sess.get('description'), sess.get('checklist'))
                        await update.message.reply_text(report[:4096])
                        return
                    code_text = build_context_from_directory(extract_path, None)
                    sess['rag_used'] = True
                    sess['rag_context_size'] = len(code_text)
                    sess['last_graph_summary'] = _extract_graph_section(code_text)
                    if user_input and 'чеклист' in user_input.lower():
                        sess['checklist'] = user_input
                        code_text = build_context_from_directory(extract_path, sess['checklist'])
                        sess['rag_context_size'] = len(code_text)
                        sess['last_graph_summary'] = _extract_graph_section(code_text)
                    report = analyze_code(code_text, sess.get('description'), sess.get('checklist'))
                    if 'Ошибка' in report or len(report) < 50:
                        report = analyze_code_simple(code_text, sess.get('description'), sess.get('checklist'))
                await update.message.reply_text(report[:4096])
                except Exception as e:
                    await update.message.reply_text(f'Ошибка при обработке архива: {str(e)}')
            elif is_textdoc:
                try:
                    with open(fpath, 'r', encoding='utf-8') as tf:
                        text = tf.read()
                    sess['description'] = text
                    if not sess.get('checklist'):
                        sess['checklist'] = text
                    await update.message.reply_text('Описание/чеклист получены. Теперь отправьте архив проекта или ссылку на GitHub.')
            except Exception as e:
                    await update.message.reply_text(f'Не удалось прочитать файл описания: {str(e)}')
            else:
                await update.message.reply_text('Формат файла не поддерживается. Пришлите архив проекта (.zip/.tar) или текстовый файл описания (.txt/.md).')

    # Если это ссылка на GitHub
    elif user_input and 'github.com' in user_input:
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                repo_url = user_input.split()[0]
                clone_github_repo(repo_url, tmp_dir)
                if sess.get('multi_pass'):
                    _ctx_once = build_context_from_directory(tmp_dir, sess.get('checklist'))
                    sess['last_graph_summary'] = _extract_graph_section(_ctx_once)
                    batches = prepare_graph_rag_batches(tmp_dir, sess.get('checklist'), 3500, 4)
                    sess['rag_used'] = True
                    sess['rag_context_size'] = sum(len(b) for b in batches)
                    report = analyze_multipass(batches, sess.get('description'), sess.get('checklist'))
                    await update.message.reply_text(report[:4096])
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
                
                report = analyze_code(code_text, sess.get('description'), sess.get('checklist'))
                if 'Ошибка' in report or len(report) < 50:
                    report = analyze_code_simple(code_text, sess.get('description'), sess.get('checklist'))
                
                await update.message.reply_text(report[:4096])
                
            except Exception as e:
                await update.message.reply_text(f'Ошибка при обработке GitHub: {str(e)}')

    else:
        await update.message.reply_text('Отправь архив, ссылку на GitHub или текстовый файл описания (.txt/.md).')

def build_context_from_directory(directory: str, checklist: str | None) -> str:
    # Упрощённый фасад над RAG для сборки контекста с обрезкой
    ctx = prepare_graph_rag_context(directory, checklist)
    if not ctx or not ctx.strip():
        return 'Не найдено файлов с кодом для анализа.'
    if len(ctx) > 9000:
        ctx = ctx[:9000] + '\n\n... [контекст обрезан]'
    return ctx

# Тестовая функция для проверки подключения
def test_connection():
    # Лёгкий пинг до yandexgpt-lite для раннего выявления сетевых проблем
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
    # TLS-безопасная загрузка файла Telegram через requests Session с certifi
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

# Главная функция
def main():
    # Точка входа: проверяем env, CA, тестируем доступ к Yandex и стартуем polling
    # Проверяем переменные окружения
    required_vars = ['TELEGRAM_BOT_TOKEN', 'YANDEX_FOLDER_ID', 'YANDEX_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Отсутствуют переменные: {', '.join(missing_vars)}")
        return
    
    _assert_ca_bundle()
    
    # Тестируем подключение к YandexGPT
    print("Тестируем подключение к YandexGPT...")
    if not test_connection():
        print("Не удалось подключиться к YandexGPT. Продолжаю запуск бота, проверки будут работать без теста.")
    
    print("Запуск бота...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("rag", rag_status))
    app.add_handler(CommandHandler("graph", show_graph))
    app.add_handler(CommandHandler("mode", mode))
    app.add_handler(MessageHandler(filters.TEXT | filters.Document.ALL, handle_input))
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()