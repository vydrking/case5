import os
import asyncio
import tempfile
from html.parser import HTMLParser
from telegram import Update
from telegram.ext import ContextTypes

from botapp.graph_rag import prepare_graph_rag_context, prepare_graph_rag_batches
from botapp.archive import extract_any_archive
from botapp.review import summarize_for_reviewer


SESSIONS: dict[int, dict] = {}

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
    SESSIONS.setdefault(chat_id, {'description': None, 'checklist': None, 'multi_pass': True, 'gen': 0})

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    sess = SESSIONS.setdefault(chat_id, {'description': None, 'checklist': None, 'multi_pass': True, 'gen': 0})
    sess['gen'] = (sess.get('gen') or 0) + 1
    await update.message.reply_text('Текущая операция отменена.')

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

async def handle_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text or update.message.caption
    chat_id = update.effective_chat.id
    sess = SESSIONS.setdefault(chat_id, {'description': None, 'checklist': None, 'multi_pass': True, 'gen': 0})
    gen = sess.get('gen') or 0
    def _is_cancelled() -> bool:
        return (SESSIONS.get(chat_id) or {}).get('gen') != gen
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
            if is_archive:
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
                        report = analyze_multipass(batches, sess.get('description'), sess.get('checklist'), _is_cancelled)
                        await update.message.reply_text(report[:4096])
                        try:
                            reviewer = summarize_for_reviewer(report)
                            await update.message.reply_text('Отчёт для ревьюера:\n' + reviewer[:4096])
                        except Exception:
                            pass
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
                    report = analyze_code(code_text, sess.get('description'), sess.get('checklist'))
                    if 'Ошибка' in report or len(report) < 50:
                        report = analyze_code_simple(code_text, sess.get('description'), sess.get('checklist'))
                    await update.message.reply_text(report[:4096])
                    try:
                        reviewer = summarize_for_reviewer(report)
                        await update.message.reply_text('Отчёт для ревьюера:\n' + reviewer[:4096])
                    except Exception:
                        pass
                except Exception as e:
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
                    report = analyze_multipass(batches, sess.get('description'), sess.get('checklist'), _is_cancelled)
                    await update.message.reply_text(report[:4096])
                    try:
                        reviewer = summarize_for_reviewer(report)
                        await update.message.reply_text('Отчёт для ревьюера:\n' + reviewer[:4096])
                    except Exception:
                        pass
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
                try:
                    reviewer = summarize_for_reviewer(report)
                    await update.message.reply_text('Отчёт для ревьюера:\n' + reviewer[:4096])
                except Exception:
                    pass
            except Exception as e:
                await update.message.reply_text(f'Ошибка при обработке GitHub: {str(e)}')
    else:
        await update.message.reply_text('Отправь архив, ссылку на GitHub или текстовый файл описания (.txt/.md).')


