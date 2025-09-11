import os
import tempfile
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
# import git
from patoolib import extract_archive
from yandex_cloud_ml_sdk import YCloudML

load_dotenv()

# Конфигурация
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
YANDEX_FOLDER_ID = os.getenv('YANDEX_FOLDER_ID')
YANDEX_API_KEY = os.getenv('YANDEX_API_KEY')

# Обработчик команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Отправь мне ZIP-архив с кодом или ссылку на GitHub-репозиторий. "
        "Можно добавить чеклист (текстом)."
    )

# Загрузка кода из GitHub
# def clone_github_repo(url, path):
    # git.Repo.clone_from(url, path)

# Анализ кода через YandexGPT с использованием Yandex Cloud ML SDK
def analyze_code(code_text, checklist=None):
    try:
        # Инициализируем SDK
        sdk = YCloudML(
            folder_id=YANDEX_FOLDER_ID,
            auth=YANDEX_API_KEY
        )
        
        # Получаем модель
        model = sdk.models.completions("yandexgpt", model_version="latest")
        model = model.configure(
            temperature=0.1,
            maxTokens=4000
        )
        
        # Формируем промпт
        system_prompt = (
            "Ты - опытный программист-аналитик. Проанализируй предоставленный код на наличие ошибок, "
            "включая синтаксические ошибки, логические ошибки, проблемы со стилем кода и потенциальные уязвимости безопасности. "
            "Не исправляй код, а только укажи на проблемы и объясни, как их можно решить."
        )
        
        user_prompt = f"Проанализируй следующий код:\n\n{code_text}"
        
        if checklist:
            user_prompt += f"\n\nДополнительный чеклист для проверки: {checklist}"
        
        # Выполняем запрос
        result = model.run(
            [
                {"role": "system", "text": system_prompt},
                {"role": "user", "text": user_prompt}
            ]
        )
        
        # Обрабатываем результат
        response_text = ""
        for alternative in result:
            response_text += str(alternative) + "\n\n"
        
        return response_text
        
    except Exception as e:
        error_msg = f"Ошибка при анализе кода: {str(e)}"
        print(error_msg)
        return error_msg

# Альтернативная версия с yandexgpt-lite (если основная не работает)
def analyze_code_lite(code_text, checklist=None):
    try:
        sdk = YCloudML(
            folder_id=YANDEX_FOLDER_ID,
            auth=YANDEX_API_KEY
        )
        
        model = sdk.models.completions("yandexgpt-lite", model_version="latest")
        model = model.configure(
            temperature=0.1,
            maxTokens=2000
        )
        
        prompt = f"Проанализируй код на ошибки (синтаксис, логика, стиль, безопасность):\n\n{code_text}"
        
        if checklist:
            prompt += f"\n\nЧеклист: {checklist}"
        
        result = model.run([{"role": "user", "text": prompt}])
        
        response_text = ""
        for alternative in result:
            response_text += str(alternative) + "\n\n"
        
        return response_text
        
    except Exception as e:
        return f"Ошибка анализа (lite): {str(e)}"

# Обработчик входящих сообщений
async def handle_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text or update.message.caption
    checklist = None

    # Если есть документ (ZIP-архив)
    if update.message.document:
        file = await update.message.document.get_file()
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = os.path.join(tmp_dir, "project.zip")
            await file.download_to_drive(zip_path)
            extract_path = os.path.join(tmp_dir, "extracted")
            os.makedirs(extract_path)
            try:
                extract_archive(zip_path, outdir=extract_path)
                code_text = read_code_from_directory(extract_path)
                if user_input and "чеклист" in user_input.lower():
                    checklist = user_input
                
                # Пробуем основную модель
                report = analyze_code(code_text, checklist)
                if "Ошибка" in report:
                    # Пробуем lite версию
                    report = analyze_code_lite(code_text, checklist)
                
                await update.message.reply_text(report[:4096])
            except Exception as e:
                await update.message.reply_text(f"Ошибка при обработке архива: {str(e)}")

    # Если это ссылка на GitHub
    elif user_input and "github.com" in user_input:
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                repo_url = user_input.split()[0]
                clone_github_repo(repo_url, tmp_dir)
                code_text = read_code_from_directory(tmp_dir)
                if len(user_input.split()) > 1:
                    checklist = " ".join(user_input.split()[1:])
                
                report = analyze_code(code_text, checklist)
                if "Ошибка" in report:
                    report = analyze_code_lite(code_text, checklist)
                
                await update.message.reply_text(report[:4096])
            except Exception as e:
                await update.message.reply_text(f"Ошибка при обработке GitHub репозитория: {str(e)}")

    else:
        await update.message.reply_text("Отправь архив или ссылку на GitHub.")

# Чтение кода из директории
def read_code_from_directory(directory):
    code_text = ""
    supported_extensions = (".py", ".js", ".java", ".cpp", ".c", ".html", ".css")
    
    file_count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(supported_extensions) and file_count < 5:  # Ограничиваем количество файлов
                try:
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        code_text += f"=== Файл: {file} ===\n{content}\n\n"
                        file_count += 1
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"Ошибка при чтении файла {file}: {str(e)}")
    
    if not code_text:
        return "Не найдено файлов с кодом для анализа."
    
    # Ограничиваем размер текста
    if len(code_text) > 6000:
        code_text = code_text[:6000] + "\n\n... [код обрезан из-за ограничения длины]"
    
    return code_text

# Главная функция
def main():
    # Проверяем наличие необходимых переменных окружения
    required_vars = ['TELEGRAM_BOT_TOKEN', 'YANDEX_FOLDER_ID', 'YANDEX_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Ошибка: Отсутствуют переменные окружения: {', '.join(missing_vars)}")
        return
    
    # Проверяем доступность SDK
    try:
        # Тестовый запрос для проверки подключения
        sdk = YCloudML(
            folder_id=YANDEX_FOLDER_ID,
            auth=YANDEX_API_KEY
        )
        print("SDK успешно инициализирован")
    except Exception as e:
        print(f"Ошибка инициализации SDK: {e}")
        return
    
    print("Запуск бота...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT | filters.Document.ALL, handle_input))
    app.run_polling()

if __name__ == "__main__":
    main()