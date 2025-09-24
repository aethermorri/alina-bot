#!/usr/bin/env python3
"""
Telegram бот «Алина» на вебхуках (FastAPI + python-telegram-bot) и OpenAI Responses API.
Готов для деплоя на Render/Railway/любом ASGI-хостинге.

⚙️ Переменные окружения (в облаке):
  TELEGRAM_BOT_TOKEN=...           # токен от @BotFather (НЕ палить)
  OPENAI_API_KEY=...               # ключ OpenAI
  OPENAI_MODEL=gpt-4.1-mini        # или gpt-5 / gpt-5-mini
  SYSTEM_PROMPT=...                # инструкции твоей «Алины» (в 1 строку или с \n)
  OPENAI_TEMPERATURE=0.7           # (опц.) креативность
  OPENAI_MAX_TOKENS=800            # (опц.) предел ответа
  PUBLIC_URL=https://<твой-сервис>.onrender.com   # URL после первого деплоя (для авто setWebhook)
  WEBHOOK_PATH=<секретный-путь>    # (опц.) секретный путь, по умолчанию = TELEGRAM_BOT_TOKEN

Запуск на Render:
  * Build Command:     pip install -r requirements.txt
  * Start Command:     uvicorn main:app --host 0.0.0.0 --port $PORT

Маршруты:
  POST /<WEBHOOK_PATH>  — приём апдейтов от Telegram
  GET  /healthz         — проверка живости

Команды:
  /start — приветствие
  /reset — очистить контекст диалога для текущего пользователя

Подсказки:
  * Бот хранит последние 20 сообщений контекста в памяти процесса (per-user).
  * Для продакшена добавь постоянное хранилище (SQLite/Redis) — тут не обязательно.
"""
import os
import logging
from typing import Dict, List, TypedDict
from contextlib import asynccontextmanager
from http import HTTPStatus

from fastapi import FastAPI, Request, Response
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from openai import AsyncOpenAI
# Клиент OpenRouter
client = AsyncOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api"),
    api_key=os.getenv("OPENAI_API_KEY"),
    default_headers={
        "HTTP-Referer": os.getenv("PUBLIC_URL", "https://example.com"),  # твой сайт/бот
        "X-Title": "Alina Telegram Bot",  # любое название проекта
    },
)


# === Конфиг через ENV ===
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Ты дружелюбный, краткий и полезный ассистент. Отвечай по-русски и на ты, если пользователь не просит иначе."
)
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "800"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PUBLIC_URL = os.getenv("PUBLIC_URL")  # напр. https://alina-bot.onrender.com
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", TELEGRAM_BOT_TOKEN or "webhook")  # секретный путь

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

# === OpenAI клиент (асинхронный) ===
client = AsyncOpenAI()

# === Память диалогов в RAM ===
class Msg(TypedDict):
    role: str  # "user" | "assistant"
    content: str

# user_id -> list[Msg]
user_histories: Dict[int, List[Msg]] = {}
MAX_TG_MSG = 4096

# === Создаём PTB Application без встроенного апдейтера (мы принимаем вебхуком) ===
ptb = Application.builder().token(TELEGRAM_BOT_TOKEN).updater(None).build()

# === Хэндлеры ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        ("Привет! Я бот на базе твоей ‘Алины’. Пиши сообщение — отвечу.\n"
         "Команды: /reset — очистить контекст.")
    )

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user:
        user_histories.pop(update.effective_user.id, None)
    await update.message.reply_text("Окей, я всё забыл про этот диалог.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return

    user_id = update.effective_user.id
    text = (update.message.text or "").strip()
    if not text:
        return

    # Показываем "typing…"
    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    except Exception:
        pass

    history = user_histories.get(user_id, [])

    # Собираем сообщения для Responses API
    input_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history[-10:],  # берём последние 10 реплик
        {"role": "user", "content": text},
    ]

    try:
        resp = await client.responses.create(
            model=OPENAI_MODEL,
            input=input_messages,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        answer = resp.output_text
    except Exception:
        logging.exception("OpenAI API error")
        await update.message.reply_text(
            "Упс. Не получилось сходить к модели. Проверь OPENAI_API_KEY/модель."
        )
        return

    # Обновляем память диалога
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": answer})
    user_histories[user_id] = history[-20:]

    # Рубим длинные ответы под лимит Telegram
    for i in range(0, len(answer), MAX_TG_MSG):
        chunk = answer[i : i + MAX_TG_MSG]
        await update.message.reply_text(chunk, disable_web_page_preview=True)

# Регистрируем хэндлеры
ptb.add_handler(CommandHandler("start", start))
ptb.add_handler(CommandHandler("reset", reset))
ptb.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

# === FastAPI-приложение ===
app = FastAPI()

@app.get("/healthz")
async def healthz() -> Response:
    return Response(content="ok", media_type="text/plain")

@app.post(f"/{WEBHOOK_PATH}")
async def telegram_webhook(request: Request) -> Response:
    """Получаем апдейты Telegram и прокидываем их в PTB."""
    data = await request.json()
    update = Update.de_json(data, ptb.bot)
    # Можно либо класть в очередь, либо напрямую обрабатывать:
    # await ptb.update_queue.put(update)
    await ptb.process_update(update)
    return Response(status_code=HTTPStatus.OK)

# === Жизненный цикл: старт/стоп PTB и установка вебхука ===
@asynccontextmanager
async def lifespan(_: FastAPI):
    # Устанавливаем вебхук, если указан PUBLIC_URL
    if PUBLIC_URL:
        try:
            await ptb.bot.set_webhook(
                url=f"{PUBLIC_URL}/{WEBHOOK_PATH}",
                allowed_updates=Update.ALL_TYPES,
            )
            logging.info("Webhook set to %s/%s", PUBLIC_URL, WEBHOOK_PATH)
        except Exception:
            logging.exception("Failed to set webhook")
    # Запускаем PTB
    async with ptb:
        await ptb.start()
        yield
        await ptb.stop()

# Подключаем lifespan к приложению FastAPI
app.router.lifespan_context = lifespan

