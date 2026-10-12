import asyncio
import hashlib
import logging
import os
import time
from collections import defaultdict, deque

from openai import AsyncOpenAI
from telegram import BotCommand, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("bot")

# ---------- Config (all from environment variables) ----------
TELEGRAM_TOKEN = os.environ["8664423242:AAETohXA850ruJwqOWnZytP6zSTGck3YgPw"]
DEEPSEEK_API_KEY = os.environ["sk-a67d2293e03d4ea3a453c40f8fcc9e5f"]
CHAT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
THINK_MODEL = os.getenv("DEEPSEEK_THINK_MODEL", "deepseek-reasoner")
PORT = int(os.getenv("PORT", "10000"))
BASE_URL = (os.getenv("WEBHOOK_URL") or os.getenv("RENDER_EXTERNAL_URL", "")).rstrip("/")
HISTORY_LEN = int(os.getenv("HISTORY_LEN", "12"))  # messages kept per user
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "8"))  # requests per minute per user
ALLOWED_IDS = {
    int(x) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip()
}  # empty = everyone

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful, friendly assistant inside a Telegram chat. "
    "Always reply in the same language the user writes in (Persian or English). "
    "Be clear and concise. Use plain text, no Markdown tables.",
)

# Secret webhook path + header token derived from the bot token (no extra env needed)
WEBHOOK_PATH = hashlib.sha256(TELEGRAM_TOKEN.encode()).hexdigest()[:32]
WEBHOOK_SECRET = hashlib.sha256((TELEGRAM_TOKEN + "s").encode()).hexdigest()[:32]

ai = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    timeout=120,
    max_retries=2,
)

# ---------- In-memory state (resets on restart/sleep) ----------
history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
think_mode = defaultdict(bool)
locks = defaultdict(asyncio.Lock)
hits = defaultdict(deque)


def rate_limited(uid: int) -> bool:
    now = time.monotonic()
    q = hits[uid]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return True
    q.append(now)
    return False


def split_text(text: str, size: int = 4000):
    """Telegram messages max out at 4096 chars; split on newlines when possible."""
    while len(text) > size:
        cut = text.rfind("\n", 0, size)
        cut = cut if cut > size // 2 else size
        yield text[:cut]
        text = text[cut:].lstrip("\n")
    if text:
        yield text


async def keep_typing(chat, stop: asyncio.Event):
    while not stop.is_set():
        try:
            await chat.send_action(ChatAction.TYPING)
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=4)
        except asyncio.TimeoutError:
            pass


# ---------- Handlers ----------
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "سلام! 👋 هر سوالی داری بپرس.\n\n"
        "/reset — پاک کردن حافظه‌ی گفتگو\n"
        "/think — روشن/خاموش کردن حالت تفکر عمیق (کندتر ولی دقیق‌تر)"
    )


async def reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    history[update.effective_user.id].clear()
    await update.message.reply_text("🧹 گفتگو از اول شروع شد.")


async def think(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    think_mode[uid] = not think_mode[uid]
    await update.message.reply_text(
        "🧠 حالت تفکر عمیق روشن شد." if think_mode[uid] else "⚡ حالت سریع فعال شد."
    )


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    uid = update.effective_user.id

    if ALLOWED_IDS and uid not in ALLOWED_IDS:
        return
    if rate_limited(uid):
        await msg.reply_text("⏳ خیلی سریع پیام می‌دی، چند ثانیه صبر کن.")
        return

    async with locks[uid]:
        hist = history[uid]
        hist.append({"role": "user", "content": msg.text})
        use_think = think_mode[uid]

        stop = asyncio.Event()
        typer = asyncio.create_task(keep_typing(msg.chat, stop))
        try:
            resp = await ai.chat.completions.create(
                model=THINK_MODEL if use_think else CHAT_MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, *hist],
                max_tokens=4000 if use_think else 1500,
            )
            answer = (resp.choices[0].message.content or "").strip() or "…"
            hist.append({"role": "assistant", "content": answer})
        except Exception:
            log.exception("DeepSeek request failed")
            hist.pop()  # drop the unanswered user message
            answer = "⚠️ یه مشکلی پیش اومد، چند لحظه‌ی دیگه دوباره امتحان کن."
        finally:
            stop.set()
            await typer

    for chunk in split_text(answer):
        await msg.reply_text(chunk)


async def on_error(update, ctx: ContextTypes.DEFAULT_TYPE):
    log.error("Unhandled error", exc_info=ctx.error)


async def post_init(app: Application):
    await app.bot.set_my_commands(
        [
            BotCommand("start", "شروع"),
            BotCommand("reset", "پاک کردن حافظه گفتگو"),
            BotCommand("think", "تفکر عمیق روشن/خاموش"),
        ]
    )


def main():
    if not BASE_URL:
        raise SystemExit("Set WEBHOOK_URL (or run on Render so RENDER_EXTERNAL_URL exists).")

    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .concurrent_updates(True)  # one slow user must not block others
        .post_init(post_init)
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("think", think))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(on_error)

    app.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=WEBHOOK_PATH,
        webhook_url=f"{BASE_URL}/{WEBHOOK_PATH}",
        secret_token=WEBHOOK_SECRET,
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
