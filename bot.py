import asyncio
import hashlib
import logging
import os
import re
import time
from collections import defaultdict, deque

import httpx
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
# Stop httpx from printing request URLs (they contain the bot token)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)

# ---------- Config (all from environment variables) ----------
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
CHAT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
THINK_MODEL = os.getenv("DEEPSEEK_THINK_MODEL", "deepseek-reasoner")
PORT = int(os.getenv("PORT", "10000"))
BASE_URL = (os.getenv("WEBHOOK_URL") or os.getenv("RENDER_EXTERNAL_URL", "")).rstrip("/")
HISTORY_LEN = int(os.getenv("HISTORY_LEN", "20"))  # messages kept per user
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "8"))  # requests per minute per user
PING_INTERVAL = int(os.getenv("PING_INTERVAL", "600"))  # seconds; must be < 15 min
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
    base_url=os.getenv("AI_BASE_URL", "https://api.deepseek.com"),
    timeout=120,
    max_retries=2,
)

# ---------- In-memory state (resets on restart/sleep) ----------
history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
think_mode = defaultdict(lambda: os.getenv("THINK_DEFAULT") == "1")
locks = defaultdict(asyncio.Lock)
hits = defaultdict(deque)

# ---------- Persistent memory (optional: set DATABASE_URL to a Postgres URL) ----------
DATABASE_URL = os.getenv("DATABASE_URL", "")
KEEP_ROWS = 100  # messages stored per chat; only the last HISTORY_LEN are sent to the model
pool = None  # asyncpg pool, created in post_init when DATABASE_URL is set


async def init_db():
    global pool
    if not DATABASE_URL:
        log.info("No DATABASE_URL: conversation memory is in RAM only (lost on restart).")
        return
    import asyncpg  # imported here so the bot still runs without the package

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=3)
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS cat_bot_history (
            id BIGSERIAL PRIMARY KEY,
            chat_key BIGINT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL
        )
        """
    )
    await pool.execute(
        "CREATE INDEX IF NOT EXISTS cat_bot_history_key ON cat_bot_history (chat_key, id)"
    )
    log.info("Persistent memory enabled (Postgres).")


async def get_history(key: int) -> list:
    if pool is None:
        return list(history[key])
    rows = await pool.fetch(
        "SELECT role, content FROM ("
        "  SELECT id, role, content FROM cat_bot_history"
        "  WHERE chat_key = $1 ORDER BY id DESC LIMIT $2"
        ") t ORDER BY id",
        key,
        HISTORY_LEN,
    )
    return [{"role": r["role"], "content": r["content"]} for r in rows]


async def add_messages(key: int, user_text: str, answer: str):
    if pool is None:
        history[key].append({"role": "user", "content": user_text})
        history[key].append({"role": "assistant", "content": answer})
        return
    await pool.execute(
        "INSERT INTO cat_bot_history (chat_key, role, content) "
        "VALUES ($1, 'user', $2), ($1, 'assistant', $3)",
        key,
        user_text,
        answer,
    )
    # Trim old rows so the free database doesn't fill up
    await pool.execute(
        "DELETE FROM cat_bot_history WHERE chat_key = $1 AND id NOT IN "
        "(SELECT id FROM cat_bot_history WHERE chat_key = $1 ORDER BY id DESC LIMIT $2)",
        key,
        KEEP_ROWS,
    )


async def clear_history(key: int):
    history[key].clear()
    if pool is not None:
        await pool.execute("DELETE FROM cat_bot_history WHERE chat_key = $1", key)


def rate_limited(uid: int) -> bool:
    now = time.monotonic()
    q = hits[uid]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return True
    q.append(now)
    return False


def clean_answer(text: str) -> str:
    """Remove the <think>...</think> reasoning block some models put in their reply."""
    return re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.S).strip()


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
    chat = update.effective_chat
    key = chat.id if chat.type in ("group", "supergroup") else update.effective_user.id
    await clear_history(key)
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
    text = msg.text
    key = uid

    if msg.chat.type in ("group", "supergroup"):
        # In groups, answer messages that start with "هوی" or that reply to the bot
        replied = msg.reply_to_message
        is_reply_to_bot = (
            replied is not None
            and replied.from_user is not None
            and replied.from_user.id == ctx.bot.id
        )
        if not (is_reply_to_bot or re.match(r"^هوی(?!\w)", text)):
            return
        text = re.sub(r"^هوی[\s،,:!]*", "", text) or "سلام"
        text = f"{msg.from_user.first_name}: {text}"
        key = msg.chat.id  # one shared conversation per group

    if ALLOWED_IDS and uid not in ALLOWED_IDS:
        return
    if rate_limited(uid):
        await msg.reply_text("⏳ خیلی سریع پیام می‌دی، چند ثانیه صبر کن.")
        return

    async with locks[key]:
        use_think = think_mode[uid]

        stop = asyncio.Event()
        typer = asyncio.create_task(keep_typing(msg.chat, stop))
        try:
            hist = await get_history(key)
            resp = await ai.chat.completions.create(
                model=THINK_MODEL if use_think else CHAT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *hist,
                    {"role": "user", "content": text},
                ],
                max_tokens=4000 if use_think else 1500,
            )
            answer = clean_answer(resp.choices[0].message.content or "") or "…"
            try:
                await add_messages(key, text, answer)
            except Exception:
                log.exception("Could not save history")
        except Exception:
            log.exception("DeepSeek request failed")
            answer = "⚠️ یه مشکلی پیش اومد، چند لحظه‌ی دیگه دوباره امتحان کن."
        finally:
            stop.set()
            await typer

    for chunk in split_text(answer):
        await msg.reply_text(chunk)


async def on_error(update, ctx: ContextTypes.DEFAULT_TYPE):
    log.error("Unhandled error", exc_info=ctx.error)


async def keep_alive():
    """Ping our own public URL so Render's free tier doesn't put the service to sleep."""
    await asyncio.sleep(60)
    async with httpx.AsyncClient(timeout=20) as client:
        while True:
            try:
                await client.get(BASE_URL + "/")  # a 404 is fine, it still counts as traffic
            except Exception as e:
                log.warning("keep-alive ping failed: %s", e)
            await asyncio.sleep(PING_INTERVAL)


async def post_init(app: Application):
    app.bot_data["keep_alive"] = asyncio.create_task(keep_alive())
    try:
        await init_db()
    except Exception:
        global pool
        pool = None
        log.exception("Database init failed, falling back to RAM-only memory")
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
