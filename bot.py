import os
import json
import time
import random
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
from instagrapi import Client
from instagrapi.exceptions import (
    ChallengeRequired,
    LoginRequired,
    ClientError,
    ClientThrottledError,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("instabot")

REPLIED_FILE = "replied.json"
SESSION_FILE = "session.json"

# Sirf un messages ko reply karo jo bot start hone ke BAAD aaye
BOT_START_TIME = datetime.now(timezone.utc).timestamp()


def load_json(path: str, default=None):
    if Path(path).exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return default if default is not None else {}


def save_json(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f)


# --------------------------------------------------------------------------- #
# Rate limits
# --------------------------------------------------------------------------- #
MAX_REPLIES_PER_HOUR = 30
THREAD_COOLDOWN_MIN = 0  # 0 = follow-up msgs ka turant reply (fast). Safety MAX_REPLIES se aati hai.


def load_rate_limits():
    data = load_json("ratelimit.json", {"timestamps": [], "threads": {}})
    return data["timestamps"], data["threads"]


def save_rate_limits(timestamps: list, threads: dict):
    save_json("ratelimit.json", {"timestamps": timestamps, "threads": threads})


def can_reply(thread_id: str, timestamps: list, threads: dict) -> bool:
    now = time.time()
    cutoff = now - 3600
    # purge old entries
    timestamps[:] = [t for t in timestamps if t > cutoff]
    if len(timestamps) >= MAX_REPLIES_PER_HOUR:
        return False
    last = threads.get(thread_id, 0)
    if now - last < THREAD_COOLDOWN_MIN * 60:
        return False
    return True


def mark_replied(thread_id: str, timestamps: list, threads: dict):
    now = time.time()
    timestamps.append(now)
    threads[thread_id] = now
    # purge again
    cutoff = now - 3600
    timestamps[:] = [t for t in timestamps if t > cutoff]


# --------------------------------------------------------------------------- #
# Login
# --------------------------------------------------------------------------- #
def code_handler(username, challenge):
    code = input("Enter the verification code sent to your email/phone: ")
    return challenge.code(code)


def login() -> Client:
    username = os.getenv("INSTAGRAM_USERNAME")
    password = os.getenv("INSTAGRAM_PASSWORD")
    if not username or not password:
        raise SystemExit("Set INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD in .env")

    cl = Client()
    cl.delay_range = [1, 2]
    cl.challenge_code_handler = code_handler

    proxy = os.getenv("PROXY", "").strip()
    if proxy:
        cl.set_proxy(proxy)

    sessionid = os.getenv("INSTAGRAM_SESSIONID", "").strip()
    if sessionid:
        log.info("Logging in via sessionid...")
        cl.login_by_sessionid(sessionid)
        cl.dump_settings(SESSION_FILE)
        log.info("Logged in.")
        return cl

    if Path(SESSION_FILE).exists():
        log.info("Loading saved session...")
        try:
            cl.load_settings(SESSION_FILE)
            cl.login(username, password)
            cl.get_timeline_feed()
            log.info("Session valid.")
            return cl
        except LoginRequired:
            log.warning("Session expired, logging in fresh...")
            old = cl.get_settings().get("uuids")
            cl.set_settings({})
            if old:
                cl.set_uuids(old)
        except Exception as e:
            log.warning(f"Session failed ({e}), fresh login...")

    try:
        cl.login(username, password)
        cl.dump_settings(SESSION_FILE)
    except ChallengeRequired:
        cl.dump_settings(SESSION_FILE)
        log.warning("Challenge required. Approve on phone, then re-run.")
        raise
    return cl


# --------------------------------------------------------------------------- #
# AI
# --------------------------------------------------------------------------- #
FAST_MODELS = {
    "openrouter": "google/gemini-2.0-flash-001",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
}


def get_ai_response(history: list) -> str:
    provider = os.getenv("AI_PROVIDER", "openrouter")
    default_model = FAST_MODELS.get(provider, "openai/gpt-4o-mini")
    system_prompt = os.getenv(
        "AI_SYSTEM_PROMPT",
        "You are replying to Instagram DMs as the account owner. "
        "Keep replies short, friendly and casual, like a real text message. "
        "No emojis unless they fit naturally.",
    )

    if provider == "gemini":
        return _gemini_response(history, system_prompt)
    if provider == "openai":
        return _openai_response(history, system_prompt, base_url=None)
    return _openai_response(
        history, system_prompt, base_url="https://openrouter.ai/api/v1"
    )


def _openai_response(history: list, system_prompt: str, base_url) -> str:
    from openai import OpenAI

    if base_url:
        client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url=base_url)
        model = os.getenv("OPENROUTER_MODEL") or FAST_MODELS["openrouter"]
        extra = {
            "extra_headers": {
                "HTTP-Referer": "https://github.com/your-username/instabot",
                "X-Title": "InstaBot",
            }
        }
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = os.getenv("OPENAI_MODEL") or FAST_MODELS["openai"]
        extra = {}

    messages = [{"role": "system", "content": system_prompt}] + history
    resp = client.chat.completions.create(
        model=model, messages=messages, max_tokens=250, **extra
    )
    return resp.choices[0].message.content.strip()


def _gemini_response(history: list, system_prompt: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(
        os.getenv("GEMINI_MODEL") or FAST_MODELS["gemini"],
        system_instruction=system_prompt,
    )
    convo = "\n".join(
        f"{'Them' if m['role'] == 'user' else 'You'}: {m['content']}"
        for m in history
    )
    return model.generate_content(convo).text.strip()


# --------------------------------------------------------------------------- #
# Per-thread processing
# --------------------------------------------------------------------------- #
def process_thread(cl: Client, thread, replied: set, timestamps: list, threads_cooldown: dict):
    thread_id = thread.id
    messages = cl.direct_messages(thread_id, amount=3)
    if not messages:
        return

    newest = messages[0]
    msg_id = str(newest.id)
    msg_time = newest.timestamp.replace(tzinfo=timezone.utc).timestamp() if hasattr(newest, 'timestamp') and newest.timestamp else 0

    # Skip if: already replied, our own msg, no text, or older than bot start
    if msg_id in replied or str(newest.user_id) == str(cl.user_id) or not newest.text:
        return
    if msg_time < BOT_START_TIME:
        return

    if not can_reply(thread_id, timestamps, threads_cooldown):
        log.info(f"Rate limit hit — waiting (hr: {len(timestamps)}/{MAX_REPLIES_PER_HOUR})")
        return

    user_map = {str(u.pk): u.username for u in (thread.users or [])}
    sender = user_map.get(str(newest.user_id)) or newest.user.username or str(newest.user_id)
    log.info(f"@{sender}: {newest.text[:80]}")

    history = []
    for m in reversed(messages):
        if not m.text:
            continue
        history.append({
            "role": "assistant" if m.user_id == cl.user_id else "user",
            "content": m.text,
        })

    try:
        t0 = time.time()
        reply = get_ai_response(history)
        elapsed = time.time() - t0
        log.info(f"AI replied in {elapsed:.1f}s")
    except Exception as e:
        log.error(f"AI error: {e}")
        return

    time.sleep(random.uniform(0.5, 1.5))
    try:
        cl.direct_answer(thread_id, reply)
        replied.add(msg_id)
        save_json(REPLIED_FILE, list(replied))
        mark_replied(thread_id, timestamps, threads_cooldown)
        save_rate_limits(timestamps, threads_cooldown)
    except Exception as e:
        log.error(f"Send failed: {e}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    log.info("Starting DM bot...")
    cl = login()
    log.info(f"Logged in as @{cl.username}")

    replied = load_json(REPLIED_FILE) or []
    replied = set(replied)
    timestamps, threads_cooldown = load_rate_limits()
    log.info(f"Replied to {len(replied)} msgs total")

    check_interval = int(os.getenv("CHECK_INTERVAL", 8))

    while True:
        try:
            threads = cl.direct_threads(amount=10, selected_filter="unread")

            for thread in threads[:3]:
                process_thread(cl, thread, replied, timestamps, threads_cooldown)
                time.sleep(random.uniform(1, 2))

            log.info(f"Sleeping {check_interval}s...")
            time.sleep(check_interval)

        except KeyboardInterrupt:
            log.info("Shutdown.")
            save_json(REPLIED_FILE, list(replied))
            break
        except LoginRequired:
            log.warning("Session dropped, re-login...")
            time.sleep(5)
            cl = login()
        except ClientThrottledError:
            log.warning("Throttled! Backing off 5 min...")
            time.sleep(300)
        except ClientError as e:
            log.error(f"Instagram: {e}")
            time.sleep(30)
        except Exception as e:
            log.error(f"Error: {e}")
            time.sleep(30)


if __name__ == "__main__":
    main()
