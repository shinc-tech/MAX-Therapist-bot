# max_bot_llm_integrated.py

import asyncio
import logging
import time
import importlib.util
import sys
import os
from typing import Dict, Any

import aiomax
from aiomax import fsm
import aiomax.exceptions as am_exc

# ---------- CONFIG ----------
TOKEN = "f9LHodD0cOIXooNwrJaDa_QX6MpAm-djrvXU1imK9BfzBBAKlwDm9Axw7vmFydW0A9Z3Be-dRYTg880Biq-3"  # вставь свой токен
SESSION_MINUTES = 40
WEEK_SECONDS = 7 * 24 * 3600  # неделя


# ---------- Автоматический поиск файла LLM ----------
# Попытки найти файл LLM в нескольких местах (порядок важен — первый найденный используется).
CANDIDATE_PATHS = [
    # текущая папка, разные возможные имена ваших скриптов
    "chat_lora_session_timer.py",
    "train_lora.py",
    "Новый текстовый документ.txt",
    # если вы запускаете в Windows-проекте, попробуем типичный абсолютный путь из лога
    r"C:\Users\EClub\PyCharmMiscProject\chat_lora_session_timer.py",
    r"C:\Users\EClub\PyCharmMiscProject\train_lora.py",
    # добавим текущую рабочую директорию + имя из загруженного файла (на тот случай, если вы запускаете не из папки скрипта)
    os.path.join(os.getcwd(), "chat_lora_session_timer.py"),
    os.path.join(os.getcwd(), "train_lora.py"),
]

llm_script_path = None
for p in CANDIDATE_PATHS:
    if p and os.path.exists(p):
        llm_script_path = os.path.abspath(p)
        break

if llm_script_path is None:
    # Если не найден — выводим детальную ошибку с перечислением путей, которые проверили
    raise FileNotFoundError(
        "Файл скрипта нейросети не найден. Я проверил следующие пути:\n"
        + "\n".join(f" - {p}" for p in CANDIDATE_PATHS)
        + "\n\nПоложите ваш скрипт (тот файл, который у вас запускает модель и в котором указаны "
        "'BASE_MODEL_PATH = \"llama_model\"' и 'LORA_DIR = \"lora_output\"') в один из этих путей "
        "или измените список CANDIDATE_PATHS в коде."
    )

# Импортируем найденный файл как модуль local_llm_module (не меняем его содержимое)
spec = importlib.util.spec_from_file_location("local_llm_module", llm_script_path)
llm_module = importlib.util.module_from_spec(spec)
sys.modules["local_llm_module"] = llm_module
spec.loader.exec_module(llm_module)

# Экспортируем функцию generate_reply из импортированного модуля
generate_reply = llm_module.generate_reply

# Проверим ожидаемую функцию
if not hasattr(llm_module, "generate_reply"):
    raise AttributeError(
        f"Импортированный модуль {llm_script_path} не содержит функцию generate_reply(history). "
        "Проверьте содержимое скрипта."
    )

# ---------- BOT ----------
bot = aiomax.Bot(TOKEN, default_format="markdown")

# ---------- HELPERS ----------
def now_ts() -> float:
    return time.time()

async def ai_process(prompt: str) -> str:
    """
    Вызывает generate_reply из импортированного модуля LLM в отдельном потоке.
    История передаём как [{'role':'user', 'content': prompt}]; модуль внутри добавляет SYSTEM_PROMPT
    и использует свои относительные каталоги (llama_model, lora_output).
    """
    try:
        reply = await asyncio.to_thread(llm_module.generate_reply, [{"role": "user", "content": prompt}])
        if not isinstance(reply, str) or not reply.strip():
            return "⚠️ Модель вернула пустой или некорректный ответ."
        return reply.strip()
    except Exception as e:
        logging.exception("Ошибка при вызове LLM:")
        return f"❌ Ошибка при работе модели: {e}"

def _init_cursor_data() -> Dict[str, Any]:
    return {
        'state': 'COLLECTING',
        'messages': [],
        'ai_busy': False,
        'session_end_ts': None,
        'session_task': None,
        'week_task': None,
        'last_message_after_end': None
    }

async def safe_send(user_id: int, text: str, tries: int = 2, delay_between: float = 0.25):
    for attempt in range(tries):
        try:
            return await bot.send_message(user_id, text)
        except am_exc.InternalError as e:
            logging.warning("InternalError при send_message: %s. Попытка %d/%d", e, attempt+1, tries)
            if attempt + 1 < tries:
                await asyncio.sleep(delay_between)
            else:
                logging.exception("Не удалось отправить сообщение после нескольких попыток.")
                raise

# ---------- HANDLERS ----------
@bot.on_bot_start()
async def on_start(pd: aiomax.BotStartPayload, cursor: fsm.FSMCursor):
    data = _init_cursor_data()
    cursor.change_data(data)
    await pd.send(
        "Привет! 👋 Расскажи немного о себе и своих проблемах. "
        "Чем больше ты расскажешь, тем эффективнее пройдёт первая сессия."
    )

@bot.on_command('reweek')
async def cmd_reweek(ctx: aiomax.CommandContext, cursor: fsm.FSMCursor):
    data = cursor.get_data() or _init_cursor_data()
    task = data.get('week_task')
    if task and not task.done():
        task.cancel()
    data = _init_cursor_data()
    cursor.change_data(data)
    await ctx.reply("Таймер недели сброшен. Можешь начать новую сессию, просто напиши 'начать'.")

@bot.on_message()
async def on_message(message: aiomax.Message, cursor: fsm.FSMCursor):
    text = (message.content or '').strip()
    if not text:
        await message.reply("Пустое сообщение. Напиши, пожалуйста, текст.")
        return

    data = cursor.get_data() or _init_cursor_data()
    state = data['state']

    # --- Сбор вводной информации ---
    if state == 'COLLECTING':
        data['messages'].append(text)
        data['state'] = 'READY_TO_START'
        cursor.change_data(data)
        await message.send("Спасибо, я тебя выслушал. Когда будешь готов начать сессию — напиши 'начать'.")
        return

    # --- Начало сессии ---
    if state == 'READY_TO_START':
        if text.lower() in ['начать', 'start', 'go']:
            data['state'] = 'IN_SESSION'
            data['session_end_ts'] = now_ts() + SESSION_MINUTES * 60
            cursor.change_data(data)

            await message.send("Сессия началась")

            # 1️⃣ Прогрев модели — скрытый, без вывода пользователю
            if not data.get('warmed_up'):
                try:
                    _ = await ai_process(
                        "Начни сессию как терапевт. Просто ответь 'готов'.",
                    )
                    data['warmed_up'] = True
                    cursor.change_data(data)
                    logging.info("LLM успешно прогрета перед сессией.")
                except Exception as e:
                    logging.warning(f"Ошибка при прогреве модели: {e}")

            # 2️⃣ Основной запрос — формируем из истории клиента
            combined = "\n---\n".join(data.get('messages', [])) or "(нет предыдущих сообщений)"
            prompt_for_start = (
                f"\n{combined}\n\n"
            )

            data['ai_busy'] = True
            cursor.change_data(data)

            ai_reply = await ai_process(prompt_for_start)
            await message.send(ai_reply)

            data['ai_busy'] = False
            cursor.change_data(data)

            # 3️⃣ После первого ответа — автоматом уточнение контекста (не видно пользователю)
            try:
                _ = await ai_process(
                    "Теперь веди себя как терапевт в начале сессии: поддержи клиента и задай 1 мягкий вопрос.",
                )
            except Exception as e:
                logging.warning(f"Ошибка при внутреннем уточнении контекста: {e}")

            # 4️⃣ Таймер окончания сессии
            async def session_timer(user_id: int, cursor: fsm.FSMCursor):
                await asyncio.sleep(SESSION_MINUTES * 60)
                d = cursor.get_data() or _init_cursor_data()
                d['state'] = 'SESSION_ENDED_WAIT_LAST'
                cursor.change_data(d)
                await safe_send(
                    user_id,
                    "Время сессии подошло к концу. Пришли своё последнее сообщение, "
                    "чтобы я подготовил саммери и домашние задания."
                )

            data['session_task'] = asyncio.create_task(session_timer(message.user_id, cursor))
            cursor.change_data(data)

        else:
            await message.reply("Если готов начать — напиши 'начать'.")
        return

    # --- В ходе сессии ---
    if state == 'IN_SESSION':
        if data.get('ai_busy'):
            await message.reply("Подожди, я ещё обрабатываю предыдущий запрос.")
            return

        remaining = max(0, int((data.get('session_end_ts') or now_ts()) - now_ts())) // 60
        prompt = f"До конца сессии {remaining} мин.\nПользователь сказал: '{text}'"

        data['ai_busy'] = True
        cursor.change_data(data)

        ai_reply = await ai_process(prompt)
        await message.send(ai_reply)

        data['ai_busy'] = False
        cursor.change_data(data)
        return

    # --- Финал сессии ---
    if state == 'SESSION_ENDED_WAIT_LAST':
        data['last_message_after_end'] = text
        cursor.change_data(data)

        await message.reply("Спасибо! Готовлю саммери и рекомендации...")

        final_prompt = (
            f'Сессия завершена. Последнее сообщение клиента: "{text}". '
            "Подготовь саммери и домашние задания на неделю."
        )
        summary = await ai_process(final_prompt)
        await message.send(summary)

        await safe_send(message.user_id, "Следующий сеанс будет через неделю.")

        async def week_timer(user_id: int, cursor: fsm.FSMCursor):
            await asyncio.sleep(WEEK_SECONDS)
            await safe_send(
                user_id,
                'Неделя прошла, мы можем начать новую сессию, как будешь готов, напиши мне "Начать".'
            )
            cursor.change_data(_init_cursor_data())

        data['state'] = 'POST_SESSION_WAIT'
        data['week_task'] = asyncio.create_task(week_timer(message.user_id, cursor))
        cursor.change_data(data)
        return

    # --- После недели (ожидание новой сессии) ---
    if state == 'POST_SESSION_WAIT':
        await message.reply("Спасибо за сообщение! Мы скоро начнём новый цикл.")
        return

    # --- fallback ---
    await message.reply("Неизвестное состояние. Напиши /start, чтобы начать заново.")

# ---------- RUN ----------
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info("Используем LLM-скрипт: %s", llm_script_path)
    bot.run()
