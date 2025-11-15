# chat_lora_session_timer.py
import time
import textwrap
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from peft import PeftModel

# ========== Настройки ==========
BASE_MODEL_PATH = "llama_model"   # путь к базовой модели
LORA_DIR = "lora_output"          # путь к LoRA адаптеру
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_INPUT_LENGTH = 1024
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.7
TOP_P = 0.9

SESSION_LIMIT_MINUTES = 40  # длительность сессии в минутах

# Системный (технический) промпт, "вшит" в код
SYSTEM_PROMPT = (
    "Ты — опытный психиатр/терапевт, специализирующийся на выгорании и восстановлении ресурсов. "
    "Говори спокойно, тепло, эмпатично и по-человечески; не давай длинных лекций — коротко, по сути, с практическими шагами. "

    "Первое сообщение перед сессией: "
    "Перед началом всех сессий клиент сначала присылает только описание своего текущего состояния "
    "(одно короткое сообщение от его имени). "
    "Твоё ответное первое сообщение должно быть максимально коротким: одно-двухстрочное саммари состояния "
    "и предложение начать сессию в тихом, уединённом месте "
    "(например: «Я слышу, что вы сейчас [кратко]. Готов(а) начать — пожалуйста, найдите тихое уединённое место.»). "
    "После этого — обычный цикл сессии. "

    "Правила сессии: "
    "Сессия длится 40 минут. "
    "Время до окончания (N) всегда указывается пользователем в начале его сообщения в виде: "
    "«До конца сессии остаётся N минут.» "
    "Ты не считаешь и не вычисляешь время самостоятельно, а просто используешь указанное N "
    "для планирования ответа и выбора подходящей фазы работы. "
    "Формат обмена: сообщение клиента → твой ответ → и так до окончания времени. "

    "Структура внутри 40 минут (ориентируйся на N): "
    "0–5 мин: приветствие, проверка состояния, намерение встречи. "
    "5–10 мин: прояснение контекста/целей (или прогресса, если не первая встреча). "
    "10–30 мин: основная работа — исследование ситуаций, эмоций, мыслей; "
    "использовать краткие терапевтические техники (CBT-вопросы, поведенческие задания, телесные техники, дыхание, заземление). "
    "30–38 мин: интеграция — выделить 1–2 инсайта, практическое применение. "
    "38–40 мин: завершение — обратная связь, простое домашнее задание, договорённость о следующей сессии. "

    "Тактические инструкции: "
    "Всегда начинай с эмпатии и краткого резюме: «Я слышу, что…», «Понимаю, что сейчас…». "
    "Используй значение N, указанное пользователем: "
    "если N ≤ 2 — предложи короткую технику заземления/дыхания и домашнее задание; "
    "если 3–8 — интеграция + практическое упражнение; "
    "если >8 — можно глубже исследовать проблему. "
    "Старайся задавать 1–2 целевых вопроса или давать 1 конкретную рекомендацию в каждом ответе, "
    "чтобы к концу сессии было заметное продвижение. "
    "Предлагай конкретные, выполнимые домашние задания (не более 2) и измеримые наблюдения (что отслеживать). "
    "При признаках серьёзной опасности (суицидальные мысли, вред себе/другим, утрата контроля) "
    "незамедлительно рекомендовать срочную помощь или контактировать экстренные службы; "
    "дать фразу: «Если вы в опасности — позвоните в службу экстренной помощи/обратитесь к врачу.» "

    "Стиль и ограничения: "
    "Язык — понятный, ненаправляющий, не осуждающий. "
    "Избегай узкоспециального жаргона; при необходимости кратко объясняй термины. "
    "В конце каждого ответа при наличии времени предлагай короткую «метрику прогресса» "
    "(что клиент почувствовал/что изменилось). "

    "Пример шаблона ответа (кратко): "
    "эмпатия + короткое резюме; "
    "один-два вопроса или одна конкретная техника/рекомендация, адаптированные под N; "
    "если время ≈ 5–10 мин до конца — план интеграции + простое ДЗ; "
    "указать безопасную опцию при ухудшении состояния."
)

logging.set_verbosity_error()

# ========== ФУНКЦИЯ generate_reply (ОБЯЗАТЕЛЬНО) ==========
@torch.no_grad()
def generate_reply(history):
    """
    Принимает: history — список dict: [{"role": "user", "content": "..."}, ...]
    Возвращает: строку с ответом модели
    """
    prompt = f"System: {SYSTEM_PROMPT}\n\n"
    for turn in history:
        role = turn["role"]
        content = turn["content"].strip()
        prompt += f"{role.capitalize()}: {content}\n"
    prompt += "Assistant:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    generated_ids = output[0][inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return reply

# ========== Инициализация токенайзера и модели ==========
# ====== BEGIN CPU/GPU-адекватный блок замены ======
import os

print("Загружаем токенизатор...")
# Всегда берём токенизатор из базовой модели
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "[PAD]"

# Определяем режим работы: форсируем CPU, если переменная окружения FORCE_CPU установлена в "1" или если cuda недоступна
FORCE_CPU = os.environ.get("FORCE_CPU", "0") in ("1", "true", "True")
HAS_CUDA = torch.cuda.is_available()
USE_CUDA = (HAS_CUDA and not FORCE_CPU)

if USE_CUDA:
    print("CUDA доступна — загружаем модель с 4-bit квантованием (GPU).")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    print("Подключаем LoRA адаптер на GPU...")
    model = PeftModel.from_pretrained(base_model, LORA_DIR, device_map="auto")
    model.eval()
    DEVICE = "cuda"
else:
    # CPU-ветка — убираем все параметры bitsandbytes/4bit, загружаем на cpu
    print("GPU не доступен или FORCE_CPU включён — загружаем модель на CPU (может потребовать много RAM).")
    # low_cpu_mem_usage уменьшает пиковую нагрузку при загрузке (включая подкачку)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="cpu",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    print("Подключаем LoRA адаптер на CPU...")
    model = PeftModel.from_pretrained(base_model, LORA_DIR, device_map={"": "cpu"})
    model.eval()
    DEVICE = "cpu"

print(f"Модель готова. Устройство: {DEVICE}")
# ====== END CPU/GPU-адекватный блок ======

# ========== Вспомогательные функции ==========
def build_chat_prompt_with_system(history):
    """
    Формирует вход для модели: системный промпт + весь history,
    где history — список dict: {"role":"user"/"assistant", "content": "..."}.
    """
    prompt = f"System: {SYSTEM_PROMPT}\n\n"
    for turn in history:
        role = turn["role"]
        content = turn["content"].strip()
        prompt += f"{role.capitalize()}: {content}\n"
    prompt += "Assistant:"
    return prompt

@torch.no_grad()
def generate_reply(history):
    """
    Генерирует ответ модели по текущей истории (включая системный промпт).
    Возвращает строку сгенерированного текста.
    """
    prompt = build_chat_prompt_with_system(history)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_ids = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return text

def minutes_remaining(start_time):
    """Сколько целых минут осталось до конца сессии (int, >=0)."""
    elapsed_minutes = (time.time() - start_time) / 60.0
    remaining = int(max(0, SESSION_LIMIT_MINUTES - elapsed_minutes))
    return remaining

# ========== Основной интерактивный цикл ==========
def interactive_chat():
    print(f"\nЗапущен чат. Сессия длится {SESSION_LIMIT_MINUTES} минут.")
    print("Введите 'exit' или 'quit' для выхода.\n")

    history = []  # история, которую получает модель (включая временные фразы)
    session_start = time.time()        # время начала текущей 40-минутной сессии
    restart_after_response = False     # флаг: перезапустить таймер после следующего ответа модели

    while True:
        try:
            user_text = input("Вы: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nВыход.")
            break

        if not user_text:
            continue
        if user_text.lower() in ("exit", "quit"):
            print("Завершение.")
            break

# вычисляем оставшееся время и формируем фразу
        remaining = minutes_remaining(session_start)
        # если осталось >0, добавляем соответствующее число минут
        if remaining > 0:
            time_phrase = f"До конца сессии остаётся {remaining} минут."
            # не меняем timer; продолжаем
        else:
            # если осталось 0 — добавляем 0 и запоминаем, что надо перезапустить таймер
            time_phrase = "До конца сессии остаётся 0 минут."
            restart_after_response = True  # перезапустить после того, как модель ответит

        # формируем сообщение пользователя, которое и попадёт в контекст модели
        user_message_for_model = f"{user_text}\n\n{time_phrase}"

        # добавляем в историю как сообщение пользователя (именно с фразой о времени)
        history.append({"role": "user", "content": user_message_for_model})

        # генерируем ответ модели (она видит user_message_for_model)
        print("Модель генерирует ответ...")
        assistant_reply = generate_reply(history)

        # печатаем ответ (без изменения) — в историю добавим ответ
        print("\nАссистент:")
        print(textwrap.fill(assistant_reply, width=100))

        # добавляем ответ в историю (чтобы следующий ход учитывал его)
        history.append({"role": "assistant", "content": assistant_reply})

        # если нужно — перезапускаем таймер СРАЗУ ПОСЛЕ ТОГО, как модель ответила
        if restart_after_response:
            session_start = time.time()
            restart_after_response = False
            print("\n[Таймер сессии был перезапущен — начат новый 40-минутный цикл]")

    # конец цикла
    print("Чат завершён. До свидания.")


if __name__ == "__main__":
    interactive_chat()