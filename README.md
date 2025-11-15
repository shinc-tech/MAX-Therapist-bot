# MAX-Therapist-bot

markdown# MAX Therapist — Локальный ИИ-терапевт (Llama 3.2 + LoRA)

Локальный веб-интерфейс и Telegram-бот для терапии выгорания.  
Работает на CPU (медленно, но стабильно).  
Модель: **Llama 3.2 ru 3B** + **LoRA-адаптер**.

---

## Требования

- Docker + Docker Compose
- 16+ GB RAM (для CPU)
- 10+ GB свободного места

---

## Подготовка

### 1. Скачайте модель и LoRA по ссылке(обязательна установка файлов с диска, а не с Huggin Fece!)

https://disk.yandex.ru/d/kxFpzRuori7w0A

создайте у себя подобную структуру

MAX-Therapist-LLM/

├── llama_model/                 
├── lora_output/                 
├── Dockerfile.cpu              
├── docker-compose.yml           
├── requirements.txt             
├── README.md                    
├── app.py                      
├── max_bot.py                  
└── train_lora.py                

# 2. Запуск

Сборка (первый раз — долго) по команде

docker compose build --no-cache

после чего для запуска ввести

docker compose up

Откройте в браузере: http://localhost:8000
