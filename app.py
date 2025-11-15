# app.py
import importlib.util
import sys, os, asyncio
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

CANDIDATE_PATHS = [ "max_bot.py", "train_lora.py"]
llm_script_path = next((p for p in CANDIDATE_PATHS if os.path.exists(p)), None)

if llm_script_path is None:
    def generate_reply(messages):
        return "LLM-скрипт не найден. Положите train_lora.py в корень проекта."
else:
    spec = importlib.util.spec_from_file_location("local_llm_module", llm_script_path)
    llm_module = importlib.util.module_from_spec(spec)
    sys.modules["local_llm_module"] = llm_module
    spec.loader.exec_module(llm_module)
    generate_reply = getattr(llm_module, "generate_reply", lambda msgs: "Функция generate_reply не найдена.")

app = FastAPI()

HTML_TEMPLATE = """
<!doctype html>
<html><head><meta charset="utf-8"><title>MAX Therapist — demo</title></head>
<body>
  <h2>MAX Therapist — локальное демо</h2>
  <form action="/ask" method="post">
    <textarea name="prompt" rows="6" cols="80" placeholder="Опишите, как вы себя чувствуете..."></textarea><br/>
    <input type="submit" value="Отправить">
  </form>
  <pre id="resp">{response}</pre>
</body></html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_TEMPLATE.format(response="")

@app.post("/ask", response_class=HTMLResponse)
async def ask(prompt: str = Form(...)):
    reply = await asyncio.to_thread(generate_reply, [{"role":"user","content": prompt}])
    return HTML_TEMPLATE.format(response=reply)