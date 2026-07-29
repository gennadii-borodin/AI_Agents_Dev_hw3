"""
FastAPI Server — API для AI-агента с SSE потоковой передачей.

Архитектура подключения:
┌──────────┐     HTTP/SSE      ┌──────────────┐     Async      ┌──────────────┐
│  Client  │ ◄──────────────►  │  FastAPI     │ ◄────────────► │  AI Agent    │
│  (HTML)  │   POST /chat/     │  Server      │   stream()     │  + OpenAI    │
│          │   stream          │  (uvicorn)   │                │  Cloud API   │
└──────────┘                   └──────────────┘                └──────────────┘
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import asyncio
from typing import Optional
from fastapi import FastAPI, Form
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.agent_core import AIAgent, AgentConfig
from agent.context_loader import get_context_loader

# Проверка API ключа при старте
_API_KEY = os.environ.get("OPENAI_API_KEY")
if not _API_KEY:
    print("=" * 60)
    print("  WARNING: OPENAI_API_KEY not set!")
    print("  The agent will NOT be able to answer requests.")
    print("  Set the environment variable:")
    print('    export OPENAI_API_KEY="sk-your-key-here"')
    print("=" * 60)

app = FastAPI(
    title="AI Agent API",
    description="Потоковый API для ИИ-агента с Chain-of-Thought и контекстной памятью",
    version="2.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальный экземпляр агента (ленивая инициализация)
_agent: Optional[AIAgent] = None


def get_agent() -> AIAgent:
    global _agent
    if _agent is None:
        config = AgentConfig(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=4096,
            use_cot=True,
        )
        _agent = AIAgent(config)
    return _agent


# ─── Pydantic Models ───

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "gpt-4o-mini"
    temperature: Optional[float] = 0.7
    use_cot: Optional[bool] = True


# ─── SSE Streaming Endpoint ───

@app.post("/chat/stream")
async def chat_stream(message: str = Form(...), use_cot: str = Form("true")):
    """
    Основной эндпоинт для потокового чата.
    Возвращает Server-Sent Events (SSE).
    """
    agent = get_agent()
    agent.config.use_cot = use_cot.lower() in ("true", "1", "yes")

    async def event_generator():
        try:
            async for event in agent.stream_response(message):
                event_type = event["type"]
                data = event["data"]

                if event_type == "token":
                    # Отправляем токен в SSE формате
                    yield f"data: {json.dumps({'type': 'token', 'content': data}, ensure_ascii=False)}\n\n"

                elif event_type == "context":
                    yield f"data: {json.dumps({'type': 'context', 'content': data}, ensure_ascii=False)}\n\n"

                elif event_type == "done":
                    yield f"data: {json.dumps({'type': 'done', 'data': data}, ensure_ascii=False)}\n\n"

                elif event_type == "error":
                    yield f"data: {json.dumps({'type': 'error', 'content': data}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': f'Stream error: {str(e)}'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        }
    )


# ─── JSON (не-потоковый) Endpoint ───

@app.post("/chat")
async def chat(request: ChatRequest):
    """Не-потоковый эндпоинт (возвращает полный JSON)."""
    agent = get_agent()
    agent.config.use_cot = request.use_cot
    agent.config.model = request.model
    agent.config.temperature = request.temperature

    response = await agent.run_non_streaming(request.message)
    return JSONResponse(content={
        "response": response.content,
        "context_used": response.context_used,
        "model": response.model,
        "usage": response.usage,
    })


# ─── Context Management ───

@app.get("/context")
async def get_context(key: Optional[str] = None):
    """Получить контекст (весь или по ключу)."""
    loader = get_context_loader()
    if key:
        value = loader.get(f"knowledge_base.{key}")
        return JSONResponse(content={"key": key, "value": value})
    return JSONResponse(content=loader._cache.get("knowledge_base", {}))


@app.get("/context/search")
async def search_context(q: str):
    """Поиск по контексту."""
    loader = get_context_loader()
    results = loader.search(q)
    return JSONResponse(content={"query": q, "results": results})


@app.post("/context/reload")
async def reload_context():
    """Перезагрузка контекста из файлов."""
    loader = get_context_loader()
    loader.reload()
    return JSONResponse(content={"status": "ok", "message": "Контекст перезагружен"})


# ─── Health Check ───

@app.get("/health")
async def health():
    loader = get_context_loader()
    return JSONResponse(content={
        "status": "healthy",
        "context_loaded": bool(loader._cache),
        "context_keys": list(loader._cache.keys()),
    })


# ─── Web UI ───

@app.get("/", response_class=HTMLResponse)
async def web_ui():
    """Веб-интерфейс для тестирования агента."""
    return HTMLResponse(
        content="""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Agent — Streaming Demo</title>
    <style>
        :root {
            --bg: #0d1117;
            --card: #161b22;
            --border: #30363d;
            --text: #c9d1d9;
            --accent: #58a6ff;
            --green: #3fb950;
            --orange: #d2991d;
            --purple: #a371f7;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            display: flex;
            justify-content: center;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            width: 100%;
        }
        header {
            text-align: center;
            padding: 24px 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 20px;
        }
        header h1 { color: var(--accent); font-size: 2em; margin-bottom: 8px; }
        header .subtitle { color: #8b949e; font-size: 0.95em; }
        .arch-badge {
            display: inline-block;
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 4px 12px;
            margin: 4px;
            font-size: 0.8em;
            color: var(--green);
        }
        .chat-area {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            min-height: 400px;
            max-height: 500px;
            overflow-y: auto;
            margin-bottom: 16px;
        }
        .message { margin-bottom: 16px; animation: fadeIn 0.3s ease; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        .message.user { text-align: right; }
        .message.user .bubble {
            background: var(--accent);
            color: #fff;
            display: inline-block;
            padding: 10px 16px;
            border-radius: 16px 4px 16px 16px;
            max-width: 80%;
        }
        .message.agent .bubble {
            background: #21262d;
            border: 1px solid var(--border);
            padding: 10px 16px;
            border-radius: 4px 16px 16px 16px;
            display: inline-block;
            max-width: 85%;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .message.agent .meta { font-size: 0.75em; color: #8b949e; margin-bottom: 4px; }
        .cot-step {
            margin: 6px 0;
            padding: 6px 12px;
            border-left: 3px solid var(--purple);
            background: #1a1f2b;
            border-radius: 0 6px 6px 0;
            font-size: 0.85em;
        }
        .cot-step .step-tag {
            font-weight: bold;
            color: var(--purple);
        }
        .context-info {
            background: #1a2332;
            border: 1px solid var(--orange);
            border-radius: 6px;
            padding: 8px 12px;
            margin: 8px 0;
            font-size: 0.8em;
            color: var(--orange);
        }
        .input-area {
            display: flex;
            gap: 10px;
        }
        .input-area input {
            flex: 1;
            padding: 14px 18px;
            border: 1px solid var(--border);
            border-radius: 10px;
            background: var(--card);
            color: var(--text);
            font-size: 1em;
            outline: none;
            transition: border-color 0.2s;
        }
        .input-area input:focus { border-color: var(--accent); }
        .input-area button {
            padding: 14px 28px;
            background: var(--accent);
            color: #fff;
            border: none;
            border-radius: 10px;
            font-size: 1em;
            cursor: pointer;
            transition: background 0.2s;
            white-space: nowrap;
        }
        .input-area button:hover { background: #4090e0; }
        .input-area button:disabled { opacity: 0.5; cursor: not-allowed; }
        .controls {
            display: flex;
            gap: 16px;
            margin-bottom: 16px;
            align-items: center;
            flex-wrap: wrap;
        }
        .controls label {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.9em;
            cursor: pointer;
        }
        .typing-indicator {
            display: none;
            color: #8b949e;
            font-style: italic;
            padding: 8px;
        }
        .typing-indicator.active { display: block; }
        .loading-dots::after {
            content: '';
            animation: dots 1.5s steps(3, end) infinite;
        }
        @keyframes dots {
            0% { content: ''; }
            33% { content: '.'; }
            66% { content: '..'; }
            100% { content: '...'; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 AI Agent — Streaming + CoT + Context</h1>
            <p class="subtitle">Потоковый ИИ-агент с Chain-of-Thought и контекстной памятью</p>
            <div style="margin-top: 12px;">
                <span class="arch-badge">🔗 Cloud LLM: OpenAI</span>
                <span class="arch-badge">📡 SSE Streaming</span>
                <span class="arch-badge">🧠 Chain-of-Thought</span>
                <span class="arch-badge">📚 Context-Aware</span>
            </div>
        </header>

        <div class="controls">
            <label>
                <input type="checkbox" id="useCOT" checked>
                Chain-of-Thought
            </label>
            <button onclick="loadContext()" style="background: var(--card); border: 1px solid var(--border); color: var(--text); padding: 8px 16px; border-radius: 8px; cursor: pointer; font-size: 0.85em;">📋 Загрузить контекст</button>
            <button onclick="clearChat()" style="background: var(--card); border: 1px solid var(--border); color: var(--text); padding: 8px 16px; border-radius: 8px; cursor: pointer; font-size: 0.85em;">🗑 Очистить</button>
        </div>

        <div class="chat-area" id="chatArea">
            <div class="message agent">
                <div class="bubble">
                    👋 Привет! Я AI-агент TechFlow AI.<br>
                    Я использую <b>Chain-of-Thought</b> подход и <b>потоковую передачу</b> через SSE.<br>
                    Задай мне вопрос — и я буду думать вслух, шаг за шагом! 🧠
                </div>
            </div>
        </div>

        <div class="typing-indicator" id="typingIndicator">
            🤔 Агент думает <span class="loading-dots"></span>
        </div>

        <div class="input-area">
            <input
                type="text"
                id="messageInput"
                placeholder="Введите ваш вопрос..."
            />
            <button id="sendButton" onclick="sendMessage()">▶ Отправить</button>
        </div>
    </div>

    <script>
        const chatArea = document.getElementById('chatArea');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const typingIndicator = document.getElementById('typingIndicator');
        const useCOT = document.getElementById('useCOT');

        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') sendMessage();
        });

        function addMessage(role, content) {
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.innerHTML = `<div class="bubble">${content}</div>`;
            chatArea.appendChild(div);
            chatArea.scrollTop = chatArea.scrollHeight;
            return div;
        }

        function addContextInfo(context) {
            const div = document.createElement('div');
            div.className = 'context-info';
            div.textContent = '📚 Контекст: ' + context;
            chatArea.appendChild(div);
            chatArea.scrollTop = chatArea.scrollHeight;
        }

        function addCoTStep(step, content) {
            const div = document.createElement('div');
            div.className = 'cot-step';
            div.innerHTML = `<span class="step-tag">[${step}]</span> ${content}`;
            chatArea.appendChild(div);
            chatArea.scrollTop = chatArea.scrollHeight;
        }

        async function sendMessage() {
            try {
                const message = messageInput.value.trim();
                if (!message) return;

                if (!messageInput || !sendButton || !chatArea || !typingIndicator) {
                    console.error('UI elements not found');
                    return;
                }

                // Disable input
                messageInput.disabled = true;
                sendButton.disabled = true;

                // Show user message
                addMessage('user', escapeHtml(message));
                messageInput.value = '';

                // Show typing indicator
                typingIndicator.classList.add('active');

                // Create agent message container
                const agentMsg = addMessage('agent', '');
                const bubble = agentMsg.querySelector('.bubble');

                const formData = new FormData();
                formData.append('message', message);
                formData.append('use_cot', useCOT.checked);

                const response = await fetch('/chat/stream', {
                    method: 'POST',
                    body: formData,
                });

                if (!response.ok) {
                    const errorText = await response.text();
                    let errorMsg = errorText;
                    try {
                        const errorJson = JSON.parse(errorText);
                        errorMsg = errorJson.detail || JSON.stringify(errorJson);
                    } catch (_) {}
                    bubble.textContent = `❌ Ошибка сервера (${response.status}): ${errorMsg}`;
                    return;
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n\n');
                    buffer = lines.pop() || '';

                    for (const line of lines) {
                        const dataLine = line.startsWith('data: ') ? line.slice(6) : line;
                        if (!dataLine) continue;

                        try {
                            const event = JSON.parse(dataLine);

                            if (event.type === 'token') {
                                bubble.textContent += event.content;
                                chatArea.scrollTop = chatArea.scrollHeight;
                            } else if (event.type === 'context') {
                                addContextInfo(event.content);
                            } else if (event.type === 'done') {
                                const meta = document.createElement('div');
                                meta.className = 'meta';
                                meta.textContent = `Модель: ${event.data.model} | Потоковая передача завершена ✓`;
                                agentMsg.appendChild(meta);
                            } else if (event.type === 'error') {
                                bubble.textContent += `\n❌ Ошибка: ${event.content}`;
                            }
                        } catch (e) {
                            bubble.textContent += `\n⚠️ Ошибка обработки данных: ${e.message}`;
                        }
                    }
                }
            } catch (error) {
                const errorMsg = `❌ Ошибка: ${error.message}`;
                const existingBubble = document.querySelector('.message.agent:last-child .bubble');
                if (existingBubble) {
                    existingBubble.textContent = errorMsg;
                } else {
                    addMessage('agent', errorMsg);
                }
            } finally {
                typingIndicator.classList.remove('active');
                messageInput.disabled = false;
                sendButton.disabled = false;
                messageInput.focus();
            }
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        async function loadContext() {
            try {
                const resp = await fetch('/context');
                const data = await resp.json();
                addMessage('agent', `<b>📚 Загруженный контекст:</b><br><pre style="font-size:0.75em;max-height:200px;overflow-y:auto;">${escapeHtml(JSON.stringify(data, null, 2))}</pre>`);
            } catch (e) {
                addMessage('agent', `❌ Ошибка загрузки контекста: ${e.message}`);
            }
        }

        function clearChat() {
            chatArea.innerHTML = '';
            addMessage('agent', '<div class="bubble">Чат очищен. Задайте новый вопрос! 🧹</div>');
        }
    </script>
</body>
</html>""",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


# ─── Entry Point ───

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
