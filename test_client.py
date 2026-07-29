"""
Тестовый клиент для проверки работы агента.
Демонстрирует:
- Потоковый режим (SSE)
- Не-потоковый режим (JSON)
- Поиск по контексту
"""

import asyncio
import json
import httpx

BASE_URL = "http://localhost:8000"

async def test_health():
    """Проверка health endpoint."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{BASE_URL}/health")
        print(f"[HEALTH] Status: {resp.status_code}")
        print(f"[HEALTH] Body: {resp.json()}")
        print()

async def test_context():
    """Проверка контекста."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{BASE_URL}/context/search?q=FlowAgent")
        print(f"[CONTEXT SEARCH] Status: {resp.status_code}")
        data = resp.json()
        print(f"[CONTEXT SEARCH] Found: {len(data.get('results', []))} results")
        for r in data.get("results", []):
            print(f"  - {r.get('type')}: {json.dumps(r.get('data', r.get('question', '')), ensure_ascii=False)[:200]}")
        print()

async def test_non_streaming():
    """Проверка не-потокового режима."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        payload = {
            "message": "Расскажи о продукте FlowAgent",
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "use_cot": True
        }
        print(f"[NON-STREAM] Sending: {payload['message']}")
        resp = await client.post(f"{BASE_URL}/chat", json=payload)
        print(f"[NON-STREAM] Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"[NON-STREAM] Response: {data.get('response', '')[:500]}...")
            print(f"[NON-STREAM] Model: {data.get('model')}")
            print(f"[NON-STREAM] Usage: {data.get('usage')}")
        else:
            print(f"[NON-STREAM] Error: {resp.text}")
        print()

async def test_streaming():
    """Проверка потокового режима."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        form_data = {
            "message": "Что такое Chain-of-Thought и как его использовать с вашим API?",
            "use_cot": "True"
        }
        print(f"[STREAM] Sending: {form_data['message']}")
        print("[STREAM] Response (token-by-token):")
        print("-" * 60)

        async with client.stream(
            "POST",
            f"{BASE_URL}/chat/stream",
            data=form_data,
        ) as response:
            print(f"[STREAM] Status: {response.status_code}")
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        event = json.loads(line[6:])
                        if event["type"] == "token":
                            print(event["content"], end="", flush=True)
                        elif event["type"] == "context":
                            print(f"\n[CONTEXT] {event['content'][:100]}...")
                        elif event["type"] == "done":
                            print(f"\n\n[DONE] Model: {event['data']['model']}")
                        elif event["type"] == "error":
                            print(f"\n[ERROR] {event['content']}")
                    except json.JSONDecodeError:
                        pass
        print()
        print("-" * 60)

async def main():
    print("=" * 60)
    print("AI Agent — Test Suite")
    print("=" * 60)
    print()

    try:
        await test_health()
    except Exception as e:
        print(f"[SKIP] Health check failed (server not running?): {e}")
        print("Make sure the server is running: python -m api.server")
        return

    try:
        await test_context()
    except Exception as e:
        print(f"[SKIP] Context test failed: {e}")

    try:
        await test_non_streaming()
    except Exception as e:
        print(f"[SKIP] Non-streaming test failed: {e}")

    try:
        await test_streaming()
    except Exception as e:
        print(f"[SKIP] Streaming test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
