"""
AI Agent Core — ядро агента с Chain-of-Thought и потоковым ответом.
Интеграция ТОЛЬКО с облачными LLM (OpenAI).
"""

import asyncio
import json
import os
from typing import AsyncIterator, Optional
from dataclasses import dataclass, field

import openai
from openai import AsyncOpenAI

from .context_loader import get_context_loader


@dataclass
class AgentConfig:
    """Конфигурация агента."""
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: str = ""
    use_cot: bool = True
    cot_depth: int = 3  # глубина CoT (кол-во шагов)


@dataclass
class AgentResponse:
    """Ответ агента."""
    content: str
    cot_steps: list[str] = field(default_factory=list)
    context_used: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    model: str = ""
    usage: dict = field(default_factory=dict)


class AIAgent:
    """
    ИИ-агент с Chain-of-Thought рассуждениями и потоковой передачей.

    Архитектура:
    ┌────────────────────────────────────────────┐
    │  User Query → Context Lookup → CoT Prompt  │
    │       ↓                                    │
    │  Cloud LLM (OpenAI) ← stream=True          │
    │       ↓                                    │
    │  Token-by-token → SSE → Frontend           │
    └────────────────────────────────────────────┘
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.context_loader = get_context_loader()
        self._client: Optional[AsyncOpenAI] = None

        # Загружаем системный промпт из контекста
        if not self.config.system_prompt:
            self.config.system_prompt = self.context_loader.get(
                "knowledge_base.prompts.system_prompt",
                "Ты — полезный ИИ-агент."
            )

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(base_url=os.environ.get("OPENAI_BASE_URL"), api_key=os.environ.get("OPENAI_API_KEY"))
        return self._client

    def _build_cot_prompt(self, query: str, context: str) -> list[dict]:
        """
        Строит Chain-of-Thought промпт.

        CoT-структура:
        1. ANALYSIS — анализ запроса
        2. CONTEXT — извлечение релевантного контекста
        3. REASONING — пошаговое рассуждение
        4. SOLUTION — выбор решения
        5. ANSWER — финальный ответ
        """
        cot_template = self.context_loader.get("knowledge_base.prompts.cot_template", "")

        system_message = f"""{self.config.system_prompt}

## Инструкции по Chain-of-Thought (CoT):
{cot_template}

## Доступный контекст:
{context}

## ВАЖНО:
- ВСЕГДА рассуждай шаг за шагом
- Каждый шаг помечай тегом: [АНАЛИЗ], [КОНТЕКСТ], [РАССУЖДЕНИЕ], [РЕШЕНИЕ], [ОТВЕТ]
- Используй контекст для точных ответов
- Если информации в контексте недостаточно, скажи об этом честно
- Отвечай на русском языке"""

        user_message = f"""Запрос пользователя: {query}

Пожалуйста, примени Chain-of-Thought подход для ответа на этот запрос."""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def _build_standard_prompt(self, query: str, context: str) -> list[dict]:
        """Строит стандартный промпт без CoT."""
        system_message = f"""{self.config.system_prompt}

Доступный контекст:
{context}

Отвечай на русском языке, используя контекст когда это уместно."""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ]

    async def stream_response(self, query: str) -> AsyncIterator[dict]:
        """
        Потоковый ответ агента.
        Возвращает события: token, cot_step, context, done.
        """
        # 1. Получаем релевантный контекст
        relevant_context = self.context_loader.get_relevant_context(query)
        system_context = self.context_loader.get_system_context()
        full_context = f"{system_context}\n\n--- Релевантный контекст ---\n{relevant_context}"

        # 2. Отправляем событие "контекст найден"
        yield {"type": "context", "data": relevant_context}

        # 3. Строим промпт (с CoT или без)
        if self.config.use_cot:
            messages = self._build_cot_prompt(query, full_context)
        else:
            messages = self._build_standard_prompt(query, full_context)

        # 4. Потоковая передача от OpenAI
        try:
            stream = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
            )

            accumulated = ""
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    accumulated += delta.content
                    yield {"type": "token", "data": delta.content}

            # 5. Отправляем итоговое событие
            yield {
                "type": "done",
                "data": {
                    "full_response": accumulated,
                    "model": self.config.model,
                    "context_used": relevant_context,
                }
            }

        except openai.APIError as e:
            yield {"type": "error", "data": f"OpenAI API Error: {str(e)}"}
        except Exception as e:
            yield {"type": "error", "data": f"Unexpected error: {str(e)}"}

    async def run_non_streaming(self, query: str) -> AgentResponse:
        """Не-потоковый режим (для отладки)."""
        try:
            relevant_context = self.context_loader.get_relevant_context(query)
            system_context = self.context_loader.get_system_context()
            full_context = f"{system_context}\n\n--- Релевантный контекст ---\n{relevant_context}"

            if self.config.use_cot:
                messages = self._build_cot_prompt(query, full_context)
            else:
                messages = self._build_standard_prompt(query, full_context)

            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=False,
            )

            content = response.choices[0].message.content
            return AgentResponse(
                content=content,
                context_used=[relevant_context],
                model=self.config.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                }
            )
        except openai.APIError as e:
            return AgentResponse(content=f"OpenAI API Error: {str(e)}")
        except Exception as e:
            return AgentResponse(content=f"Unexpected error: {str(e)}")


# Утилита для парсинга CoT-шагов из ответа
def parse_cot_steps(response: str) -> list[dict]:
    """Извлекает шаги CoT из ответа агента."""
    steps = []
    tags = ["[АНАЛИЗ]", "[КОНТЕКСТ]", "[РАССУЖДЕНИЕ]", "[РЕШЕНИЕ]", "[ОТВЕТ]"]

    remaining = response
    for i, tag in enumerate(tags):
        if tag in remaining:
            start = remaining.index(tag)
            end_tags = tags[i + 1:] if i + 1 < len(tags) else []
            end = len(remaining)
            for et in end_tags:
                pos = remaining.find(et, start + len(tag))
                if pos != -1 and pos < end:
                    end = pos
            step_content = remaining[start + len(tag):end].strip()
            steps.append({"step": tag.strip("[]"), "content": step_content})

    return steps
