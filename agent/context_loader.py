"""
Context Loader — загружает и управляет контекстом из файлов/API.
Поддерживает поиск по ключам и семантическую фильтрацию.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional


class ContextLoader:
    """Загрузчик контекста для AI-агента."""

    def __init__(self, context_dir: str = "context"):
        self.context_dir = Path(context_dir)
        self._cache: dict[str, Any] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Загружает все JSON-файлы из директории контекста."""
        if not self.context_dir.exists():
            self.context_dir.mkdir(parents=True, exist_ok=True)

        for file_path in self.context_dir.glob("*.json"):
            with open(file_path, "r", encoding="utf-8") as f:
                self._cache[file_path.stem] = json.load(f)

    def reload(self) -> None:
        """Перезагрузка всего контекста."""
        self._cache.clear()
        self._load_all()

    def get(self, key: str, default: Any = None) -> Any:
        """Получить значение по ключу (поддерживает вложенность через точку)."""
        keys = key.split(".")
        value = self._cache
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def search(self, query: str) -> list[dict]:
        """Простой поиск по контексту (ключи и FAQ)."""
        results = []
        query_lower = query.lower()

        # Поиск по FAQ
        faq = self.get("knowledge_base.faq", [])
        for item in faq:
            if query_lower in item.get("q", "").lower() or query_lower in item.get("a", "").lower():
                results.append({"type": "faq", "question": item["q"], "answer": item["a"]})

        # Поиск по продуктам
        products = self.get("knowledge_base.company.products", [])
        for product in products:
            if query_lower in product.get("name", "").lower() or query_lower in product.get("description", "").lower():
                results.append({"type": "product", "data": product})

        return results

    def get_system_context(self) -> str:
        """Формирует системный контекст для промпта."""
        company = self.get("knowledge_base.company", {})
        tech = self.get("knowledge_base.technical", {})
        cot = self.get("knowledge_base.prompts.cot_template", "")

        context_parts = [
            f"Компания: {company.get('name', 'N/A')}",
            f"Описание: {company.get('description', 'N/A')}",
            f"Продукты: {', '.join(p['name'] for p in company.get('products', []))}",
            f"API версия: {tech.get('api_version', 'N/A')}",
            f"Поддерживаемые модели: {', '.join(tech.get('supported_models', []))}",
            f"Chain-of-Thought шаблон: {cot}",
        ]
        return "\n".join(context_parts)

    def get_relevant_context(self, query: str) -> str:
        """Извлекает релевантный контекст для конкретного запроса."""
        search_results = self.search(query)
        if not search_results:
            return "Релевантный контекст не найден."

        parts = []
        for r in search_results[:3]:  # Берём топ-3
            if r["type"] == "faq":
                parts.append(f"FAQ: Q={r['question']}\nA={r['answer']}")
            elif r["type"] == "product":
                p = r["data"]
                parts.append(f"Продукт: {p['name']} — {p['description']}")

        return "\n".join(parts)


# Синглтон для удобства
_context_instance: Optional[ContextLoader] = None


def get_context_loader(context_dir: str = "context") -> ContextLoader:
    global _context_instance
    if _context_instance is None:
        _context_instance = ContextLoader(context_dir)
    return _context_instance
