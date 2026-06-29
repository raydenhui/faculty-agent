"""LLM factory and web-search tools for FacultyAI.

Provides ``get_llm()`` which turns the config into a LangChain ``BaseChatModel``,
and lightweight async search helpers used by the agent graph.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from .config import LLMConfig


def get_llm(config: LLMConfig) -> BaseChatModel:
    """Return a LangChain chat model from *config*."""
    provider = config.provider.lower()

    if provider in ("openai", "deepseek", "openai_compatible"):
        kwargs: dict[str, Any] = {
            "model": config.model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)

    if provider == "azure":
        return AzureChatOpenAI(
            azure_deployment=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.api_key or None,
            azure_endpoint=config.azure_endpoint or None,
            api_version=config.api_version or None,
        )

    if provider == "anthropic":
        return ChatAnthropic(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.api_key or None,
        )

    if provider == "google":
        return ChatGoogleGenerativeAI(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.api_key or None,
        )

    raise ValueError(f"Unsupported LLM provider: {config.provider}")


# ---------------------------------------------------------------------------
# Web search helpers (duckduckgo via ddgs, bing via requests)
# ---------------------------------------------------------------------------


async def web_search(
    query: str, provider: str = "duckduckgo", api_key: str = "", max_results: int = 5
) -> list[dict[str, str]]:
    """Async web search returning a list of ``{"title", "href", "body"}`` dicts."""
    if provider.lower() == "duckduckgo":
        return await _search_duckduckgo(query, max_results)
    if provider.lower() == "bing":
        return await _search_bing(query, api_key, max_results)
    return []


async def _search_duckduckgo(query: str, max_results: int) -> list[dict[str, str]]:
    try:
        from ddgs import DDGS
    except Exception:  # pragma: no cover
        return []

    loop = asyncio.get_event_loop()

    def _run() -> list[dict[str, str]]:
        with DDGS() as ddgs:
            results = []
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    {
                        "title": r.get("title", ""),
                        "href": r.get("href", ""),
                        "body": r.get("body", ""),
                    }
                )
            return results

    return await loop.run_in_executor(None, _run)


async def _search_bing(query: str, api_key: str, max_results: int) -> list[dict[str, str]]:
    if not api_key:
        return []
    import aiohttp

    url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {"q": query, "count": max_results}

    async with aiohttp.ClientSession() as session, session.get(url, headers=headers, params=params) as resp:
        if resp.status != 200:
            return []
        data = await resp.json()
        return [
            {
                "title": item.get("name", ""),
                "href": item.get("url", ""),
                "body": item.get("snippet", ""),
            }
            for item in data.get("webPages", {}).get("value", [])
        ]
