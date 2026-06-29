"""LangGraph agent that orchestrates the scraping workflow per (university, department).

Graph:
    START ─► (dept missing? → discover_departments) ─► discover_url
             ─► fetch_page ─► run_scrapegraph ─► validate_and_finalize ─► END

Each node is wrapped with tenacity retry logic.  Results are cached via
CacheManager and state is persisted for resume via LangGraph checkpointing.
Playwright is used as a fallback when a page requires JavaScript rendering.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from typing import Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from tenacity import retry, stop_after_attempt, wait_exponential

from .cache import CacheManager
from .config import AppConfig
from .llm_factory import web_search
from .schema import Schema, build_extraction_prompt

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    university: str
    department: str | None
    listing_url: str | None
    page_html: str | None
    extracted_records: list[dict[str, Any]]
    error: str | None
    discovered_departments: list[str]
    need_discovery: bool


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_agent_graph(
    config: AppConfig,
    schema: Schema,
    llm: BaseChatModel,
    cache: CacheManager,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Create the compiled LangGraph state graph for scraping."""

    graph = StateGraph(AgentState)

    graph.add_node("discover_departments", _discover_departments_node(config, llm, cache))
    graph.add_node("discover_url", _discover_url_node(config, llm, cache))
    graph.add_node("fetch_page", _fetch_page_node(config, cache))
    graph.add_node("run_scrapegraph", _run_scrapegraph_node(config, schema, llm, cache))
    graph.add_node("validate_and_finalize", _validate_and_finalize_node(config, schema))

    graph.set_conditional_entry_point(
        _route_dept,
        {
            "discover": "discover_departments",
            "direct": "discover_url",
        },
    )
    graph.add_edge("discover_departments", END)
    graph.add_edge("discover_url", "fetch_page")
    graph.add_edge("fetch_page", "run_scrapegraph")
    graph.add_edge("run_scrapegraph", "validate_and_finalize")
    graph.add_edge("validate_and_finalize", END)

    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


def _route_dept(state: AgentState) -> str:
    if state.get("need_discovery") or state.get("department") is None:
        return "discover"
    return "direct"


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------


def _retry_for_node(config: AppConfig, node_name: str):
    return retry(
        stop=stop_after_attempt(config.scraping.max_retries_per_step),
        wait=wait_exponential(multiplier=1, min=config.scraping.request_delay_sec, max=30),
        reraise=True,
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def _discover_departments_node(
    config: AppConfig,
    llm: BaseChatModel,
    cache: CacheManager,
):
    async def _node(state: AgentState) -> AgentState:
        return await _discover_departments_impl(state, config, llm, cache)

    return _node


async def _discover_departments_impl(
    state: AgentState,
    config: AppConfig,
    llm: BaseChatModel,
    cache: CacheManager,
) -> AgentState:
    uni = state["university"]
    try:
        query = f"{uni} academic departments list site:.edu"

        max_attempts = config.scraping.max_retries_per_step
        search_results: list[dict[str, str]] = []
        for attempt in range(max_attempts):
            search_results = await web_search(
                query,
                provider=config.search.provider,
                api_key=config.search.bing_api_key,
                max_results=5,
            )
            if search_results:
                break
            await asyncio.sleep(config.scraping.request_delay_sec * (attempt + 1))

        urls_text = "\n".join(r["href"] for r in search_results if r.get("href"))

        prompt = f"""Given the university "{uni}", find all academic departments/schools.
From these search result URLs, suggest the most likely departments. If you cannot determine exact departments,
return a list of common department names for this type of institution.

Search results:
{urls_text or "No URLs available. Please use your knowledge of this university."}

Return ONLY a JSON list of department name strings, like: ["Computer Science", "Physics", "Mathematics"]"""

        response = await llm.ainvoke(prompt)
        text = _llm_response_text(response)

        json_match = re.search(r"\[.*?\]", text, re.DOTALL)
        if json_match:
            depts = json.loads(json_match.group())
            if isinstance(depts, list):
                state["discovered_departments"] = depts
                return state

        state["discovered_departments"] = []
    except Exception as e:
        state["discovered_departments"] = []
        state["error"] = str(e)

    return state


def _discover_url_node(
    config: AppConfig,
    llm: BaseChatModel,
    cache: CacheManager,
):
    async def _node(state: AgentState) -> AgentState:
        return await _discover_url_impl(state, config, llm, cache)

    return _node


async def _discover_url_impl(
    state: AgentState,
    config: AppConfig,
    llm: BaseChatModel,
    cache: CacheManager,
) -> AgentState:
    uni = state["university"]
    dept = state["department"] or ""

    if dept:
        query = f"{uni} {dept} faculty directory staff listing page"
    else:
        query = f"{uni} faculty directory staff listing page"

    try:
        max_attempts = config.scraping.max_retries_per_step
        search_results: list[dict[str, str]] = []
        for attempt in range(max_attempts):
            search_results = await web_search(
                query,
                provider=config.search.provider,
                api_key=config.search.bing_api_key,
                max_results=5,
            )
            if search_results:
                break
            await asyncio.sleep(config.scraping.request_delay_sec * (attempt + 1))

        urls_text = "\n".join(
            f"{i}: {r['title']} - {r['href']}"
            for i, r in enumerate(search_results)
            if r.get("href")
        )

        if not urls_text:
            state["error"] = "No search results found for faculty listing URL."
            return state

        prompt = f"""Find the most likely faculty/directory listing page URL for:
University: {uni}
Department: {dept or "All"}

Search results:
{urls_text}

Respond with ONLY the single most relevant URL and nothing else."""

        response = await llm.ainvoke(prompt)
        text = _llm_response_text(response).strip()

        url_match = re.search(r"https?://[^\s]+", text)
        if url_match:
            state["listing_url"] = url_match.group().rstrip(".)")
        else:
            state["listing_url"] = search_results[0]["href"] if search_results else None

        # Similar department fallback
        if not state.get("listing_url") and config.department.find_similar_department:
            similar_query = f"{uni} faculty directory listing page"
            similar_results = await web_search(
                similar_query,
                provider=config.search.provider,
                api_key=config.search.bing_api_key,
                max_results=3,
            )
            if similar_results:
                prompt = f"""Find the best faculty listing page for {uni} from:
{json.dumps([r["href"] for r in similar_results])}

Return ONLY the URL."""
                response2 = await llm.ainvoke(prompt)
                text2 = _llm_response_text(response2).strip()
                url_match2 = re.search(r"https?://[^\s]+", text2)
                if url_match2:
                    state["listing_url"] = url_match2.group().rstrip(".)")

    except Exception as e:
        state["error"] = str(e)

    return state


def _fetch_page_node(
    config: AppConfig,
    cache: CacheManager,
):
    """Fetch page HTML, using cache and Playwright fallback for JS pages."""

    async def _node(state: AgentState) -> AgentState:
        url = state.get("listing_url")
        if not url:
            state["error"] = "No URL to fetch."
            return state

        # 1. Check cache
        cached = cache.get_url_content(url)
        if cached:
            state["page_html"] = cached
            return state

        # 2. Try basic HTTP fetch
        html = await _http_fetch(url, config)
        if html and _has_content(html):
            cache.set_url_content(url, html)
            state["page_html"] = html
            return state

        # 3. Playwright fallback
        if html is None or not _has_content(html):
            html = await _playwright_fetch(url, config)
            if html:
                cache.set_url_content(url, html)
                state["page_html"] = html
                return state

        state["page_html"] = None
        return state

    return _node


async def _http_fetch(url: str, config: AppConfig) -> str | None:
    try:
        import aiohttp

        async with aiohttp.ClientSession() as session, session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=config.scraping.browser_timeout),
        ) as resp:
            if resp.status == 200:
                return await resp.text()
    except Exception:
        pass
    return None


async def _playwright_fetch(url: str, config: AppConfig) -> str | None:
    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=config.scraping.headless)
            page = await browser.new_page()
            try:
                await page.goto(url, timeout=config.scraping.browser_timeout * 1000)
                await page.wait_for_load_state("networkidle")
                html = await page.content()
                return html
            finally:
                await browser.close()
    except Exception:
        pass
    return None


def _has_content(html: str) -> bool:
    """Crude check: does the HTML contain enough text to be a listing page?"""
    text = re.sub(r"<[^>]+>", " ", html)
    return len(text.strip()) > 200


def _run_scrapegraph_node(
    config: AppConfig,
    schema: Schema,
    llm: BaseChatModel,
    cache: CacheManager,
):
    async def _node(state: AgentState) -> AgentState:
        return await _run_scrapegraph_impl(state, config, schema, llm, cache)

    return _node


async def _run_scrapegraph_impl(
    state: AgentState,
    config: AppConfig,
    schema: Schema,
    llm: BaseChatModel,
    cache: CacheManager,
) -> AgentState:
    url = state.get("listing_url")
    if not url:
        state["error"] = "No listing URL available for scraping."
        return state

    prompt_text = build_extraction_prompt(schema)

    # Check extraction cache
    input_hash = _cache_input_hash(url, prompt_text)
    cached_records = cache.get_extraction(input_hash)
    if cached_records is not None:
        state["extracted_records"] = cached_records
        return state

    # Use pre-fetched HTML if available, otherwise let ScrapeGraphAI fetch
    source = state.get("page_html") or url

    try:
        from scrapegraphai.graphs import SmartScraperGraph

        scraper = SmartScraperGraph(
            prompt=prompt_text,
            source=source,
            config={
                "llm": {
                    "model": config.llm.model,
                    "temperature": config.llm.temperature,
                },
            },
        )

        loop = asyncio.get_event_loop()

        def _run() -> dict:
            return scraper.run()

        result = await loop.run_in_executor(None, _run)

        records = _extract_records_result(result)

        if records:
            cache.set_extraction(input_hash, records)

        state["extracted_records"] = records

    except Exception as e:
        state["error"] = f"ScrapeGraphAI error: {e}"
        state["extracted_records"] = []

    return state


def _extract_records_result(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, dict):
        extracted = result.get("output", result.get("data", result))
        if isinstance(extracted, list):
            return extracted
        if isinstance(extracted, dict):
            return [extracted]
        return []
    if isinstance(result, list):
        return result
    return []


def _cache_input_hash(url: str, prompt: str) -> str:
    payload = json.dumps({"url": url, "prompt": prompt}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:40]


def _validate_and_finalize_node(
    config: AppConfig,
    schema: Schema,
):
    async def _node(state: AgentState) -> AgentState:
        records = state.get("extracted_records", [])

        validated: list[dict[str, Any]] = []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            clean: dict[str, Any] = {}
            for col in schema.extracted_columns():
                val = rec.get(col.name, "")
                if val is None:
                    val = ""
                if col.name.lower() == "email" and val and not _is_valid_email(str(val)):
                    val = ""
                clean[col.name] = val
            validated.append(clean)

        state["extracted_records"] = validated
        return state

    return _node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm_response_text(response: Any) -> str:
    if hasattr(response, "content"):
        return str(response.content)
    if isinstance(response, str):
        return response
    return str(response)


_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


def _is_valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(email))
