"""Tests for the agent graph helpers and routing logic."""

from __future__ import annotations

from pathlib import Path

from facultyai.cache import CacheManager
from facultyai.config import AppConfig, LLMConfig
from facultyai.llm_factory import get_llm
from facultyai.schema import ColumnDef, Schema
from facultyai.scraper_graph import (
    AgentState,
    _cache_input_hash,
    _extract_records_result,
    _has_content,
    _is_valid_email,
    _llm_response_text,
    _route_dept,
    build_agent_graph,
)


class TestHelpers:
    def test_valid_email(self) -> None:
        assert _is_valid_email("jsmith@mit.edu") is True
        assert _is_valid_email("a.b@c.edu.sg") is True
        assert _is_valid_email("not-an-email") is False
        assert _is_valid_email("") is False
        assert _is_valid_email("user@domain") is False

    def test_has_content(self) -> None:
        short = "<html><body>hi</body></html>"
        assert _has_content(short) is False

        long = "<html><body>" + "x " * 200 + "</body></html>"
        assert _has_content(long) is True

    def test_llm_response_text_str(self) -> None:
        assert _llm_response_text("hello") == "hello"

    def test_llm_response_text_object(self) -> None:
        class FakeResponse:
            content = "fake content"

        assert _llm_response_text(FakeResponse()) == "fake content"

    def test_extract_records_from_dict_output(self) -> None:
        result = _extract_records_result({"output": [{"name": "A"}, {"name": "B"}]})
        assert result == [{"name": "A"}, {"name": "B"}]

    def test_extract_records_from_list(self) -> None:
        result = _extract_records_result([{"name": "A"}, {"name": "B"}])
        assert result == [{"name": "A"}, {"name": "B"}]

    def test_extract_records_from_single_dict(self) -> None:
        result = _extract_records_result({"output": {"name": "A"}})
        assert result == [{"name": "A"}]

    def test_extract_records_from_string(self) -> None:
        result = _extract_records_result("not a list")
        assert result == []

    def test_cache_input_hash_stable(self) -> None:
        h1 = _cache_input_hash("http://example.com", "extract name")
        h2 = _cache_input_hash("http://example.com", "extract name")
        assert h1 == h2

    def test_cache_input_hash_changes(self) -> None:
        h1 = _cache_input_hash("http://example.com", "extract name")
        h2 = _cache_input_hash("http://example.com", "extract email")
        assert h1 != h2


class TestRouting:
    def test_route_need_discovery(self) -> None:
        state: AgentState = {"university": "MIT", "need_discovery": True}
        assert _route_dept(state) == "discover"

    def test_route_null_department(self) -> None:
        state: AgentState = {"university": "MIT"}
        assert _route_dept(state) == "discover"

    def test_route_with_department(self) -> None:
        state: AgentState = {"university": "MIT", "department": "EECS", "need_discovery": False}
        assert _route_dept(state) == "direct"


class TestGraphConstruction:
    def test_build_graph_returns_compiled_graph(self, tmp_path: Path) -> None:
        config = AppConfig(llm=LLMConfig(api_key="sk-fake-for-test"))
        schema = Schema(columns=[ColumnDef(name="Name", type="extracted")])
        llm = get_llm(config.llm)
        cm = CacheManager(tmp_path / "cache")

        try:
            graph = build_agent_graph(config, schema, llm, cm)
            assert graph is not None
            assert hasattr(graph, "ainvoke")
        finally:
            cm.close()
