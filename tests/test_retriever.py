"""Unit tests for app.retriever — Task 6.

Test layers
-----------
1. Unit tests (no Qdrant) — always run.
2. Integration tests (require Qdrant) — marked ``@pytest.mark.integration``,
   skipped automatically when Qdrant is not reachable.

Run unit tests only::

    pytest tests/test_retriever.py -m "not integration"

Run all::

    pytest tests/test_retriever.py
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import app.retriever as retriever_module
from app.retriever import (
    DEFAULT_MIN_SCORE,
    DEFAULT_TOP_K,
    RetrievedChunk,
    SearchResult,
    _build_section_filter,
    _chunk_uuid,
    _embed_query,
    _point_to_chunk,
    build_filter,
    embed_query,
    qdrant_point_to_chunk,
    rerank_chunks,
    retrieve,
    search_chunks,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _make_scored_point(
    chunk_id: str = "neo_city_factsheet_001",
    score: float = 0.85,
    section: str = "factsheet",
    topic: str = "project_overview",
    text: str = "Thông tin tổng quan về dự án NEO CITY.",
    status: str = "estimated",
    legal_sensitivity: str = "medium",
    source_title: str = "FACTSHEET DỰ ÁN",
) -> Any:
    """Build a minimal mock ScoredPoint."""
    point = MagicMock()
    point.id = str(uuid.uuid4())
    point.score = score
    point.payload = {
        "id": chunk_id,
        "section": section,
        "topic": topic,
        "source_title": source_title,
        "status": status,
        "legal_sensitivity": legal_sensitivity,
        "text": text,
        "project": "NEO CITY",
        "source_doc": "All database - NEO CITY.docx",
        "version": "2026-05",
    }
    return point


def _make_mock_client(scored_points: list[Any]) -> MagicMock:
    client = MagicMock()
    mock_result = MagicMock()
    mock_result.points = scored_points
    client.query_points.return_value = mock_result
    return client


def _make_mock_embedder(vector: list[float] | None = None) -> MagicMock:
    vec = vector or [0.1] * 384
    embedder = MagicMock()
    mock_arr = MagicMock()
    mock_arr.tolist.return_value = vec
    embedder.embed.return_value = iter([mock_arr])
    return embedder


# ---------------------------------------------------------------------------
# _chunk_uuid (legacy)
# ---------------------------------------------------------------------------

class TestChunkUuid:
    def test_returns_string(self):
        assert isinstance(_chunk_uuid("neo_city_factsheet_001"), str)

    def test_deterministic(self):
        assert _chunk_uuid("some_id") == _chunk_uuid("some_id")

    def test_different_ids_give_different_uuids(self):
        assert _chunk_uuid("id_a") != _chunk_uuid("id_b")

    def test_valid_uuid_format(self):
        parsed = uuid.UUID(_chunk_uuid("test_chunk"))
        assert parsed.version == 5


# ---------------------------------------------------------------------------
# _build_section_filter (legacy)
# ---------------------------------------------------------------------------

class TestBuildSectionFilter:
    def test_empty_list_returns_none(self):
        assert _build_section_filter([]) is None

    def test_single_section_returns_must_filter(self):
        f = _build_section_filter(["legal"])
        assert f is not None
        assert len(f.must) == 1
        assert f.must[0].key == "section"
        assert f.must[0].match.value == "legal"

    def test_multiple_sections_returns_should_filter(self):
        f = _build_section_filter(["pricing", "price_sheet"])
        assert f is not None
        assert f.should is not None
        assert len(f.should) == 2
        values = {c.match.value for c in f.should}
        assert values == {"pricing", "price_sheet"}

    def test_none_when_no_filter_needed(self):
        assert _build_section_filter([]) is None


# ---------------------------------------------------------------------------
# build_filter (new — project + section)
# ---------------------------------------------------------------------------

class TestBuildFilter:
    def test_project_only(self):
        f = build_filter("NEO CITY", [])
        assert f is not None
        # must has exactly the project condition
        assert len(f.must) == 1
        assert f.must[0].key == "project"
        assert f.must[0].match.value == "NEO CITY"

    def test_project_and_single_section(self):
        f = build_filter("NEO CITY", ["legal"])
        assert f is not None
        assert len(f.must) == 2
        keys = {c.key for c in f.must}
        assert "project" in keys
        assert "section" in keys

    def test_project_and_multiple_sections(self):
        f = build_filter("NEO CITY", ["pricing", "price_sheet"])
        assert f is not None
        # must has: project condition + nested should filter
        assert len(f.must) == 2
        project_cond = next(
            c for c in f.must if hasattr(c, "key") and c.key == "project"
        )
        assert project_cond.match.value == "NEO CITY"
        # Second must item is a nested Filter with should conditions
        nested = next(c for c in f.must if hasattr(c, "should") and c.should)
        section_values = {c.match.value for c in nested.should}
        assert section_values == {"pricing", "price_sheet"}

    def test_empty_project_empty_sections_returns_none(self):
        assert build_filter("", []) is None

    def test_always_includes_project_neo_city(self):
        f = build_filter("NEO CITY", ["legal"])
        project_conditions = [
            c for c in f.must if hasattr(c, "key") and c.key == "project"
        ]
        assert len(project_conditions) == 1
        assert project_conditions[0].match.value == "NEO CITY"


# ---------------------------------------------------------------------------
# qdrant_point_to_chunk (new)
# ---------------------------------------------------------------------------

class TestQdrantPointToChunk:
    def test_returns_dict(self):
        point = _make_scored_point()
        result = qdrant_point_to_chunk(point)
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        point = _make_scored_point()
        result = qdrant_point_to_chunk(point)
        for key in ("id", "score", "section", "topic", "status",
                    "legal_sensitivity", "source_title", "text"):
            assert key in result, f"Missing key: {key}"

    def test_id_from_payload(self):
        point = _make_scored_point(chunk_id="pricing_001")
        result = qdrant_point_to_chunk(point)
        assert result["id"] == "pricing_001"

    def test_id_falls_back_to_point_id(self):
        point = MagicMock()
        point.id = "fallback-uuid"
        point.score = 0.5
        point.payload = {}
        result = qdrant_point_to_chunk(point)
        assert result["id"] == "fallback-uuid"

    def test_score_preserved(self):
        point = _make_scored_point(score=0.77)
        result = qdrant_point_to_chunk(point)
        assert result["score"] == pytest.approx(0.77)

    def test_section_preserved(self):
        point = _make_scored_point(section="legal")
        assert qdrant_point_to_chunk(point)["section"] == "legal"

    def test_legal_sensitivity_preserved(self):
        point = _make_scored_point(legal_sensitivity="critical")
        assert qdrant_point_to_chunk(point)["legal_sensitivity"] == "critical"

    def test_text_preserved(self):
        text = "NEO CITY chưa đủ điều kiện mở bán."
        point = _make_scored_point(text=text)
        assert qdrant_point_to_chunk(point)["text"] == text

    def test_empty_payload_defaults(self):
        point = MagicMock()
        point.id = "x"
        point.score = 0.3
        point.payload = {}
        result = qdrant_point_to_chunk(point)
        assert result["section"] == ""
        assert result["text"] == ""

    def test_preserves_full_schema_payload_fields(self):
        point = _make_scored_point()
        result = qdrant_point_to_chunk(point)
        assert result["project"] == "NEO CITY"
        assert result["source_doc"] == "All database - NEO CITY.docx"
        assert result["version"] == "2026-05"


# ---------------------------------------------------------------------------
# _point_to_chunk (legacy)
# ---------------------------------------------------------------------------

class TestPointToChunk:
    def test_basic_conversion(self):
        point = _make_scored_point(chunk_id="test_001", score=0.9, section="pricing")
        chunk = _point_to_chunk(point)
        assert isinstance(chunk, RetrievedChunk)
        assert chunk.chunk_id == "test_001"
        assert chunk.score == pytest.approx(0.9)
        assert chunk.section == "pricing"

    def test_missing_payload_fields_default_empty(self):
        point = MagicMock()
        point.id = "abc"
        point.score = 0.5
        point.payload = {}
        chunk = _point_to_chunk(point)
        assert chunk.chunk_id == "abc"
        assert chunk.section == ""
        assert chunk.text == ""

    def test_text_preserved(self):
        text = "Dự án NEO CITY tọa lạc tại khu vực trung tâm."
        chunk = _point_to_chunk(_make_scored_point(text=text))
        assert chunk.text == text

    def test_payload_preserved(self):
        chunk = _point_to_chunk(_make_scored_point())
        assert chunk.payload["project"] == "NEO CITY"


# ---------------------------------------------------------------------------
# _embed_query (legacy)
# ---------------------------------------------------------------------------

class TestEmbedQuery:
    def _make_embedder(self, vector: list[float]) -> MagicMock:
        embedder = MagicMock()
        mock_arr = MagicMock()
        mock_arr.tolist.return_value = vector
        embedder.embed.return_value = iter([mock_arr])
        return embedder

    def test_returns_list_of_floats(self):
        result = _embed_query(self._make_embedder([0.1, 0.2, 0.3]), "test query")
        assert result == pytest.approx([0.1, 0.2, 0.3])

    def test_calls_embed_with_list(self):
        embedder = self._make_embedder([0.0])
        _embed_query(embedder, "hello")
        embedder.embed.assert_called_once_with(["hello"])

    def test_handles_plain_list_return(self):
        embedder = MagicMock()
        embedder.embed.return_value = iter([[0.4, 0.5]])
        assert _embed_query(embedder, "query") == pytest.approx([0.4, 0.5])

    def test_embedder_cached_by_model_name(self, monkeypatch):
        retriever_module._EMBEDDER_CACHE.clear()

        class FakeEmbedding:
            init_count = 0

            def __init__(self, model_name: str):
                self.model_name = model_name
                FakeEmbedding.init_count += 1

            def embed(self, texts):
                assert texts
                return iter([[0.1, 0.2]])

        import sys
        import types

        monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeEmbedding))

        first = embed_query("query 1", "fake-model")
        second = embed_query("query 2", "fake-model")

        assert first == pytest.approx([0.1, 0.2])
        assert second == pytest.approx([0.1, 0.2])
        assert FakeEmbedding.init_count == 1


# ---------------------------------------------------------------------------
# search_chunks — unit test with mocked Qdrant + embedder (legacy API)
# ---------------------------------------------------------------------------

class TestSearchChunksUnit:
    def test_returns_search_result(self):
        result = search_chunks(
            "NEO CITY là gì?", "general",
            client=_make_mock_client([_make_scored_point()]),
            embedder=_make_mock_embedder(),
        )
        assert isinstance(result, SearchResult)

    def test_chunk_count_matches_qdrant_return(self):
        points = [_make_scored_point(score=0.9 - i * 0.1) for i in range(3)]
        result = search_chunks(
            "query", "general",
            client=_make_mock_client(points),
            embedder=_make_mock_embedder(),
        )
        assert len(result.chunks) == 3

    def test_intent_stored_in_result(self):
        result = search_chunks(
            "giá bán?", "pricing",
            client=_make_mock_client([]),
            embedder=_make_mock_embedder(),
        )
        assert result.intent == "pricing"

    def test_legal_intent_applies_section_filter(self):
        client = _make_mock_client([])
        search_chunks("pháp lý", "legal", client=client, embedder=_make_mock_embedder())
        query_filter = client.query_points.call_args.kwargs.get("query_filter")
        assert query_filter is not None
        assert len(query_filter.must) == 1
        assert query_filter.must[0].match.value == "legal"

    def test_general_intent_no_filter(self):
        client = _make_mock_client([])
        search_chunks("Tổng quan dự án", "general", client=client, embedder=_make_mock_embedder())
        assert client.query_points.call_args.kwargs.get("query_filter") is None

    def test_empty_result_when_no_hits(self):
        result = search_chunks(
            "câu hỏi lạ", "general",
            client=_make_mock_client([]),
            embedder=_make_mock_embedder(),
        )
        assert result.chunks == []

    def test_top_k_passed_to_qdrant(self):
        client = _make_mock_client([])
        search_chunks("query", "general", top_k=7, client=client, embedder=_make_mock_embedder())
        assert client.query_points.call_args.kwargs["limit"] == 7

    def test_default_top_k(self):
        client = _make_mock_client([])
        search_chunks("query", "general", client=client, embedder=_make_mock_embedder())
        assert client.query_points.call_args.kwargs["limit"] == DEFAULT_TOP_K

    def test_pricing_intent_includes_price_sheet(self):
        client = _make_mock_client([])
        search_chunks("bảng giá", "pricing", client=client, embedder=_make_mock_embedder())
        query_filter = client.query_points.call_args.kwargs.get("query_filter")
        assert query_filter is not None
        assert query_filter.should is not None
        section_values = {c.match.value for c in query_filter.should}
        assert "pricing" in section_values
        assert "price_sheet" in section_values


# ---------------------------------------------------------------------------
# retrieve() — new intent-aware API
# ---------------------------------------------------------------------------

class TestRetrieve:
    """Unit tests for retrieve() using mocked Qdrant client and embedder."""

    # -- pricing ----------------------------------------------------------------

    def test_pricing_question_targets_pricing_sections(self):
        """Pricing question must use sections ['pricing', 'price_sheet']."""
        client = _make_mock_client([])
        embedder = _make_mock_embedder()
        result = retrieve(
            "Căn 2PN giá bao nhiêu?",
            client=client,
            embedder=embedder,
        )
        assert "pricing" in result["target_sections"]
        assert "price_sheet" in result["target_sections"]

    def test_pricing_question_passes_project_filter_to_qdrant(self):
        client = _make_mock_client([])
        embedder = _make_mock_embedder()
        retrieve("Căn 2PN giá bao nhiêu?", client=client, embedder=embedder)
        query_filter = client.query_points.call_args.kwargs.get("query_filter")
        assert query_filter is not None
        project_conds = [
            c for c in query_filter.must if hasattr(c, "key") and c.key == "project"
        ]
        assert len(project_conds) == 1
        assert project_conds[0].match.value == "NEO CITY"

    def test_pricing_question_risk_high(self):
        client = _make_mock_client([])
        embedder = _make_mock_embedder()
        result = retrieve("Căn 2PN giá bao nhiêu?", client=client, embedder=embedder)
        assert result["risk_level"] == "high"

    # -- legal ------------------------------------------------------------------

    def test_legal_question_forces_legal_section_only(self):
        """Legal question: target_sections must be exactly ['legal']."""
        client = _make_mock_client([])
        embedder = _make_mock_embedder()
        result = retrieve(
            "Dự án đã mở bán chưa?",
            client=client,
            embedder=embedder,
        )
        assert result["target_sections"] == ["legal"]

    def test_legal_question_must_use_legal_only_true(self):
        client = _make_mock_client([])
        embedder = _make_mock_embedder()
        result = retrieve("Dự án đã mở bán chưa?", client=client, embedder=embedder)
        assert result["must_use_legal_only"] is True

    def test_legal_question_risk_critical(self):
        client = _make_mock_client([])
        embedder = _make_mock_embedder()
        result = retrieve("Dự án đã mở bán chưa?", client=client, embedder=embedder)
        assert result["risk_level"] == "critical"

    def test_legal_question_qdrant_filter_section_legal(self):
        client = _make_mock_client([])
        embedder = _make_mock_embedder()
        retrieve("Dự án đã mở bán chưa?", client=client, embedder=embedder)
        query_filter = client.query_points.call_args.kwargs.get("query_filter")
        section_conds = [
            c for c in query_filter.must if hasattr(c, "key") and c.key == "section"
        ]
        assert len(section_conds) == 1
        assert section_conds[0].match.value == "legal"

    # -- guaranteed profit ------------------------------------------------------

    def test_profit_guarantee_risk_critical(self):
        """Profit guarantee question must produce risk=critical."""
        client = _make_mock_client([])
        embedder = _make_mock_embedder()
        result = retrieve(
            "Có cam kết lợi nhuận không?",
            client=client,
            embedder=embedder,
        )
        assert result["risk_level"] == "critical"

    def test_profit_guarantee_sections_include_legal_and_market(self):
        client = _make_mock_client([])
        embedder = _make_mock_embedder()
        result = retrieve("Có cam kết lợi nhuận không?", client=client, embedder=embedder)
        assert "legal" in result["target_sections"]
        assert "market" in result["target_sections"]

    def test_profit_guarantee_must_use_legal_only_false(self):
        """Profit guarantee: legal sections included but must_use_legal_only=False."""
        client = _make_mock_client([])
        embedder = _make_mock_embedder()
        result = retrieve("Có cam kết lợi nhuận không?", client=client, embedder=embedder)
        assert result["must_use_legal_only"] is False

    # -- unknown intent ---------------------------------------------------------

    def test_unknown_intent_does_not_call_qdrant(self):
        """Unknown intent must return early — Qdrant must NOT be called."""
        client = _make_mock_client([])
        embedder = _make_mock_embedder()
        result = retrieve(
            "Câu hỏi ngoài dữ liệu ABCXYZ",
            client=client,
            embedder=embedder,
        )
        client.query_points.assert_not_called()
        assert result["chunks"] == []
        assert result["reason"] == "unknown_intent"

    def test_unknown_intent_target_sections_empty(self):
        client = _make_mock_client([])
        embedder = _make_mock_embedder()
        result = retrieve("Câu hỏi ngoài dữ liệu ABCXYZ", client=client, embedder=embedder)
        assert result["target_sections"] == []

    # -- min_score filtering ----------------------------------------------------

    def test_min_score_filters_low_score_chunks(self):
        """Chunks below min_score must be dropped from result."""
        high = _make_scored_point(chunk_id="hi", score=0.80, section="pricing")
        low = _make_scored_point(chunk_id="lo", score=0.05, section="pricing")
        client = _make_mock_client([high, low])
        embedder = _make_mock_embedder()
        result = retrieve(
            "Căn 2PN giá bao nhiêu?",
            min_score=0.15,
            client=client,
            embedder=embedder,
        )
        chunk_ids = [c["id"] for c in result["chunks"]]
        assert "hi" in chunk_ids
        assert "lo" not in chunk_ids

    def test_min_score_all_filtered_gives_reason(self):
        """When all chunks are below min_score, reason should be set."""
        low = _make_scored_point(chunk_id="lo", score=0.05, section="pricing")
        client = _make_mock_client([low])
        embedder = _make_mock_embedder()
        result = retrieve(
            "Căn 2PN giá bao nhiêu?",
            min_score=0.15,
            client=client,
            embedder=embedder,
        )
        assert result["chunks"] == []
        assert result["reason"] == "no_chunks_above_min_score"

    def test_min_score_zero_keeps_all_chunks(self):
        """min_score=0 must keep all chunks regardless of score."""
        points = [_make_scored_point(score=s, section="pricing")
                  for s in [0.8, 0.1, 0.01]]
        client = _make_mock_client(points)
        embedder = _make_mock_embedder()
        result = retrieve(
            "Căn 2PN giá bao nhiêu?",
            min_score=0.0,
            client=client,
            embedder=embedder,
        )
        assert len(result["chunks"]) == 3

    # -- result structure -------------------------------------------------------

    def test_result_has_all_required_keys(self):
        client = _make_mock_client([_make_scored_point()])
        embedder = _make_mock_embedder()
        result = retrieve("Căn 2PN giá bao nhiêu?", client=client, embedder=embedder)
        for key in ("question", "intent", "risk_level", "target_sections",
                    "must_use_legal_only", "chunks", "reason"):
            assert key in result, f"Missing key: {key}"

    def test_chunk_has_all_required_keys(self):
        point = _make_scored_point(score=0.9, section="pricing",
                                   legal_sensitivity="high")
        client = _make_mock_client([point])
        embedder = _make_mock_embedder()
        result = retrieve(
            "Căn 2PN giá bao nhiêu?",
            min_score=0.0,
            client=client,
            embedder=embedder,
        )
        assert len(result["chunks"]) == 1
        chunk = result["chunks"][0]
        for key in ("id", "score", "section", "topic", "status",
                    "legal_sensitivity", "source_title", "text"):
            assert key in chunk, f"Chunk missing key: {key}"

    def test_question_echoed_in_result(self):
        client = _make_mock_client([])
        embedder = _make_mock_embedder()
        q = "Dự án đã mở bán chưa?"
        result = retrieve(q, client=client, embedder=embedder)
        assert result["question"] == q

    def test_reason_none_when_chunks_found(self):
        point = _make_scored_point(score=0.9, section="pricing")
        client = _make_mock_client([point])
        embedder = _make_mock_embedder()
        result = retrieve(
            "Căn 2PN giá bao nhiêu?",
            min_score=0.0,
            client=client,
            embedder=embedder,
        )
        assert result["reason"] is None

    # -- no OPENAI_API_KEY required ---------------------------------------------

    def test_no_openai_api_key_required(self):
        """retrieve() must work without OPENAI_API_KEY in environment."""
        import os
        client = _make_mock_client([])
        embedder = _make_mock_embedder()
        env_backup = os.environ.pop("OPENAI_API_KEY", None)
        try:
            result = retrieve("Căn 2PN giá bao nhiêu?", client=client, embedder=embedder)
            assert "chunks" in result
        finally:
            if env_backup is not None:
                os.environ["OPENAI_API_KEY"] = env_backup

    # -- limit passed to Qdrant -------------------------------------------------

    def test_limit_passed_to_qdrant(self):
        client = _make_mock_client([])
        embedder = _make_mock_embedder()
        retrieve("Căn 2PN giá bao nhiêu?", limit=7, client=client, embedder=embedder)
        assert client.query_points.call_args.kwargs["limit"] == 7

    def test_default_limit_is_20(self):
        client = _make_mock_client([])
        embedder = _make_mock_embedder()
        retrieve("Căn 2PN giá bao nhiêu?", client=client, embedder=embedder)
        assert client.query_points.call_args.kwargs["limit"] == 20

    def test_with_payload_true(self):
        client = _make_mock_client([])
        embedder = _make_mock_embedder()
        retrieve("Căn 2PN giá bao nhiêu?", client=client, embedder=embedder)
        assert client.query_points.call_args.kwargs.get("with_payload") is True

    def test_top_k_limits_retrieved_chunks(self):
        points = [_make_scored_point(chunk_id=str(i), score=0.8) for i in range(10)]
        client = _make_mock_client(points)
        embedder = _make_mock_embedder()
        result = retrieve("Căn 2PN giá bao nhiêu?", top_k=3, client=client, embedder=embedder)
        assert len(result["chunks"]) == 3


# ---------------------------------------------------------------------------
# rerank_chunks
# ---------------------------------------------------------------------------

class TestRerankChunks:
    def test_pricing_apartment_topic_boost(self):
        c1 = {"id": "1", "score": 0.8, "topic": "family_apartment_policy", "text": "Chính sách"}
        c2 = {"id": "2", "score": 0.7, "topic": "apartment_pricing", "text": "Giá căn hộ"}
        chunks = [c1, c2]
        clf = MagicMock()
        clf.intent = "pricing"
        
        result = rerank_chunks("Căn 2PN giá bao nhiêu?", chunks, classification=clf)
        
        assert result[0]["id"] == "2"

    def test_pricing_with_product_token_penalizes_sales_policy(self):
        chunks = [
            {"id": "1", "score": 0.82, "section": "sales_policy", "topic": "payment_policy", "text": "ChÃ­nh sÃ¡ch thanh toÃ¡n"},
            {"id": "2", "score": 0.74, "section": "pricing", "topic": "apartment_pricing", "text": "GiÃ¡ 2PN"},
        ]
        result = rerank_chunks(
            "CÄƒn 2PN giÃ¡ bao nhiÃªu?",
            chunks,
            classification={"intent": "pricing", "target_sections": ["pricing", "price_sheet"]},
        )
        assert result[0]["id"] == "2"

    def test_product_query_penalizes_pricing_principles(self):
        chunks = [
            {"id": "1", "score": 0.82, "section": "pricing", "topic": "pricing_principles", "text": "NguyÃªn táº¯c giÃ¡"},
            {"id": "2", "score": 0.75, "section": "factsheet", "topic": "apartment_products", "text": "Gá»“m Studio+, 1PN+1, 2PN"},
        ]
        result = rerank_chunks(
            "Dá»± Ã¡n cÃ³ nhá»¯ng loáº¡i cÄƒn nÃ o?",
            chunks,
            classification={"intent": "product", "target_sections": ["factsheet", "pricing"]},
        )
        assert result[0]["id"] == "2"

    def test_sales_policy_prefers_sales_policy_over_price_sheet(self):
        chunks = [
            {"id": "1", "score": 0.82, "section": "price_sheet", "topic": "pricing_sheet_overview", "text": "Báº£ng giÃ¡"},
            {"id": "2", "score": 0.76, "section": "sales_policy", "topic": "payment_policy", "text": "ChÃ­nh sÃ¡ch thanh toÃ¡n"},
        ]
        result = rerank_chunks(
            "ChÃ­nh sÃ¡ch thanh toÃ¡n tháº¿ nÃ o?",
            chunks,
            classification={"intent": "sales_policy", "target_sections": ["sales_policy", "price_sheet"]},
        )
        assert result[0]["id"] == "2"

    def test_amenities_prefers_amenities_topic_over_brochure_summary(self):
        chunks = [
            {"id": "1", "score": 0.82, "section": "factsheet", "topic": "brochure_summary", "text": "Tá»•ng quan dá»± Ã¡n"},
            {"id": "2", "score": 0.74, "section": "factsheet", "topic": "amenities", "text": "Há»“ trung tÃ¢m Neo Lake vÃ  Neo Square"},
        ]
        result = rerank_chunks(
            "NEO CITY cÃ³ tiá»‡n Ã­ch gÃ¬?",
            chunks,
            classification={"intent": "amenities", "target_sections": ["factsheet"]},
        )
        assert result[0]["id"] == "2"

    def test_legal_opening_sale_prefers_legal_status_and_warnings(self):
        chunks = [
            {"id": "1", "score": 0.82, "section": "legal", "topic": "legal_note", "text": "Ghi chÃº phÃ¡p lÃ½"},
            {"id": "2", "score": 0.72, "section": "legal", "topic": "legal_status_and_warnings", "text": "ChÆ°a má»Ÿ bÃ¡n, chÆ°a nháº­n cá»c"},
        ]
        result = rerank_chunks(
            "Dá»± Ã¡n Ä‘Ã£ Ä‘á»§ Ä‘iá»u kiá»‡n má»Ÿ bÃ¡n chÆ°a?",
            chunks,
            classification={"intent": "legal", "target_sections": ["legal"], "must_use_legal_only": True, "risk_level": "critical"},
        )
        assert result[0]["id"] == "2"
        assert result[1]["id"] == "1"

    def test_pricing_lowrise_topic_boost(self):
        c1 = {"id": "1", "score": 0.8, "topic": "apartment_pricing", "text": "Giá căn hộ"}
        c2 = {"id": "2", "score": 0.7, "topic": "lowrise_pricing", "text": "Giá shophouse"}
        chunks = [c1, c2]
        clf = MagicMock()
        clf.intent = "pricing"
        
        result = rerank_chunks("Shophouse giá bao nhiêu?", chunks, classification=clf)
        
        assert result[0]["id"] == "2"

    def test_legal_topic_boost(self):
        c1 = {"id": "1", "score": 0.8, "topic": "sales_policy", "text": "Chính sách"}
        c2 = {"id": "2", "score": 0.7, "topic": "legal_status_and_warnings", "text": "Pháp lý mở bán"}
        chunks = [c1, c2]
        clf = MagicMock()
        clf.intent = "legal"
        
        result = rerank_chunks("Dự án đã mở bán chưa?", chunks, classification=clf)
        
        assert result[0]["id"] == "2"

    def test_persona_family_boost(self):
        c1 = {"id": "1", "score": 0.8, "topic": "buyer_persona_young_professional", "text": "Người trẻ"}
        c2 = {"id": "2", "score": 0.7, "topic": "buyer_persona_family", "text": "Gia đình"}
        chunks = [c1, c2]
        clf = MagicMock()
        clf.intent = "persona"
        
        result = rerank_chunks("Gia đình trẻ phù hợp sản phẩm nào?", chunks, classification=clf)
        
        assert result[0]["id"] == "2"

    def test_persona_young_boost(self):
        c1 = {"id": "1", "score": 0.8, "topic": "buyer_persona_family", "text": "Gia đình"}
        c2 = {"id": "2", "score": 0.7, "topic": "buyer_persona_young_professional", "text": "Người trẻ"}
        chunks = [c1, c2]
        clf = MagicMock()
        clf.intent = "persona"
        
        result = rerank_chunks("Người trẻ mua căn đầu tiên phù hợp căn nào?", chunks, classification=clf)
        
        assert result[0]["id"] == "2"

    def test_scores_preserved_and_rerank_added(self):
        c1 = {"id": "1", "score": 0.8, "topic": "unknown", "text": "Something"}
        result = rerank_chunks("Test?", [c1])
        assert "score" in result[0]
        assert result[0]["score"] == 0.8
        assert "rerank_score" in result[0]
        assert result[0]["rerank_score"] >= 0.8
        
    def test_top_k_limits_output(self):
        chunks = [{"id": str(i), "score": 0.5, "topic": "", "text": "Text"} for i in range(10)]
        result = rerank_chunks("Test?", chunks, top_k=3)
        assert len(result) == 3


class TestRetrieveRerankIntegration:
    def test_retrieve_calls_rerank_chunks(self):
        point = _make_scored_point(score=0.9, section="pricing")
        client = _make_mock_client([point])
        embedder = _make_mock_embedder()
        with patch("app.retriever.rerank_chunks") as rerank_mock:
            rerank_mock.return_value = [
                {
                    "id": "pricing_001",
                    "score": 0.9,
                    "section": "pricing",
                    "topic": "apartment_pricing",
                    "status": "estimated",
                    "legal_sensitivity": "high",
                    "source_title": "FACTSHEET",
                    "text": "Giá căn hộ",
                    "rerank_score": 1.1,
                }
            ]
            result = retrieve("Căn 2PN giá bao nhiêu?", client=client, embedder=embedder)
        rerank_mock.assert_called_once()
        assert result["chunks"][0]["id"] == "pricing_001"


class TestRerankChunksExtended:
    def test_original_score_unchanged(self):
        c1 = {"id": "1", "score": 0.8, "topic": "apartment_pricing", "text": "Giá căn hộ"}
        result = rerank_chunks("Căn 2PN giá bao nhiêu?", [c1])
        assert result[0]["score"] == pytest.approx(0.8)

    def test_classification_as_dict(self):
        chunks = [
            {"id": "1", "score": 0.8, "section": "sales_policy", "topic": "payment_policy", "text": "Chính sách thanh toán"},
            {"id": "2", "score": 0.7, "section": "pricing", "topic": "apartment_pricing", "text": "Giá căn hộ"},
        ]
        result = rerank_chunks(
            "Chính sách thanh toán thế nào?",
            chunks,
            classification={"intent": "sales_policy", "target_sections": ["sales_policy", "price_sheet"]},
        )
        assert result[0]["id"] == "1"

    def test_classification_none(self):
        result = rerank_chunks("Test?", [{"id": "1", "score": 0.5, "text": "Text"}], classification=None)
        assert result[0]["id"] == "1"
        assert "rerank_score" in result[0]

    def test_missing_topic_text_source_title_does_not_crash(self):
        chunks = [{"id": "1", "score": 0.5, "section": "factsheet", "topic": None, "text": None, "source_title": None}]
        result = rerank_chunks("NEO CITY là dự án gì?", chunks, classification={"intent": "project_overview"})
        assert result[0]["id"] == "1"

    def test_product_intent_prefers_product_chunk(self):
        chunks = [
            {"id": "1", "score": 0.8, "section": "concept_positioning", "topic": "brand_core", "text": "Câu chuyện thương hiệu"},
            {"id": "2", "score": 0.7, "section": "factsheet", "topic": "apartment_products", "text": "Loại căn 2PN và 3PN"},
        ]
        result = rerank_chunks("2PN diện tích bao nhiêu?", chunks, classification={"intent": "product", "target_sections": ["factsheet", "pricing"]})
        assert result[0]["id"] == "2"

    def test_location_intent_prefers_location_connectivity(self):
        chunks = [
            {"id": "1", "score": 0.8, "section": "market", "topic": "market_gap", "text": "Dư địa thị trường"},
            {"id": "2", "score": 0.7, "section": "location_connectivity", "topic": "transport_connectivity", "text": "Kết nối sân bay Nội Bài"},
        ]
        result = rerank_chunks("Mê Linh kết nối sân bay Nội Bài thế nào?", chunks, classification={"intent": "location", "target_sections": ["location_connectivity", "market"]})
        assert result[0]["id"] == "2"

    def test_market_intent_prefers_market(self):
        chunks = [
            {"id": "1", "score": 0.8, "section": "location_connectivity", "topic": "transport_connectivity", "text": "Kết nối giao thông"},
            {"id": "2", "score": 0.7, "section": "market", "topic": "market_overview", "text": "Thị trường Mê Linh có tiềm năng"},
        ]
        result = rerank_chunks("Tiềm năng thị trường Mê Linh ra sao?", chunks, classification={"intent": "market", "target_sections": ["market"]})
        assert result[0]["id"] == "2"

    # ------------------------------------------------------------------
    # Task 7 - High-impact failure pattern tests for reranker
    # ------------------------------------------------------------------

    def test_location_009_dong_anh_me_linh_ranks_location_above_market(self):
        """location_009: Pure location questions should rank location_connectivity above market."""
        chunks = [
            {"id": "1", "score": 0.82, "section": "market", "topic": "market_overview", "text": "Thị trường Mê Linh"},
            {"id": "2", "score": 0.75, "section": "location_connectivity", "topic": "transport_connectivity", "text": "Đông Anh và Mê Linh kết nối"},
        ]
        result = rerank_chunks(
            "Đông Anh và Mê Linh có gì khác nhau về vị trí?",
            chunks,
            classification={"intent": "location", "target_sections": ["location_connectivity", "market"]}
        )
        # location_connectivity should outrank market for pure location questions
        assert result[0]["id"] == "2"

    def test_pricing_penalizes_sales_policy_for_pure_price_questions(self):
        """Pricing questions should penalize sales_policy unless policy is explicitly asked."""
        chunks = [
            {"id": "1", "score": 0.82, "section": "sales_policy", "topic": "payment_policy", "text": "Chính sách thanh toán"},
            {"id": "2", "score": 0.74, "section": "pricing", "topic": "apartment_pricing", "text": "Giá căn hộ 2PN"},
        ]
        result = rerank_chunks(
            "Căn 2PN giá bao nhiêu?",
            chunks,
            classification={"intent": "pricing", "target_sections": ["pricing", "price_sheet"]}
        )
        # pricing should outrank sales_policy for pure price questions
        assert result[0]["id"] == "2"

    def test_sales_policy_prefers_sales_policy_over_pricing(self):
        """Sales policy questions should rank sales_policy above pricing."""
        chunks = [
            {"id": "1", "score": 0.82, "section": "pricing", "topic": "apartment_pricing", "text": "Giá căn hộ"},
            {"id": "2", "score": 0.76, "section": "sales_policy", "topic": "payment_policy", "text": "Chính sách thanh toán"},
        ]
        result = rerank_chunks(
            "Chính sách thanh toán thế nào?",
            chunks,
            classification={"intent": "sales_policy", "target_sections": ["sales_policy", "price_sheet"]}
        )
        # sales_policy should outrank pricing for policy questions
        assert result[0]["id"] == "2"

    def test_legal_heavily_penalizes_non_legal_chunks(self):
        """Legal questions should heavily penalize non-legal chunks."""
        chunks = [
            {"id": "1", "score": 0.85, "section": "factsheet", "topic": "project_overview", "text": "Tổng quan dự án"},
            {"id": "2", "score": 0.70, "section": "legal", "topic": "legal_status_and_warnings", "text": "Chưa mở bán, chưa nhận cọc"},
        ]
        result = rerank_chunks(
            "Dự án đã mở bán chưa?",
            chunks,
            classification={"intent": "legal", "target_sections": ["legal"], "must_use_legal_only": True, "risk_level": "critical"}
        )
        # legal should outrank factsheet even with lower base score
        assert result[0]["id"] == "2"

    def test_product_penalizes_pricing_principles_without_price_keywords(self):
        """Product questions without price keywords should penalize pricing_principles."""
        chunks = [
            {"id": "1", "score": 0.82, "section": "pricing", "topic": "pricing_principles", "text": "Nguyên tắc giá"},
            {"id": "2", "score": 0.75, "section": "factsheet", "topic": "apartment_products", "text": "Loại căn 2PN và 3PN"},
        ]
        result = rerank_chunks(
            "Dự án có những loại căn nào?",
            chunks,
            classification={"intent": "product", "target_sections": ["factsheet", "pricing"]}
        )
        # factsheet product should outrank pricing_principles
        assert result[0]["id"] == "2"

    def test_pricing_boosts_apartment_pricing_for_apartment_tokens(self):
        """Pricing questions with apartment tokens should boost apartment_pricing."""
        chunks = [
            {"id": "1", "score": 0.80, "section": "factsheet", "topic": "project_overview", "text": "Tổng quan"},
            {"id": "2", "score": 0.75, "section": "pricing", "topic": "apartment_pricing", "text": "Giá căn hộ 2PN"},
        ]
        result = rerank_chunks(
            "Căn 2PN giá bao nhiêu?",
            chunks,
            classification={"intent": "pricing", "target_sections": ["pricing", "price_sheet"]}
        )
        # apartment_pricing should be boosted
        assert result[0]["id"] == "2"

    def test_pricing_boosts_lowrise_pricing_for_lowrise_tokens(self):
        """Pricing questions with lowrise tokens should boost lowrise_pricing."""
        chunks = [
            {"id": "1", "score": 0.80, "section": "pricing", "topic": "apartment_pricing", "text": "Giá căn hộ"},
            {"id": "2", "score": 0.75, "section": "pricing", "topic": "lowrise_pricing", "text": "Giá shophouse"},
        ]
        result = rerank_chunks(
            "Shophouse giá bao nhiêu?",
            chunks,
            classification={"intent": "pricing", "target_sections": ["pricing", "price_sheet"]}
        )
        # lowrise_pricing should be boosted for shophouse questions
        assert result[0]["id"] == "2"

    def test_sales_policy_boosts_policy_topics(self):
        """Sales policy questions should boost payment_policy, booking_policy, etc."""
        chunks = [
            {"id": "1", "score": 0.82, "section": "pricing", "topic": "apartment_pricing", "text": "Giá căn hộ"},
            {"id": "2", "score": 0.76, "section": "sales_policy", "topic": "payment_policy", "text": "Chính sách thanh toán"},
        ]
        result = rerank_chunks(
            "Chính sách thanh toán thế nào?",
            chunks,
            classification={"intent": "sales_policy", "target_sections": ["sales_policy", "price_sheet"]}
        )
        # payment_policy should be boosted
        assert result[0]["id"] == "2"

    def test_sales_policy_penalizes_pricing_principles(self):
        """Sales policy questions should penalize pricing_principles."""
        chunks = [
            {"id": "1", "score": 0.82, "section": "pricing", "topic": "pricing_principles", "text": "Nguyên tắc giá"},
            {"id": "2", "score": 0.76, "section": "sales_policy", "topic": "payment_policy", "text": "Chính sách thanh toán"},
        ]
        result = rerank_chunks(
            "Chính sách thanh toán thế nào?",
            chunks,
            classification={"intent": "sales_policy", "target_sections": ["sales_policy", "price_sheet"]}
        )
        # payment_policy should outrank pricing_principles
        assert result[0]["id"] == "2"

    def test_sales_policy_prefers_sales_policy_or_price_sheet(self):
        chunks = [
            {"id": "1", "score": 0.8, "section": "pricing", "topic": "apartment_pricing", "text": "Giá căn hộ"},
            {"id": "2", "score": 0.7, "section": "sales_policy", "topic": "payment_policy", "text": "Chính sách thanh toán và chiết khấu"},
        ]
        result = rerank_chunks("Chính sách thanh toán thế nào?", chunks, classification={"intent": "sales_policy", "target_sections": ["sales_policy", "price_sheet"]})
        assert result[0]["id"] == "2"

    def test_concept_intent_prefers_concept_positioning(self):
        chunks = [
            {"id": "1", "score": 0.8, "section": "factsheet", "topic": "project_overview", "text": "Tổng quan dự án"},
            {"id": "2", "score": 0.7, "section": "concept_positioning", "topic": "brand_core", "text": "Brand core và concept"},
        ]
        result = rerank_chunks("Tagline của NEO CITY là gì?", chunks, classification={"intent": "concept", "target_sections": ["concept_positioning"]})
        assert result[0]["id"] == "2"

    def test_project_overview_prefers_factsheet_then_concept(self):
        chunks = [
            {"id": "1", "score": 0.8, "section": "concept_positioning", "topic": "brand_core", "text": "NEO CITY là đô thị tân sinh"},
            {"id": "2", "score": 0.7, "section": "factsheet", "topic": "project_overview", "text": "Tổng quan dự án NEO CITY"},
        ]
        result = rerank_chunks("NEO CITY là dự án gì?", chunks, classification={"intent": "project_overview", "target_sections": ["factsheet", "concept_positioning"]})
        assert result[0]["id"] == "2"


# ---------------------------------------------------------------------------
# SearchResult dataclass (legacy)
# ---------------------------------------------------------------------------

class TestSearchResultDataclass:
    def test_immutable(self):
        result = SearchResult(query="test", intent="general", section_filter=[], chunks=[])
        with pytest.raises((AttributeError, TypeError)):
            result.query = "changed"  # type: ignore[misc]

    def test_chunks_default_empty_requires_explicit(self):
        result = SearchResult(query="q", intent="legal", section_filter=["legal"], chunks=[])
        assert result.chunks == []


# ---------------------------------------------------------------------------
# Integration tests (skipped when Qdrant not available)
# ---------------------------------------------------------------------------

def _qdrant_available() -> bool:
    try:
        import requests
        r = requests.get("http://localhost:6333/healthz", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _qdrant_available(), reason="Qdrant not running at localhost:6333")
class TestSearchChunksIntegration:
    """Live integration tests — require Qdrant + populated neo_city_chunks collection."""

    def test_general_query_returns_chunks(self):
        result = search_chunks("NEO CITY dự án là gì?", "general", top_k=3)
        assert len(result.chunks) > 0

    def test_pricing_query_returns_pricing_sections(self):
        result = search_chunks("Giá bán dự kiến?", "pricing", top_k=5)
        if result.chunks:
            assert {c.section for c in result.chunks}.issubset({"pricing", "price_sheet"})

    def test_legal_query_returns_legal_sections(self):
        result = search_chunks("Pháp lý dự án", "legal", top_k=5)
        if result.chunks:
            assert {c.section for c in result.chunks}.issubset({"legal"})

    def test_scores_are_between_0_and_1(self):
        result = search_chunks("dự án NEO CITY", "general", top_k=5)
        for chunk in result.chunks:
            assert 0.0 <= chunk.score <= 1.0

    def test_chunks_have_required_fields(self):
        result = search_chunks("dự án NEO CITY", "general", top_k=3)
        required = {"id", "section", "topic", "text", "status", "legal_sensitivity"}
        for chunk in result.chunks:
            missing = required - set(chunk.payload.keys())
            assert not missing, f"Chunk missing fields: {missing}"


@pytest.mark.integration
@pytest.mark.skipif(not _qdrant_available(), reason="Qdrant not running at localhost:6333")
class TestRetrieveIntegration:
    """Live integration tests for retrieve()."""

    def test_pricing_query_retrieves_pricing(self):
        result = retrieve("Căn 2PN giá bao nhiêu?", limit=5)
        assert result["intent"] in ("pricing", "product")
        if result["chunks"]:
            for chunk in result["chunks"]:
                assert chunk["section"] in ("pricing", "price_sheet", "factsheet")

    def test_legal_query_retrieves_only_legal(self):
        result = retrieve("Dự án đã mở bán chưa?", limit=5)
        assert result["target_sections"] == ["legal"]
        for chunk in result["chunks"]:
            assert chunk["section"] == "legal"

    def test_profit_query_risk_critical(self):
        result = retrieve("Có cam kết lợi nhuận không?", limit=5)
        assert result["risk_level"] == "critical"
class TestTask7RerankPatterns:
    def test_product_area_query_can_prefer_pricing_product_chunk(self):
        chunks = [
            {"id": "1", "score": 0.82, "section": "factsheet", "topic": "apartment_products", "text": "Loai can studio, 1PN+1, 2PN"},
            {"id": "2", "score": 0.76, "section": "pricing", "topic": "studio_one_bedroom_policy", "text": "Studio va 1PN+1 dien tich du kien"},
        ]
        result = rerank_chunks(
            "Can 1PN+1 co dien tich bao nhieu?",
            chunks,
            classification={"intent": "product", "target_sections": ["factsheet", "pricing"]},
        )
        assert result[0]["id"] == "2"

    def test_product_handover_query_stays_on_factsheet_over_pricing_principles(self):
        chunks = [
            {"id": "1", "score": 0.81, "section": "pricing", "topic": "pricing_principles", "text": "Nguyen tac gia ban"},
            {"id": "2", "score": 0.74, "section": "factsheet", "topic": "apartment_products", "text": "Can ho ban giao hoan thien theo tieu chuan du kien"},
        ]
        result = rerank_chunks(
            "Can ho NEO CITY ban giao hoan thien hay tho?",
            chunks,
            classification={"intent": "product", "target_sections": ["factsheet", "pricing"]},
        )
        assert result[0]["id"] == "2"

    def test_sales_policy_query_prefers_price_sheet_discount_topic_over_pricing_principles(self):
        chunks = [
            {"id": "1", "score": 0.83, "section": "pricing", "topic": "pricing_principles", "text": "Nguyen tac gia"},
            {"id": "2", "score": 0.75, "section": "price_sheet", "topic": "supplemental_incentives", "text": "Chiet khau thanh toan nhanh va early bird"},
        ]
        result = rerank_chunks(
            "Chiet khau khi thanh toan som bao nhieu phan tram?",
            chunks,
            classification={"intent": "sales_policy", "target_sections": ["sales_policy", "price_sheet"]},
        )
        assert result[0]["id"] == "2"

    def test_amenity_policy_like_query_prefers_policy_combo_chunk(self):
        chunks = [
            {"id": "1", "score": 0.80, "section": "factsheet", "topic": "amenities", "text": "Co mam non va learning hub"},
            {"id": "2", "score": 0.74, "section": "sales_policy", "topic": "combo_family", "text": "Uu dai learning hub va mam non noi khu"},
        ]
        result = rerank_chunks(
            "Combo uu dai learning hub va mam non cua NEO CITY la gi?",
            chunks,
            classification={"intent": "sales_policy", "target_sections": ["sales_policy", "price_sheet"]},
        )
        assert result[0]["id"] == "2"
