"""Tests for app/answer.py — Task 6: corpus-aligned answer generation."""

from __future__ import annotations

import pytest

from app.answer import (
    AnswerResult,
    _compute_confidence,
    _extract_relevant_lines,
    answer_from_retrieval,
    chatbot_answer_from_retrieval,
    generate_answer,
)
from app.guardrails import (
    FALLBACK_ANSWER,
    INVESTMENT_RETURN_RESPONSE,
    LEGAL_CAUTION_FLAG,
    LEGAL_PROHIBITED_PHRASES,
    PRICING_CAUTION_FLAG,
    PRICING_PROHIBITED_PHRASES,
    SALES_POLICY_CAUTION_FLAG,
)

# ---------------------------------------------------------------------------
# Real corpus-derived fixture texts
# Source: data/processed/neo_city_chunks.jsonl
# ---------------------------------------------------------------------------

# Derived from neo_city_factsheet_012 (brochure_summary)
_FACTSHEET_CORPUS_TEXT = (
    "NEO CITY là khu đô thị thế hệ mới quy mô khoảng 60ha tại Mê Linh, "
    "được phát triển để đón làn sóng giãn dân khỏi trung tâm cũ của Hà Nội "
    "và phục vụ lớp cư dân trẻ, năng động, tư duy công nghệ."
)

# Derived from neo_city_pricing_002 (apartment_pricing)
_PRICING_CORPUS_TEXT = (
    "Khung giá bán dự kiến – sản phẩm căn hộ cao tầng:\n"
    "Studio+: 54–57 triệu đồng/m², khoảng 1,73–2,17 tỷ đồng/căn\n"
    "1PN+1: 51–54 triệu đồng/m², khoảng 2,15–2,70 tỷ đồng/căn\n"
    "2PN: 48–51 triệu đồng/m², khoảng 2,80–3,57 tỷ đồng/căn\n"
    "Định hướng giá: 42–57 triệu đồng/m², tùy tòa, tầng, view, thời điểm mở bán."
)

# Derived from neo_city_legal_004 (legal_status_and_warnings)
_LEGAL_CORPUS_TEXT = (
    "Tình trạng pháp lý hiện tại:\n"
    "Giai đoạn dự án: Đã được phê duyệt quy hoạch và xây dựng\n"
    "Tình trạng mở bán: Chưa mở bán\n"
    "Tình trạng huy động vốn: Chưa huy động vốn từ khách hàng\n"
    "Tình trạng triển khai xây dựng: Chưa triển khai xây dựng chính thức"
)

# Derived from neo_city_sales_policy_003 (payment_policy)
_SALES_POLICY_CORPUS_TEXT = (
    "Chính sách thanh toán chuẩn:\n"
    "Booking: 50 triệu đồng\n"
    "Ký HĐMB: 10% giá trị hợp đồng\n"
    "Hỗ trợ vay tới 70%, ân hạn nợ gốc 24 tháng\n"
    "Chiết khấu thanh toán nhanh 95%: 7–9%"
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _factsheet_chunk(topic: str = "brochure_summary", score: float = 0.72) -> dict:
    return {
        "id": f"neo_city_factsheet_{topic}_001",
        "section": "factsheet",
        "topic": topic,
        "text": _FACTSHEET_CORPUS_TEXT,
        "legal_sensitivity": "medium",
        "score": score,
        "rerank_score": score,
    }


def _pricing_chunk(topic: str = "apartment_pricing", score: float = 0.78) -> dict:
    return {
        "id": f"neo_city_pricing_{topic}_001",
        "section": "pricing",
        "topic": topic,
        "text": _PRICING_CORPUS_TEXT,
        "legal_sensitivity": "high",
        "score": score,
        "rerank_score": score,
    }


def _legal_chunk(topic: str = "legal_status_and_warnings", score: float = 0.75) -> dict:
    return {
        "id": f"neo_city_legal_{topic}_001",
        "section": "legal",
        "topic": topic,
        "text": _LEGAL_CORPUS_TEXT,
        "legal_sensitivity": "critical",
        "score": score,
        "rerank_score": score,
    }


def _sales_policy_chunk(topic: str = "payment_policy", score: float = 0.67) -> dict:
    return {
        "id": f"neo_city_sales_policy_{topic}_001",
        "section": "sales_policy",
        "topic": topic,
        "text": _SALES_POLICY_CORPUS_TEXT,
        "legal_sensitivity": "high",
        "score": score,
        "rerank_score": score,
    }


def _clf(intent: str, risk_level: str = "low", must_use_legal_only: bool = False) -> dict:
    return {
        "intent": intent,
        "risk_level": risk_level,
        "must_use_legal_only": must_use_legal_only,
        "target_sections": [],
    }


# ---------------------------------------------------------------------------
# AnswerResult structure
# ---------------------------------------------------------------------------


def test_answer_result_is_frozen() -> None:
    result = generate_answer("test", [_factsheet_chunk()])
    with pytest.raises((AttributeError, TypeError)):
        result.answer_mode = "fallback"  # type: ignore[misc]


def test_answer_result_has_all_fields() -> None:
    result = generate_answer("NEO CITY ở đâu?", [_factsheet_chunk()])
    assert hasattr(result, "answer_text")
    assert hasattr(result, "used_chunk_ids")
    assert hasattr(result, "used_sections")
    assert hasattr(result, "confidence")
    assert hasattr(result, "answer_mode")


# ---------------------------------------------------------------------------
# Project overview — real NEO CITY corpus facts
# ---------------------------------------------------------------------------


def test_overview_question_answered_from_factsheet() -> None:
    chunks = [_factsheet_chunk()]
    result = generate_answer("NEO CITY là dự án gì?", chunks, _clf("project_overview"))
    assert result.answer_mode == "answered"
    assert "NEO CITY" in result.answer_text


def test_overview_answer_uses_real_corpus_location_me_linh() -> None:
    """Answer must reference Mê Linh (the real location), not Long An."""
    chunks = [_factsheet_chunk()]
    result = generate_answer("NEO CITY ở đâu?", chunks, _clf("project_overview"))
    assert result.answer_mode == "answered"
    assert "Mê Linh" in result.answer_text


def test_overview_answer_uses_real_corpus_size_60ha() -> None:
    """Answer must reference ~60ha (the real size), not a fabricated number."""
    chunks = [_factsheet_chunk()]
    result = generate_answer("Quy mô dự án?", chunks, _clf("project_overview"))
    assert result.answer_mode == "answered"
    assert "60ha" in result.answer_text


def test_overview_answer_does_not_contain_fake_location() -> None:
    chunks = [_factsheet_chunk()]
    result = generate_answer("NEO CITY ở đâu?", chunks, _clf("project_overview"))
    assert "Long An" not in result.answer_text
    assert "Bến Lức" not in result.answer_text


def test_overview_answer_does_not_contain_fake_size() -> None:
    chunks = [_factsheet_chunk()]
    result = generate_answer("Quy mô?", chunks, _clf("project_overview"))
    assert "246ha" not in result.answer_text


def test_overview_answer_has_used_chunk_ids() -> None:
    chunk = _factsheet_chunk()
    result = generate_answer("NEO CITY?", [chunk], _clf("project_overview"))
    assert result.used_chunk_ids == [chunk["id"]]


def test_overview_answer_has_used_sections() -> None:
    chunk = _factsheet_chunk()
    result = generate_answer("NEO CITY?", [chunk], _clf("project_overview"))
    assert result.used_sections == ["factsheet"]


def test_overview_answer_has_positive_confidence() -> None:
    result = generate_answer("NEO CITY?", [_factsheet_chunk()], _clf("project_overview"))
    assert result.confidence > 0.0
    assert result.confidence <= 1.0


def test_multiple_factsheet_chunks_all_included() -> None:
    chunks = [
        _factsheet_chunk("brochure_summary", 0.80),
        _factsheet_chunk("project_overview", 0.70),
    ]
    result = generate_answer("NEO CITY?", chunks, _clf("project_overview"))
    assert result.answer_mode == "answered"
    assert len(result.used_chunk_ids) == 2


# ---------------------------------------------------------------------------
# Pricing — real corpus price ranges required
# ---------------------------------------------------------------------------


def test_pricing_answer_uses_real_corpus_price_ranges() -> None:
    """Answer must reference real pricing (54–57 triệu range), not fabricated prices."""
    chunks = [_pricing_chunk()]
    result = generate_answer("Giá NEO CITY bao nhiêu?", chunks, _clf("pricing", "high"))
    assert result.answer_mode == "answered"
    # Real corpus has 54–57 triệu for Studio+, overall range 42–57 triệu
    assert any(marker in result.answer_text for marker in ["54", "57", "42–57"])


def test_pricing_answer_does_not_contain_fake_price() -> None:
    chunks = [_pricing_chunk()]
    result = generate_answer("Giá NEO CITY?", chunks, _clf("pricing", "high"))
    assert "35 triệu" not in result.answer_text


def test_pricing_answer_includes_caution_flag_text() -> None:
    chunks = [_pricing_chunk()]
    result = generate_answer("Giá NEO CITY bao nhiêu?", chunks, _clf("pricing", "high"))
    assert result.answer_mode == "answered"
    assert PRICING_CAUTION_FLAG in result.answer_text


def test_pricing_answer_contains_cautious_wording() -> None:
    chunks = [_pricing_chunk()]
    result = generate_answer("Giá NEO CITY?", chunks, _clf("pricing", "high"))
    assert "tài liệu định hướng" in result.answer_text


def test_pricing_answer_does_not_contain_prohibited_phrases() -> None:
    chunks = [_pricing_chunk()]
    result = generate_answer("Giá NEO CITY?", chunks, _clf("pricing", "high"))
    answer_lower = result.answer_text.lower()
    for phrase in PRICING_PROHIBITED_PHRASES:
        assert phrase.lower() not in answer_lower, (
            f"Pricing answer must not contain prohibited phrase: {phrase!r}"
        )


def test_pricing_answer_does_not_contain_legal_prohibited_phrases() -> None:
    chunks = [_pricing_chunk()]
    result = generate_answer("Giá NEO CITY?", chunks, _clf("pricing", "high"))
    answer_lower = result.answer_text.lower()
    for phrase in LEGAL_PROHIBITED_PHRASES:
        assert phrase.lower() not in answer_lower


def test_pricing_answer_includes_chunk_id_and_section() -> None:
    chunk = _pricing_chunk()
    result = generate_answer("Giá NEO CITY?", [chunk], _clf("pricing", "high"))
    assert chunk["id"] in result.used_chunk_ids
    assert "pricing" in result.used_sections


# ---------------------------------------------------------------------------
# Legal — real corpus legal wording required
# ---------------------------------------------------------------------------


def test_legal_answer_uses_real_legal_status_wording() -> None:
    """Answer must reflect the real legal status (Chưa mở bán), not fabricated text."""
    chunks = [_legal_chunk()]
    result = generate_answer(
        "Tình trạng pháp lý dự án?", chunks, _clf("legal", "critical")
    )
    assert result.answer_mode == "answered"
    assert "Chưa mở bán" in result.answer_text


def test_legal_answer_includes_fundraising_status() -> None:
    chunks = [_legal_chunk()]
    result = generate_answer(
        "Pháp lý NEO CITY?", chunks, _clf("legal", "critical")
    )
    assert "Chưa huy động vốn" in result.answer_text


def test_legal_answer_with_clear_legal_chunk_is_allowed() -> None:
    chunks = [_legal_chunk(), _factsheet_chunk()]
    result = generate_answer(
        "Tình trạng pháp lý dự án?", chunks, _clf("legal", "critical")
    )
    assert result.answer_mode == "answered"
    # Non-legal chunks must be filtered out by guardrails
    assert all(sec == "legal" for sec in result.used_sections)


def test_legal_answer_includes_legal_caution_flag() -> None:
    chunks = [_legal_chunk()]
    result = generate_answer(
        "Pháp lý NEO CITY?", chunks, _clf("legal", "critical")
    )
    assert LEGAL_CAUTION_FLAG in result.answer_text


def test_legal_answer_does_not_emit_prohibited_phrases() -> None:
    chunks = [_legal_chunk()]
    result = generate_answer(
        "Pháp lý NEO CITY?", chunks, _clf("legal", "critical")
    )
    answer_lower = result.answer_text.lower()
    for phrase in LEGAL_PROHIBITED_PHRASES:
        assert phrase.lower() not in answer_lower, (
            f"Legal answer must not contain prohibited phrase: {phrase!r}"
        )


def test_legal_answer_only_uses_legal_chunks() -> None:
    chunks = [
        _legal_chunk("legal_status_and_warnings"),
        _pricing_chunk(),
        _factsheet_chunk(),
    ]
    result = generate_answer(
        "Pháp lý dự án?", chunks, _clf("legal", "critical")
    )
    assert result.answer_mode == "answered"
    for chunk_id in result.used_chunk_ids:
        assert "legal" in chunk_id


# ---------------------------------------------------------------------------
# Legal question with weak/partial support → fallback
# ---------------------------------------------------------------------------


def test_legal_question_no_legal_chunks_falls_back() -> None:
    chunks = [_factsheet_chunk(), _pricing_chunk()]
    result = generate_answer(
        "Tình trạng pháp lý?", chunks, _clf("legal", "critical")
    )
    assert result.answer_mode == "fallback"
    assert result.answer_text == FALLBACK_ANSWER
    assert result.used_chunk_ids == []
    assert result.confidence == 0.0


def test_legal_question_insufficient_legal_chunks_falls_back() -> None:
    chunks = [_legal_chunk()]
    result = generate_answer(
        "Pháp lý dự án?", chunks, _clf("legal", "critical"), min_chunks_required=2
    )
    assert result.answer_mode == "fallback"
    assert result.answer_text == FALLBACK_ANSWER


# ---------------------------------------------------------------------------
# Sales policy — real corpus policy wording required
# ---------------------------------------------------------------------------


def test_sales_policy_answer_uses_real_corpus_payment_wording() -> None:
    """Answer must reference real policy (HĐMB 10%), not fabricated terms."""
    chunks = [_sales_policy_chunk()]
    result = generate_answer(
        "Chính sách thanh toán?", chunks, _clf("sales_policy", "medium")
    )
    assert result.answer_mode == "answered"
    # Real corpus: Ký HĐMB 10%, booking 50 triệu, vay 70%
    assert "HĐMB" in result.answer_text or "10%" in result.answer_text


def test_sales_policy_answer_does_not_contain_fake_payment_term() -> None:
    chunks = [_sales_policy_chunk()]
    result = generate_answer(
        "Chính sách?", chunks, _clf("sales_policy", "medium")
    )
    # The old fake fixture had "15% khi ký HĐMB" — real corpus says 10%
    assert "15% khi ký HĐMB" not in result.answer_text


def test_sales_policy_answer_includes_caution_flag() -> None:
    chunks = [_sales_policy_chunk()]
    result = generate_answer(
        "Chính sách thanh toán?", chunks, _clf("sales_policy", "medium")
    )
    assert result.answer_mode == "answered"
    assert SALES_POLICY_CAUTION_FLAG in result.answer_text


def test_sales_policy_answer_contains_non_official_wording() -> None:
    chunks = [_sales_policy_chunk()]
    result = generate_answer(
        "Chính sách?", chunks, _clf("sales_policy", "medium")
    )
    # "dự kiến" appears in SALES_POLICY_CAUTION_FLAG and _SALES_POLICY_INTRO
    assert "dự kiến" in result.answer_text


def test_sales_policy_answer_does_not_emit_legal_prohibited_phrases() -> None:
    chunks = [_sales_policy_chunk()]
    result = generate_answer(
        "Chính sách bán hàng?", chunks, _clf("sales_policy", "medium")
    )
    answer_lower = result.answer_text.lower()
    for phrase in LEGAL_PROHIBITED_PHRASES:
        assert phrase.lower() not in answer_lower


# ---------------------------------------------------------------------------
# Guaranteed profit → blocked
# ---------------------------------------------------------------------------


def test_guaranteed_profit_returns_blocked() -> None:
    result = generate_answer("Cam kết lợi nhuận bao nhiêu?", [_pricing_chunk()])
    assert result.answer_mode == "blocked"


def test_guaranteed_profit_answer_text_is_investment_return_response() -> None:
    result = generate_answer("Có cam kết tăng giá không?", [])
    assert result.answer_text == INVESTMENT_RETURN_RESPONSE


def test_guaranteed_profit_no_chunks_in_result() -> None:
    result = generate_answer("Đảm bảo sinh lời không?", [_factsheet_chunk()])
    assert result.used_chunk_ids == []
    assert result.used_sections == []
    assert result.confidence == 0.0


def test_guaranteed_profit_blocked_even_with_many_chunks() -> None:
    chunks = [_factsheet_chunk(), _pricing_chunk(), _legal_chunk()]
    result = generate_answer("Dự án cam kết lợi nhuận?", chunks)
    assert result.answer_mode == "blocked"


# ---------------------------------------------------------------------------
# Deposit / opening / fundraising → safely handled
# ---------------------------------------------------------------------------


def test_deposit_question_without_legal_chunks_falls_back() -> None:
    chunks = [_pricing_chunk(), _factsheet_chunk()]
    result = generate_answer("Tôi có thể đặt cọc không?", chunks)
    assert result.answer_mode == "fallback"
    assert result.answer_text == FALLBACK_ANSWER


def test_deposit_question_with_legal_chunks_allowed() -> None:
    chunks = [_legal_chunk("legal_status_and_warnings"), _pricing_chunk()]
    result = generate_answer("Có thể đặt cọc không?", chunks)
    assert result.answer_mode == "answered"
    assert all(sec == "legal" for sec in result.used_sections)


def test_opening_sale_question_without_legal_falls_back() -> None:
    chunks = [_factsheet_chunk()]
    result = generate_answer("Dự án đã mở bán chưa?", chunks)
    assert result.answer_mode == "fallback"


def test_fundraising_question_without_legal_falls_back() -> None:
    chunks = [_sales_policy_chunk()]
    result = generate_answer("Dự án đang huy động vốn không?", chunks)
    assert result.answer_mode == "fallback"


def test_deposit_answer_does_not_emit_prohibited_phrases() -> None:
    chunks = [_legal_chunk()]
    result = generate_answer("Có thể đặt cọc không?", chunks)
    if result.answer_mode == "answered":
        answer_lower = result.answer_text.lower()
        for phrase in LEGAL_PROHIBITED_PHRASES:
            assert phrase.lower() not in answer_lower


# ---------------------------------------------------------------------------
# Empty chunks → fallback
# ---------------------------------------------------------------------------


def test_empty_chunks_returns_fallback() -> None:
    result = generate_answer("NEO CITY?", [])
    assert result.answer_mode == "fallback"
    assert result.answer_text == FALLBACK_ANSWER


def test_empty_chunks_confidence_is_zero() -> None:
    result = generate_answer("NEO CITY?", [])
    assert result.confidence == 0.0


def test_empty_chunks_ids_and_sections_empty() -> None:
    result = generate_answer("NEO CITY?", [])
    assert result.used_chunk_ids == []
    assert result.used_sections == []


# ---------------------------------------------------------------------------
# Output schema: used_chunk_ids, used_sections, confidence
# ---------------------------------------------------------------------------


def test_answered_result_used_chunk_ids_populated() -> None:
    chunks = [_factsheet_chunk("a", 0.70), _factsheet_chunk("b", 0.65)]
    result = generate_answer("NEO CITY?", chunks, _clf("project_overview"))
    assert result.answer_mode == "answered"
    assert len(result.used_chunk_ids) == 2


def test_answered_result_used_sections_deduplicated() -> None:
    chunks = [_factsheet_chunk("a"), _factsheet_chunk("b")]
    result = generate_answer("NEO CITY?", chunks, _clf("project_overview"))
    assert result.used_sections == ["factsheet"]


def test_answered_result_confidence_between_0_and_1() -> None:
    chunks = [_factsheet_chunk(score=0.65), _factsheet_chunk("b", 0.60)]
    result = generate_answer("NEO CITY?", chunks, _clf("project_overview"))
    assert 0.0 <= result.confidence <= 1.0


def test_fallback_result_confidence_is_zero() -> None:
    result = generate_answer("NEO CITY?", [])
    assert result.confidence == 0.0


def test_blocked_result_confidence_is_zero() -> None:
    result = generate_answer("Cam kết lợi nhuận?", [])
    assert result.confidence == 0.0


def test_max_three_chunks_in_answer() -> None:
    chunks = [_factsheet_chunk(str(i), 0.80 - i * 0.05) for i in range(5)]
    result = generate_answer("NEO CITY?", chunks, _clf("project_overview"))
    assert len(result.used_chunk_ids) <= 3


# ---------------------------------------------------------------------------
# answer_from_retrieval convenience wrapper
# ---------------------------------------------------------------------------


def test_answer_from_retrieval_fallback_for_no_chunks() -> None:
    retrieval = {
        "question": "NEO CITY?",
        "intent": "project_overview",
        "risk_level": "low",
        "must_use_legal_only": False,
        "target_sections": ["factsheet"],
        "chunks": [],
        "reason": "no_chunks_above_min_score",
    }
    result = answer_from_retrieval(retrieval)
    assert result.answer_mode == "fallback"
    assert result.answer_text == FALLBACK_ANSWER


def test_answer_from_retrieval_answered_for_factsheet_chunks() -> None:
    retrieval = {
        "question": "NEO CITY ở đâu?",
        "intent": "project_overview",
        "risk_level": "low",
        "must_use_legal_only": False,
        "target_sections": ["factsheet"],
        "chunks": [_factsheet_chunk()],
        "reason": None,
    }
    result = answer_from_retrieval(retrieval)
    assert result.answer_mode == "answered"
    assert result.used_chunk_ids != []


def test_answer_from_retrieval_uses_real_location_in_answer() -> None:
    retrieval = {
        "question": "NEO CITY ở đâu?",
        "intent": "project_overview",
        "risk_level": "low",
        "must_use_legal_only": False,
        "target_sections": ["factsheet"],
        "chunks": [_factsheet_chunk()],
        "reason": None,
    }
    result = answer_from_retrieval(retrieval)
    assert "Mê Linh" in result.answer_text


def test_answer_from_retrieval_blocked_for_profit_question() -> None:
    retrieval = {
        "question": "Cam kết lợi nhuận?",
        "intent": "market",
        "risk_level": "medium",
        "must_use_legal_only": False,
        "target_sections": ["market"],
        "chunks": [_factsheet_chunk()],
        "reason": None,
    }
    result = answer_from_retrieval(retrieval)
    assert result.answer_mode == "blocked"
    assert result.answer_text == INVESTMENT_RETURN_RESPONSE


def test_answer_from_retrieval_legal_falls_back_without_legal_chunks() -> None:
    retrieval = {
        "question": "Tình trạng pháp lý?",
        "intent": "legal",
        "risk_level": "critical",
        "must_use_legal_only": False,
        "target_sections": ["legal"],
        "chunks": [_factsheet_chunk(), _pricing_chunk()],
        "reason": None,
    }
    result = answer_from_retrieval(retrieval)
    assert result.answer_mode == "fallback"


def test_answer_from_retrieval_pricing_includes_caution() -> None:
    retrieval = {
        "question": "Giá NEO CITY?",
        "intent": "pricing",
        "risk_level": "high",
        "must_use_legal_only": False,
        "target_sections": ["pricing"],
        "chunks": [_pricing_chunk()],
        "reason": None,
    }
    result = answer_from_retrieval(retrieval)
    assert result.answer_mode == "answered"
    assert PRICING_CAUTION_FLAG in result.answer_text


# ---------------------------------------------------------------------------
# _extract_relevant_lines unit tests
# ---------------------------------------------------------------------------


def test_extract_relevant_lines_short_text_returned_verbatim() -> None:
    text = "NEO CITY tại Mê Linh."
    assert _extract_relevant_lines(text, "NEO CITY ở đâu?") == text


def test_extract_relevant_lines_long_text_truncated() -> None:
    # Build a text definitely longer than _MAX_PASSAGE_CHARS (380)
    long_text = "\n".join(f"Đây là dòng số {i} không liên quan đến câu hỏi." for i in range(30))
    result = _extract_relevant_lines(long_text, "Studio giá bao nhiêu?")
    assert len(result) <= 380 + 100  # allow some overshoot for the first mandatory line


def test_extract_relevant_lines_prefers_question_keyword_lines() -> None:
    text = (
        "Dự án không liên quan đến câu hỏi.\n"
        "Studio+: 54–57 triệu đồng/m².\n"
        "Đây cũng không liên quan.\n"
        "1PN+1: 51–54 triệu đồng/m²."
    )
    result = _extract_relevant_lines(text * 5, "giá studio bao nhiêu")
    assert "Studio" in result or "studio" in result.lower()


def test_extract_relevant_lines_preserves_original_order() -> None:
    # Unique lines long enough to trigger extraction (total > 380 chars).
    unique_lines = [
        f"Dòng {i:02d}: NEO CITY thông tin mô tả chi tiết số {i} không trùng lặp."
        for i in range(15)
    ]
    text = "\n".join(unique_lines)
    assert len(text) > 380  # must trigger extraction

    result = _extract_relevant_lines(text, "giá studio")
    result_lines = [ln for ln in result.splitlines() if ln.strip()]

    # Each selected line should appear in the original list
    # and the selected indices must be in ascending order (original order preserved).
    idx_map = {ln: i for i, ln in enumerate(unique_lines)}
    selected_indices = [idx_map[ln] for ln in result_lines if ln in idx_map]
    assert selected_indices == sorted(selected_indices)


def test_extract_relevant_lines_empty_question_returns_content() -> None:
    text = "A\n" * 200  # long enough to trigger extraction
    result = _extract_relevant_lines(text, "")
    assert len(result) > 0


# ---------------------------------------------------------------------------
# _compute_confidence unit tests
# ---------------------------------------------------------------------------


def test_compute_confidence_zero_for_empty_chunks() -> None:
    assert _compute_confidence([]) == 0.0


def test_compute_confidence_uses_rerank_score() -> None:
    chunks = [{"section": "factsheet", "rerank_score": 0.80}]
    # Single chunk: avg=0.80, section_bonus=+0.05, count_penalty=-0.10 → 0.75
    conf = _compute_confidence(chunks)
    assert conf == pytest.approx(0.75, abs=0.01)


def test_compute_confidence_section_consistency_bonus() -> None:
    chunks = [
        {"section": "pricing", "rerank_score": 0.60},
        {"section": "pricing", "rerank_score": 0.60},
    ]
    conf_consistent = _compute_confidence(chunks)

    chunks_mixed = [
        {"section": "pricing", "rerank_score": 0.60},
        {"section": "factsheet", "rerank_score": 0.60},
    ]
    conf_mixed = _compute_confidence(chunks_mixed)

    assert conf_consistent > conf_mixed


def test_compute_confidence_clamped_to_one() -> None:
    chunks = [
        {"section": "factsheet", "rerank_score": 0.98},
        {"section": "factsheet", "rerank_score": 0.98},
    ]
    assert _compute_confidence(chunks) <= 1.0


def test_compute_confidence_falls_back_to_score_field() -> None:
    chunks = [{"section": "factsheet", "score": 0.55}, {"section": "factsheet", "score": 0.55}]
    conf = _compute_confidence(chunks)
    assert conf > 0.0


# ---------------------------------------------------------------------------
# chatbot_answer_from_retrieval demo mode
# ---------------------------------------------------------------------------


def test_chatbot_answer_from_retrieval_legal_is_concise() -> None:
    retrieval = {
        "question": "Dự án đã mở bán chưa?",
        "intent": "legal",
        "risk_level": "critical",
        "must_use_legal_only": True,
        "target_sections": ["legal"],
        "chunks": [
            {
                "id": "neo_city_legal_004",
                "section": "legal",
                "topic": "legal_status_and_warnings",
                "text": (
                    "Tình trạng mở bán: Chưa mở bán\n"
                    "Tình trạng huy động vốn: Chưa huy động vốn từ khách hàng\n"
                    "Tài liệu này chỉ phục vụ mục đích giới thiệu định hướng phát triển dự án."
                ),
                "score": 0.4,
                "rerank_score": 1.9,
            }
        ],
        "reason": None,
    }
    answer = chatbot_answer_from_retrieval(retrieval)
    assert "chưa mở bán chính thức" in answer.lower()
    assert "thông báo giao dịch chính thức" in answer.lower()
    assert "|" not in answer


def test_chatbot_answer_from_retrieval_pricing_is_concise() -> None:
    retrieval = {
        "question": "Giá Studio+ dự kiến bao nhiêu?",
        "intent": "pricing",
        "risk_level": "high",
        "must_use_legal_only": False,
        "target_sections": ["pricing", "price_sheet"],
        "chunks": [
            {
                "id": "neo_city_pricing_002",
                "section": "pricing",
                "topic": "apartment_pricing",
                "text": (
                    "Giá bán dự kiến\n"
                    "Studio+: 54–57 triệu đồng/m²\n"
                    "Định hướng giá: 42–57 triệu đồng/m², tùy tòa, tầng, view, thời điểm mở bán."
                ),
                "score": 0.6,
                "rerank_score": 1.7,
            }
        ],
        "reason": None,
    }
    answer = chatbot_answer_from_retrieval(retrieval)
    assert "chưa phải giá bán chính thức" in answer.lower()
    assert "54–57 triệu đồng/m²" in answer
    assert "|" not in answer


def test_chatbot_answer_from_retrieval_pricing_avoids_header_only_output() -> None:
    retrieval = {
        "question": "Giá căn 2PN dự kiến bao nhiêu?",
        "intent": "pricing",
        "risk_level": "high",
        "must_use_legal_only": False,
        "target_sections": ["pricing", "price_sheet"],
        "chunks": [
            {
                "id": "neo_city_pricing_002",
                "section": "pricing",
                "topic": "apartment_pricing",
                "text": (
                    "2. Khung giá bán dự kiến\n"
                    "| Loại sản phẩm | Diện tích dự kiến | Giá bán dự kiến | Tổng giá trị dự kiến |\n"
                    "Studio+: 54–57 triệu đồng/m²\n"
                    "2PN: 48–51 triệu đồng/m², khoảng 2,80–3,57 tỷ đồng/căn\n"
                    "Định hướng giá: 42–57 triệu đồng/m², tùy tòa, tầng, view, thời điểm mở bán."
                ),
                "score": 0.62,
                "rerank_score": 1.8,
            }
        ],
        "reason": None,
    }
    answer = chatbot_answer_from_retrieval(retrieval)
    assert "48–51 triệu đồng/m²" in answer
    assert "| Loại sản phẩm |" not in answer
    assert "| 2PN |" not in answer


def test_chatbot_answer_from_retrieval_pricing_uses_only_requested_product_line() -> None:
    retrieval = {
        "question": "Giá căn 2PN+1 dự kiến bao nhiêu?",
        "intent": "pricing",
        "risk_level": "high",
        "must_use_legal_only": False,
        "target_sections": ["pricing", "price_sheet"],
        "chunks": [
            {
                "id": "neo_city_pricing_002",
                "section": "pricing",
                "topic": "apartment_pricing",
                "text": (
                    "2. Khung giá bán dự kiến\n"
                    "2PN: 48–51 triệu đồng/m², khoảng 2,80–3,57 tỷ đồng/căn\n"
                    "2PN+1: 45–48 triệu đồng/m², khoảng 3,15–3,84 tỷ đồng/căn\n"
                    "Định hướng giá: 42–57 triệu đồng/m², tùy tòa, tầng, view, thời điểm mở bán."
                ),
                "score": 0.62,
                "rerank_score": 1.8,
            }
        ],
        "reason": None,
    }
    answer = chatbot_answer_from_retrieval(retrieval)
    assert "2PN+1: 45–48 triệu đồng/m²" in answer
    assert "2PN: 48–51 triệu đồng/m²" not in answer


def test_chatbot_answer_from_retrieval_blocked_passthrough() -> None:
    retrieval = {
        "question": "Có cam kết lợi nhuận không?",
        "intent": "market",
        "risk_level": "medium",
        "must_use_legal_only": False,
        "target_sections": ["market"],
        "chunks": [],
        "reason": None,
    }
    answer = chatbot_answer_from_retrieval(retrieval)
    assert answer == INVESTMENT_RETURN_RESPONSE
