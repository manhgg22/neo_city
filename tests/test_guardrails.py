"""Tests for app/guardrails.py — Task 4 safety guardrails."""

from __future__ import annotations

import pytest

from app.guardrails import (
    FALLBACK_ANSWER,
    INVESTMENT_RETURN_RESPONSE,
    LEGAL_CAUTION_FLAG,
    PRICING_CAUTION_FLAG,
    SALES_POLICY_CAUTION_FLAG,
    GuardrailResult,
    apply_guardrails,
    check_answer_safety,
    contains_prohibited_phrase,
    fallback_answer,
    filter_legal_chunks,
    investment_return_response,
    is_deposit_or_opening_question,
    is_profit_guarantee_question,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def legal_chunk(topic: str = "legal_status_and_warnings") -> dict:
    return {
        "id": f"neo_city_legal_{topic}_001",
        "section": "legal",
        "topic": topic,
        "text": "Dự án đang trong quá trình hoàn thiện hồ sơ pháp lý.",
        "legal_sensitivity": "critical",
    }


def pricing_chunk(topic: str = "price_overview") -> dict:
    return {
        "id": f"neo_city_pricing_{topic}_001",
        "section": "pricing",
        "topic": topic,
        "text": "Giá dự kiến từ 45 triệu đồng/m².",
        "legal_sensitivity": "high",
    }


def sales_policy_chunk(topic: str = "payment_schedule") -> dict:
    return {
        "id": f"neo_city_sales_policy_{topic}_001",
        "section": "sales_policy",
        "topic": topic,
        "text": "Thanh toán 15% khi ký HĐMB.",
        "legal_sensitivity": "high",
    }


def factsheet_chunk() -> dict:
    return {
        "id": "neo_city_factsheet_001",
        "section": "factsheet",
        "topic": "project_overview",
        "text": "NEO CITY là dự án khu đô thị tại Long An.",
        "legal_sensitivity": "medium",
    }


def legal_classification(risk_level: str = "critical") -> dict:
    return {"intent": "legal", "risk_level": risk_level, "must_use_legal_only": False}


def pricing_classification() -> dict:
    return {"intent": "pricing", "risk_level": "high", "must_use_legal_only": False}


def sales_policy_classification() -> dict:
    return {"intent": "sales_policy", "risk_level": "medium", "must_use_legal_only": False}


def general_classification() -> dict:
    return {"intent": "general", "risk_level": "low", "must_use_legal_only": False}


# ---------------------------------------------------------------------------
# is_profit_guarantee_question
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "Dự án có cam kết lợi nhuận không?",
        "Bạn có đảm bảo tăng giá không?",
        "Chắc chắn sinh lời khi mua ở đây không?",
        "guaranteed profit from this project?",
        "guaranteed return on investment?",
        "Dự án cam kết sinh lời bao nhiêu?",
        "Có cam kết tăng giá không?",
    ],
)
def test_profit_guarantee_question_detected(question: str) -> None:
    assert is_profit_guarantee_question(question), f"Expected True for: {question!r}"


@pytest.mark.parametrize(
    "question",
    [
        "Giá dự kiến của NEO CITY là bao nhiêu?",
        "Dự án có vị trí như thế nào?",
        "Khi nào mở bán?",
        "Lợi nhuận tiềm năng của khu vực Long An?",  # mentions lợi nhuận without guarantee
    ],
)
def test_profit_guarantee_question_not_triggered(question: str) -> None:
    assert not is_profit_guarantee_question(question), f"Expected False for: {question!r}"


# ---------------------------------------------------------------------------
# is_deposit_or_opening_question
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "Tôi có thể đặt cọc ngay bây giờ không?",
        "Dự án đã mở bán chưa?",
        "Có đủ điều kiện mở bán chưa?",
        "Dự án có được phép bán không?",
        "Nhận cọc chưa?",
        "Có huy động vốn không?",
        "Thu tiền đặt chỗ chưa?",
    ],
)
def test_deposit_or_opening_question_detected(question: str) -> None:
    assert is_deposit_or_opening_question(question), f"Expected True for: {question!r}"


@pytest.mark.parametrize(
    "question",
    [
        "Giá dự kiến là bao nhiêu?",
        "Vị trí dự án ở đâu?",
        "Chính sách thanh toán như thế nào?",
    ],
)
def test_deposit_or_opening_question_not_triggered(question: str) -> None:
    assert not is_deposit_or_opening_question(question), f"Expected False for: {question!r}"


# ---------------------------------------------------------------------------
# filter_legal_chunks
# ---------------------------------------------------------------------------


def test_filter_legal_chunks_returns_only_legal_section() -> None:
    chunks = [
        legal_chunk("legal_status_and_warnings"),
        pricing_chunk(),
        factsheet_chunk(),
        legal_chunk("land_use_rights"),
    ]
    result = filter_legal_chunks(chunks)
    assert all(c["section"] == "legal" for c in result)
    assert len(result) == 2


def test_filter_legal_chunks_puts_legal_status_first() -> None:
    chunks = [
        legal_chunk("land_use_rights"),
        legal_chunk("legal_status_and_warnings"),
        legal_chunk("other_legal"),
    ]
    result = filter_legal_chunks(chunks)
    assert result[0]["topic"] == "legal_status_and_warnings"


def test_filter_legal_chunks_legal_status_anchor_preferred_even_if_last() -> None:
    chunks = [
        legal_chunk("permits"),
        legal_chunk("construction_approval"),
        legal_chunk("legal_status_and_warnings"),
    ]
    result = filter_legal_chunks(chunks)
    assert result[0]["topic"] == "legal_status_and_warnings"


def test_filter_legal_chunks_empty_input() -> None:
    assert filter_legal_chunks([]) == []


def test_filter_legal_chunks_no_legal_chunks() -> None:
    assert filter_legal_chunks([pricing_chunk(), factsheet_chunk()]) == []


# ---------------------------------------------------------------------------
# contains_prohibited_phrase
# ---------------------------------------------------------------------------


def test_contains_prohibited_phrase_matches_case_insensitively() -> None:
    assert contains_prohibited_phrase("Giá Chính Thức Là 50 triệu", ("giá chính thức là",))


def test_contains_prohibited_phrase_no_match() -> None:
    assert not contains_prohibited_phrase("Giá dự kiến khoảng 50 triệu", ("giá chính thức là",))


# ---------------------------------------------------------------------------
# Helper public functions
# ---------------------------------------------------------------------------


def test_fallback_answer_matches_constant() -> None:
    assert fallback_answer() == FALLBACK_ANSWER


def test_investment_return_response_matches_constant() -> None:
    assert investment_return_response() == INVESTMENT_RETURN_RESPONSE


# ---------------------------------------------------------------------------
# apply_guardrails — Rule 1: Guaranteed profit → block_unsafe
# ---------------------------------------------------------------------------


def test_guaranteed_profit_question_blocked_with_no_chunks() -> None:
    result = apply_guardrails("Cam kết lợi nhuận bao nhiêu?", [])
    assert result.action == "block_unsafe"
    assert result.forced_response == INVESTMENT_RETURN_RESPONSE
    assert result.chunks == []
    assert result.caution_flags == []


def test_guaranteed_profit_question_blocked_even_with_chunks() -> None:
    chunks = [pricing_chunk(), legal_chunk()]
    result = apply_guardrails("Dự án có đảm bảo sinh lời không?", chunks)
    assert result.action == "block_unsafe"
    assert result.forced_response == INVESTMENT_RETURN_RESPONSE


def test_guaranteed_profit_block_returns_correct_reason() -> None:
    result = apply_guardrails("Cam kết tăng giá không?", [])
    assert result.reason == "guaranteed_profit_question"


# ---------------------------------------------------------------------------
# apply_guardrails — Rule 2: Legal intent → legal-only chunks
# ---------------------------------------------------------------------------


def test_legal_query_with_non_legal_chunks_gets_filtered() -> None:
    chunks = [pricing_chunk(), factsheet_chunk(), sales_policy_chunk()]
    result = apply_guardrails("Tình trạng pháp lý dự án?", chunks, legal_classification())
    assert result.action == "fallback"
    assert result.chunks == []
    assert result.reason == "legal_query_no_legal_chunks"


def test_legal_query_with_legal_chunks_allowed() -> None:
    chunks = [legal_chunk(), pricing_chunk()]
    result = apply_guardrails("Tình trạng pháp lý dự án?", chunks, legal_classification())
    assert result.action == "allow"
    assert all(c["section"] == "legal" for c in result.chunks)
    assert LEGAL_CAUTION_FLAG in result.caution_flags


def test_legal_query_insufficient_legal_chunks_fallback() -> None:
    chunks = [legal_chunk()]
    result = apply_guardrails(
        "Pháp lý dự án?", chunks, legal_classification(), min_chunks_required=2
    )
    assert result.action == "fallback"
    assert result.reason == "legal_query_no_legal_chunks"


def test_legal_status_and_warnings_chunk_is_preferred_anchor() -> None:
    chunks = [
        legal_chunk("construction_approval"),
        factsheet_chunk(),
        legal_chunk("legal_status_and_warnings"),
        pricing_chunk(),
    ]
    result = apply_guardrails("Pháp lý dự án ra sao?", chunks, legal_classification())
    assert result.action == "allow"
    assert result.chunks[0]["topic"] == "legal_status_and_warnings"


def test_must_use_legal_only_flag_enforces_legal_filter() -> None:
    clf = {"intent": "general", "risk_level": "low", "must_use_legal_only": True}
    chunks = [pricing_chunk(), factsheet_chunk()]
    result = apply_guardrails("Câu hỏi tổng quát", chunks, clf)
    assert result.action == "fallback"
    assert result.reason == "legal_query_no_legal_chunks"


def test_critical_risk_level_triggers_legal_filter() -> None:
    clf = {"intent": "general", "risk_level": "critical", "must_use_legal_only": False}
    chunks = [legal_chunk("legal_status_and_warnings"), pricing_chunk()]
    result = apply_guardrails("Câu hỏi", chunks, clf)
    assert result.action == "allow"
    assert all(c["section"] == "legal" for c in result.chunks)


# ---------------------------------------------------------------------------
# apply_guardrails — Rule 3: Deposit / opening / fundraising
# ---------------------------------------------------------------------------


def test_deposit_question_without_legal_chunks_fallback() -> None:
    chunks = [pricing_chunk(), sales_policy_chunk()]
    result = apply_guardrails("Tôi có thể đặt cọc ngay không?", chunks)
    assert result.action == "fallback"
    assert result.reason == "deposit_opening_question_no_legal_support"


def test_deposit_question_with_legal_chunks_allowed() -> None:
    chunks = [legal_chunk("legal_status_and_warnings"), pricing_chunk()]
    result = apply_guardrails("Tôi có thể đặt cọc ngay không?", chunks)
    assert result.action == "allow"
    assert all(c["section"] == "legal" for c in result.chunks)
    assert LEGAL_CAUTION_FLAG in result.caution_flags


def test_opening_sale_question_without_legal_fallback() -> None:
    chunks = [factsheet_chunk()]
    result = apply_guardrails("Dự án đã mở bán chưa?", chunks)
    assert result.action == "fallback"


def test_fundraising_question_without_legal_fallback() -> None:
    chunks = [sales_policy_chunk()]
    result = apply_guardrails("Dự án có đang huy động vốn không?", chunks)
    assert result.action == "fallback"
    assert result.reason == "deposit_opening_question_no_legal_support"


# ---------------------------------------------------------------------------
# apply_guardrails — Rule 4: Pricing → caution flags
# ---------------------------------------------------------------------------


def test_pricing_query_gets_caution_flag() -> None:
    chunks = [pricing_chunk()]
    result = apply_guardrails("Giá NEO CITY bao nhiêu?", chunks, pricing_classification())
    assert result.action == "allow"
    assert PRICING_CAUTION_FLAG in result.caution_flags
    assert result.forced_response is None


def test_pricing_query_no_chunks_fallback() -> None:
    result = apply_guardrails("Giá NEO CITY?", [], pricing_classification())
    assert result.action == "fallback"
    assert result.reason == "pricing_no_chunks"


def test_high_risk_level_triggers_pricing_caution() -> None:
    clf = {"intent": "general", "risk_level": "high", "must_use_legal_only": False}
    chunks = [pricing_chunk()]
    result = apply_guardrails("Câu hỏi về giá", chunks, clf)
    assert result.action == "allow"
    assert PRICING_CAUTION_FLAG in result.caution_flags


def test_pricing_caution_chunks_are_not_filtered() -> None:
    chunks = [pricing_chunk(), factsheet_chunk()]
    result = apply_guardrails("Giá dự kiến?", chunks, pricing_classification())
    assert result.action == "allow"
    assert len(result.chunks) == 2


# ---------------------------------------------------------------------------
# apply_guardrails — Rule 5: Sales policy → non-official caution
# ---------------------------------------------------------------------------


def test_sales_policy_query_gets_caution_flag() -> None:
    chunks = [sales_policy_chunk()]
    result = apply_guardrails(
        "Chính sách thanh toán như thế nào?", chunks, sales_policy_classification()
    )
    assert result.action == "allow"
    assert SALES_POLICY_CAUTION_FLAG in result.caution_flags
    assert PRICING_CAUTION_FLAG not in result.caution_flags


def test_sales_policy_query_no_chunks_fallback() -> None:
    result = apply_guardrails(
        "Chính sách bán hàng?", [], sales_policy_classification()
    )
    assert result.action == "fallback"
    assert result.reason == "sales_policy_no_chunks"


# ---------------------------------------------------------------------------
# apply_guardrails — Rule 6: Insufficient context fallback
# ---------------------------------------------------------------------------


def test_weak_context_returns_fallback() -> None:
    result = apply_guardrails("Câu hỏi chung", [], min_chunks_required=1)
    assert result.action == "fallback"
    assert result.reason == "insufficient_context"


def test_weak_context_with_min_chunks_two() -> None:
    result = apply_guardrails("Câu hỏi chung", [factsheet_chunk()], min_chunks_required=2)
    assert result.action == "fallback"
    assert result.reason == "insufficient_context"


# ---------------------------------------------------------------------------
# apply_guardrails — Default allow
# ---------------------------------------------------------------------------


def test_general_query_with_sufficient_chunks_allowed() -> None:
    chunks = [factsheet_chunk()]
    result = apply_guardrails("NEO CITY ở đâu?", chunks, general_classification())
    assert result.action == "allow"
    assert result.caution_flags == []
    assert result.forced_response is None
    assert result.reason == "allowed"


def test_no_classification_general_query_allowed() -> None:
    chunks = [factsheet_chunk()]
    result = apply_guardrails("NEO CITY ở đâu?", chunks)
    assert result.action == "allow"
    assert result.reason == "allowed"


# ---------------------------------------------------------------------------
# apply_guardrails — GuardrailResult structure
# ---------------------------------------------------------------------------


def test_guardrail_result_is_frozen() -> None:
    result = apply_guardrails("test", [factsheet_chunk()])
    with pytest.raises((AttributeError, TypeError)):
        result.action = "block_unsafe"  # type: ignore[misc]


def test_guardrail_result_block_unsafe_has_forced_response() -> None:
    result = apply_guardrails("Cam kết lợi nhuận không?", [])
    assert result.action == "block_unsafe"
    assert result.forced_response is not None
    assert len(result.forced_response) > 0


def test_guardrail_result_allow_has_no_forced_response() -> None:
    result = apply_guardrails("NEO CITY ở đâu?", [factsheet_chunk()])
    assert result.action == "allow"
    assert result.forced_response is None


# ---------------------------------------------------------------------------
# check_answer_safety
# ---------------------------------------------------------------------------


def test_check_answer_safety_detects_pricing_prohibited_phrase() -> None:
    answer = "Giá chính thức là 50 triệu đồng/m²."
    violations = check_answer_safety(answer, "pricing")
    assert any("pricing_prohibited" in v for v in violations)


def test_check_answer_safety_detects_legal_prohibited_phrase_for_legal() -> None:
    answer = "Dự án đã đủ điều kiện mở bán từ tháng 6."
    violations = check_answer_safety(answer, "legal")
    assert any("legal_prohibited" in v for v in violations)


def test_check_answer_safety_detects_legal_prohibited_phrase_for_pricing() -> None:
    answer = "Có thể đặt cọc ngay hôm nay."
    violations = check_answer_safety(answer, "pricing")
    assert any("legal_prohibited" in v for v in violations)


def test_check_answer_safety_clean_answer_no_violations() -> None:
    answer = "NEO CITY dự kiến ra mắt vào năm 2026 theo tài liệu định hướng."
    assert check_answer_safety(answer, "pricing") == []
    assert check_answer_safety(answer, "legal") == []


def test_check_answer_safety_general_intent_not_checked_for_pricing() -> None:
    answer = "Giá chính thức là 50 triệu."
    violations = check_answer_safety(answer, "general")
    assert violations == []


def test_check_answer_safety_sales_policy_checks_legal_phrases() -> None:
    answer = "Đã được phép huy động vốn từ quý 3."
    violations = check_answer_safety(answer, "sales_policy")
    assert any("legal_prohibited" in v for v in violations)
