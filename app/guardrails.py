"""app/guardrails.py

Safety guardrails for NEO CITY RAG retrieval results.

Public API
----------
apply_guardrails(question, chunks, classification) -> GuardrailResult
    Inspect retriever output and decide whether it is safe to pass to answer
    generation.  Returns a GuardrailResult with:
      - action        : "allow" | "fallback" | "block_unsafe"
      - chunks        : filtered, safe-to-use chunk list
      - caution_flags : warnings the answer generator must include
      - forced_response : verbatim answer override (only when block_unsafe)
      - reason        : diagnostic string for logging

Helper predicates (also exposed for tests)
------------------------------------------
is_profit_guarantee_question(question) -> bool
is_deposit_or_opening_question(question) -> bool
filter_legal_chunks(chunks) -> list[dict]
contains_prohibited_phrase(text, phrases) -> bool
fallback_answer() -> str
investment_return_response() -> str
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Verbatim strings from AGENTS.md (do not paraphrase)
# ---------------------------------------------------------------------------

PRICING_CAUTIOUS_PHRASES = (
    "theo tài liệu định hướng hiện tại",
    "dự kiến",
    "tùy tòa, tầng, view, thời điểm mở bán và chính sách từng đợt",
    "chưa phải giá bán chính thức nếu chưa có công bố chính thức",
)

PRICING_PROHIBITED_PHRASES = (
    "giá chính thức là",
    "chắc chắn giá là",
    "cam kết mức giá",
)

LEGAL_PROHIBITED_PHRASES = (
    "đã đủ điều kiện mở bán",
    "có thể đặt cọc ngay",
    "đã được phép huy động vốn",
)

INVESTMENT_RETURN_RESPONSE = (
    "Tài liệu hiện tại không đưa ra cam kết lợi nhuận. Các luận điểm về hạ tầng, "
    "thị trường và xu hướng giãn dân chỉ nên được hiểu là cơ sở tham khảo, "
    "không phải cam kết tăng giá hoặc cam kết sinh lời."
)

FALLBACK_ANSWER = (
    "Tôi chưa tìm thấy dữ liệu đủ rõ trong tài liệu NEO CITY hiện tại để trả lời "
    "chính xác câu hỏi này."
)

# Caution flags injected into the answer when the action is "allow"
PRICING_CAUTION_FLAG = (
    "Giá được trích từ tài liệu định hướng, chưa phải giá bán chính thức. "
    "Số liệu có thể thay đổi tùy tòa, tầng, view, thời điểm mở bán và chính sách từng đợt."
)

SALES_POLICY_CAUTION_FLAG = (
    "Chính sách bán hàng được trích từ tài liệu dự kiến và chưa có xác nhận chính thức. "
    "Thông tin có thể thay đổi khi dự án chính thức mở bán."
)

LEGAL_CAUTION_FLAG = (
    "Thông tin pháp lý chỉ lấy từ mục tình trạng pháp lý. "
    "Không suy diễn từ tài liệu marketing hoặc kinh doanh."
)

# Section/topic constants
_LEGAL_SECTION = "legal"
_LEGAL_STATUS_TOPIC = "legal_status_and_warnings"
_PRICING_SECTIONS = frozenset({"pricing", "price_sheet"})
_SALES_POLICY_SECTIONS = frozenset({"sales_policy", "price_sheet"})

# ---------------------------------------------------------------------------
# Keyword lists for guardrail question analysis (normalized — no diacritics)
# ---------------------------------------------------------------------------

_PROFIT_GUARANTEE_KEYWORDS: tuple[str, ...] = (
    "cam ket loi nhuan",
    "cam ket tang gia",
    "cam ket sinh loi",
    "dam bao loi nhuan",
    "dam bao tang gia",
    "dam bao sinh loi",
    "chac chan lai",
    "chac chan tang",
    "chac thang",
    "guaranteed profit",
    "guaranteed return",
)
_PROFIT_CORE: tuple[str, ...] = ("loi nhuan", "sinh loi", "tang gia", "loi tuc")
_GUARANTEE_CORE: tuple[str, ...] = (
    "dam bao", "bao dam", "cam ket", "chac chan", "chac thang", "chac an",
)

_DEPOSIT_KEYWORDS: tuple[str, ...] = (
    "dat coc",
    "nhan coc",
    "co the dat coc",
    "dat cho",
)
_OPENING_KEYWORDS: tuple[str, ...] = (
    "mo ban",
    "du dieu kien mo ban",
    "chinh thuc mo ban",
    "duoc ban chua",
    "duoc phep ban",
)
_FUNDRAISING_KEYWORDS: tuple[str, ...] = (
    "huy dong von",
    "thu tien",
    "duoc thu tien",
    "nhan tien",
)


# ---------------------------------------------------------------------------
# Normalization helper
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Lowercase, strip diacritics (including đ/Đ), collapse whitespace."""
    text = text.replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", stripped.lower().strip())


def _any_match(norm: str, keywords: tuple[str, ...] | list[str]) -> bool:
    return any(kw in norm for kw in keywords)


# ---------------------------------------------------------------------------
# Classification extraction helper
# ---------------------------------------------------------------------------


def _extract_classification(
    classification: Any,
) -> tuple[str, str, bool]:
    """Return (intent, risk_level, must_use_legal_only) from a classification."""
    if classification is None:
        return "", "low", False
    if isinstance(classification, dict):
        return (
            str(classification.get("intent", "") or ""),
            str(classification.get("risk_level", "low") or "low"),
            bool(classification.get("must_use_legal_only", False)),
        )
    # ClassificationResult dataclass
    return (
        str(getattr(classification, "intent", "") or ""),
        str(getattr(classification, "risk_level", "low") or "low"),
        bool(getattr(classification, "must_use_legal_only", False)),
    )


# ---------------------------------------------------------------------------
# GuardrailResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuardrailResult:
    """Output of :func:`apply_guardrails`.

    action:
        "allow"        — chunks are safe; pass to answer generation with
                         any caution_flags prepended.
        "fallback"     — context insufficient or wrong section; use
                         FALLBACK_ANSWER verbatim.
        "block_unsafe" — dangerous claim type; use forced_response verbatim.

    chunks:
        Filtered, safe-to-use chunk list (may be empty when action != "allow").

    caution_flags:
        Ordered list of caution messages to prepend / inject into the answer.

    forced_response:
        Set only when action == "block_unsafe"; the answer generator MUST
        return this string without modification.

    reason:
        Human-readable explanation for logging.
    """

    action: Literal["allow", "fallback", "block_unsafe"]
    chunks: list[dict]
    caution_flags: list[str]
    forced_response: str | None
    reason: str | None


# ---------------------------------------------------------------------------
# Public predicates
# ---------------------------------------------------------------------------


def is_profit_guarantee_question(question: str) -> bool:
    """Return True if the question asks about guaranteed profit/appreciation."""
    norm = _normalize(question)
    if _any_match(norm, _PROFIT_GUARANTEE_KEYWORDS):
        return True
    return _any_match(norm, _PROFIT_CORE) and _any_match(norm, _GUARANTEE_CORE)


def is_deposit_or_opening_question(question: str) -> bool:
    """Return True if the question is about deposit, sale opening, or fundraising."""
    norm = _normalize(question)
    return (
        _any_match(norm, _DEPOSIT_KEYWORDS)
        or _any_match(norm, _OPENING_KEYWORDS)
        or _any_match(norm, _FUNDRAISING_KEYWORDS)
    )


def filter_legal_chunks(chunks: list[dict]) -> list[dict]:
    """Return only legal-section chunks, with legal_status_and_warnings first."""
    legal = [c for c in chunks if (c.get("section") or "") == _LEGAL_SECTION]
    anchor = [c for c in legal if (c.get("topic") or "") == _LEGAL_STATUS_TOPIC]
    rest = [c for c in legal if (c.get("topic") or "") != _LEGAL_STATUS_TOPIC]
    return anchor + rest


def contains_prohibited_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    """Return True if *text* contains any phrase from *phrases* (case-insensitive)."""
    text_lower = text.lower()
    return any(phrase.lower() in text_lower for phrase in phrases)


def fallback_answer() -> str:
    """Return the standard fallback answer from AGENTS.md."""
    return FALLBACK_ANSWER


def investment_return_response() -> str:
    """Return the AGENTS.md-safe investment-return response."""
    return INVESTMENT_RETURN_RESPONSE


# ---------------------------------------------------------------------------
# Main guardrail function
# ---------------------------------------------------------------------------


def apply_guardrails(
    question: str,
    chunks: list[dict],
    classification: Any = None,
    min_chunks_required: int = 1,
) -> GuardrailResult:
    """Apply safety guardrails to retrieval output before answer generation.

    Parameters
    ----------
    question:
        The original user question.
    chunks:
        List of chunk dicts returned by the retriever (already reranked).
    classification:
        ClassificationResult dataclass or equivalent dict produced by
        ``app.intent_classifier.classify()``.  May be None.
    min_chunks_required:
        Minimum number of chunks required to proceed; if fewer are available
        after filtering, the action becomes "fallback".

    Returns
    -------
    GuardrailResult
        Structured decision for the answer generator.
    """
    intent, risk_level, must_use_legal_only = _extract_classification(classification)

    # ------------------------------------------------------------------
    # Rule 1 — Guaranteed profit / appreciation (block regardless of chunks)
    # ------------------------------------------------------------------
    if is_profit_guarantee_question(question):
        return GuardrailResult(
            action="block_unsafe",
            chunks=[],
            caution_flags=[],
            forced_response=INVESTMENT_RETURN_RESPONSE,
            reason="guaranteed_profit_question",
        )

    # ------------------------------------------------------------------
    # Rule 2 — Legal intent: enforce legal-section-only retrieval
    # ------------------------------------------------------------------
    if intent == "legal" or must_use_legal_only or risk_level == "critical":
        legal_chunks = filter_legal_chunks(chunks)

        if len(legal_chunks) < min_chunks_required:
            return GuardrailResult(
                action="fallback",
                chunks=[],
                caution_flags=[],
                forced_response=None,
                reason="legal_query_no_legal_chunks",
            )

        return GuardrailResult(
            action="allow",
            chunks=legal_chunks,
            caution_flags=[LEGAL_CAUTION_FLAG],
            forced_response=None,
            reason="legal_query_legal_chunks_present",
        )

    # ------------------------------------------------------------------
    # Rule 3 — Deposit / sale opening / fundraising without legal support
    # ------------------------------------------------------------------
    if is_deposit_or_opening_question(question):
        legal_chunks = filter_legal_chunks(chunks)
        if not legal_chunks:
            return GuardrailResult(
                action="fallback",
                chunks=[],
                caution_flags=[],
                forced_response=None,
                reason="deposit_opening_question_no_legal_support",
            )
        return GuardrailResult(
            action="allow",
            chunks=legal_chunks,
            caution_flags=[LEGAL_CAUTION_FLAG],
            forced_response=None,
            reason="deposit_opening_question_legal_support_present",
        )

    # ------------------------------------------------------------------
    # Rule 4 — Pricing: require cautious framing
    # ------------------------------------------------------------------
    if intent == "pricing" or (risk_level == "high" and intent != "sales_policy"):
        if not chunks:
            return GuardrailResult(
                action="fallback",
                chunks=[],
                caution_flags=[],
                forced_response=None,
                reason="pricing_no_chunks",
            )
        return GuardrailResult(
            action="allow",
            chunks=chunks,
            caution_flags=[PRICING_CAUTION_FLAG],
            forced_response=None,
            reason="pricing_caution_applied",
        )

    # ------------------------------------------------------------------
    # Rule 5 — Sales policy: non-official framing required
    # ------------------------------------------------------------------
    if intent == "sales_policy":
        if not chunks:
            return GuardrailResult(
                action="fallback",
                chunks=[],
                caution_flags=[],
                forced_response=None,
                reason="sales_policy_no_chunks",
            )
        return GuardrailResult(
            action="allow",
            chunks=chunks,
            caution_flags=[SALES_POLICY_CAUTION_FLAG],
            forced_response=None,
            reason="sales_policy_caution_applied",
        )

    # ------------------------------------------------------------------
    # Rule 6 — Insufficient context fallback (any remaining intent)
    # ------------------------------------------------------------------
    if len(chunks) < min_chunks_required:
        return GuardrailResult(
            action="fallback",
            chunks=[],
            caution_flags=[],
            forced_response=None,
            reason="insufficient_context",
        )

    # ------------------------------------------------------------------
    # Default — allow with no additional cautions
    # ------------------------------------------------------------------
    return GuardrailResult(
        action="allow",
        chunks=chunks,
        caution_flags=[],
        forced_response=None,
        reason="allowed",
    )


def check_answer_safety(answer: str, intent: str) -> list[str]:
    """Scan a generated answer for prohibited phrases.

    Returns a list of violations found (empty list = safe).
    This is a secondary check to catch any prohibited phrases that slipped
    through to the generated text.
    """
    violations: list[str] = []
    answer_lower = answer.lower()

    if intent in ("pricing",):
        for phrase in PRICING_PROHIBITED_PHRASES:
            if phrase.lower() in answer_lower:
                violations.append(f"pricing_prohibited: {phrase!r}")

    if intent in ("legal", "pricing", "sales_policy"):
        for phrase in LEGAL_PROHIBITED_PHRASES:
            if phrase.lower() in answer_lower:
                violations.append(f"legal_prohibited: {phrase!r}")

    return violations
