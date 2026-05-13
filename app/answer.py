"""app/answer.py

Constrained answer assembler for the NEO CITY RAG pipeline.

Public API
----------
generate_answer(question, chunks, classification, min_chunks_required) -> AnswerResult
    Apply guardrails to the retrieved chunks, then assemble a safe,
    context-only answer.  Never calls an external LLM.

answer_from_retrieval(retrieval_result, min_chunks_required) -> AnswerResult
    Convenience wrapper: accepts the dict returned by app.retriever.retrieve()
    and delegates to generate_answer.

AnswerResult
    Frozen dataclass: answer_text, used_chunk_ids, used_sections,
    confidence, answer_mode.

chatbot_answer_from_retrieval(retrieval_result) -> str
    Return a concise customer-facing answer suitable for terminal demos.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

from app.guardrails import (
    FALLBACK_ANSWER,
    INVESTMENT_RETURN_RESPONSE,
    GuardrailResult,
    apply_guardrails,
)

# ---------------------------------------------------------------------------
# Intent-aware intro sentences (contain AGENTS.md required cautious wording)
# ---------------------------------------------------------------------------

_PRICING_INTRO = (
    "Theo tài liệu định hướng hiện tại của NEO CITY, thông tin dự kiến như sau:"
)
_LEGAL_INTRO = "Theo thông tin pháp lý hiện có trong tài liệu NEO CITY:"
_SALES_POLICY_INTRO = (
    "Theo tài liệu dự kiến của NEO CITY, chính sách hiện tại như sau:"
)
_GENERAL_INTRO = "Theo tài liệu NEO CITY:"

_CHUNK_SEPARATOR = "\n\n"
_MAX_CHUNKS_IN_ANSWER = 3
_MAX_PASSAGE_CHARS = 500

# Common Vietnamese function words excluded from keyword matching.
# "ban" is intentionally excluded: in real estate context it means "bán" (sell),
# not the function word "ban" (you/your), so it carries useful signal for queries
# like "mở bán", "cho phép bán", "được bán chưa".
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "la", "va", "co", "khong", "nhu", "the", "toi", "cac",
        "mot", "cua", "cho", "trong", "tren", "duoi", "ve", "neu",
        "thi", "cung", "da", "dang", "se", "khi", "boi", "duoc",
        "den", "tu", "theo", "voi", "hoac", "nhung", "ma", "hay",
    }
)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnswerResult:
    """Structured answer returned by generate_answer.

    answer_text:
        The full answer string, safe to present to the user.
    used_chunk_ids:
        IDs of the chunks whose text contributed to answer_text.
    used_sections:
        Deduplicated ordered list of section names from used chunks.
    confidence:
        Deterministic float in [0, 1]; 0.0 for fallback/blocked answers.
    answer_mode:
        "answered"  — context found, answer assembled from chunks.
        "fallback"  — insufficient context; FALLBACK_ANSWER returned.
        "blocked"   — guardrail triggered; forced_response returned.
    """

    answer_text: str
    used_chunk_ids: list[str]
    used_sections: list[str]
    confidence: float
    answer_mode: Literal["answered", "fallback", "blocked"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_answer(
    question: str,
    chunks: list[dict],
    classification: Any = None,
    min_chunks_required: int = 1,
) -> AnswerResult:
    """Apply guardrails then assemble a safe answer from guarded chunks.

    Parameters
    ----------
    question:
        Raw user question.
    chunks:
        Chunk dicts from the retriever (already reranked).
    classification:
        ClassificationResult or equivalent dict from
        app.intent_classifier.classify().  May be None.
    min_chunks_required:
        Minimum guarded chunks needed to produce an "answered" result.
        Fewer remaining after guardrail filtering yields "fallback".
    """
    guardrail = apply_guardrails(question, chunks, classification, min_chunks_required)

    if guardrail.action == "block_unsafe":
        return AnswerResult(
            answer_text=guardrail.forced_response or INVESTMENT_RETURN_RESPONSE,
            used_chunk_ids=[],
            used_sections=[],
            confidence=0.0,
            answer_mode="blocked",
        )

    if guardrail.action == "fallback":
        return AnswerResult(
            answer_text=FALLBACK_ANSWER,
            used_chunk_ids=[],
            used_sections=[],
            confidence=0.0,
            answer_mode="fallback",
        )

    return _assemble_answer(guardrail, classification, question)


def answer_from_retrieval(
    retrieval_result: dict,
    min_chunks_required: int = 1,
) -> AnswerResult:
    """Convenience wrapper: accepts a dict from app.retriever.retrieve().

    Extracts question, chunks, and classification fields then delegates to
    generate_answer.
    """
    question = str(retrieval_result.get("question", "") or "")
    chunks = list(retrieval_result.get("chunks", []) or [])
    classification = {
        "intent": retrieval_result.get("intent", ""),
        "risk_level": retrieval_result.get("risk_level", "low"),
        "must_use_legal_only": bool(retrieval_result.get("must_use_legal_only", False)),
        "target_sections": list(retrieval_result.get("target_sections", []) or []),
    }
    return generate_answer(question, chunks, classification, min_chunks_required)


def chatbot_answer_from_retrieval(
    retrieval_result: dict,
    min_chunks_required: int = 1,
) -> str:
    """Return a concise, customer-facing answer for terminal demos."""
    answer = answer_from_retrieval(retrieval_result, min_chunks_required)
    if answer.answer_mode != "answered":
        return answer.answer_text

    question = str(retrieval_result.get("question", "") or "")
    chunks = list(retrieval_result.get("chunks", []) or [])
    intent = str(retrieval_result.get("intent", "") or "")

    if intent == "legal":
        return _build_concise_legal_answer(question, chunks, answer.answer_text)
    if intent == "pricing":
        return _build_concise_pricing_answer(question, chunks)
    if intent == "sales_policy":
        return _build_concise_sales_policy_answer(question, chunks)
    return _build_concise_general_answer(question, chunks)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _assemble_answer(
    guardrail: GuardrailResult,
    classification: Any,
    question: str = "",
) -> AnswerResult:
    """Build an AnswerResult from an 'allow' GuardrailResult."""
    chunks = guardrail.chunks
    intent = _extract_intent(classification)

    used_chunks = chunks[:_MAX_CHUNKS_IN_ANSWER]
    used_chunk_ids = [str(c.get("id", "")) for c in used_chunks]
    # Deduplicate section names while preserving insertion order.
    used_sections = list(dict.fromkeys(c.get("section", "") for c in used_chunks))

    body_parts: list[str] = []
    for chunk in used_chunks:
        raw = (chunk.get("text", "") or "").strip()
        if raw:
            passage = _extract_relevant_lines(raw, question)
            body_parts.append(passage)

    body = _CHUNK_SEPARATOR.join(body_parts)
    intro = _select_intro(intent)

    parts: list[str] = []
    if guardrail.caution_flags:
        parts.append("\n".join(guardrail.caution_flags))
    parts.append(f"{intro}\n\n{body}")

    answer_text = "\n\n".join(parts)
    confidence = _compute_confidence(used_chunks)

    return AnswerResult(
        answer_text=answer_text,
        used_chunk_ids=used_chunk_ids,
        used_sections=used_sections,
        confidence=confidence,
        answer_mode="answered",
    )


def _extract_relevant_lines(text: str, question: str, max_chars: int = _MAX_PASSAGE_CHARS) -> str:
    """Extract the most question-relevant lines from chunk text.

    If the text is already within max_chars, it is returned unchanged.
    Otherwise, each non-empty line is scored by keyword overlap with the
    question; the highest-scoring lines are selected greedily up to
    max_chars and returned in their original order.

    At least one line is always returned (even if it exceeds max_chars)
    so the answer is never empty when the chunk has content.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text

    q_words = set(_normalize_for_matching(question).split()) - _STOP_WORDS

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text[:max_chars].strip()

    if not q_words:
        # No meaningful question keywords — return the first max_chars characters.
        return "\n".join(lines)[:max_chars].strip()

    # Score: (keyword overlap DESC, index ASC) so earlier high-scoring lines win ties.
    scored: list[tuple[int, int, int, str]] = []
    for idx, line in enumerate(lines):
        line_words = set(_normalize_for_matching(line).split())
        overlap = len(q_words.intersection(line_words))
        scored.append((overlap, -idx, idx, line))
    scored.sort(reverse=True)

    selected: list[tuple[int, str]] = []
    total = 0
    for overlap, _neg_idx, idx, line in scored:
        if total > 0 and total + len(line) + 1 > max_chars:
            break
        selected.append((idx, line))
        total += len(line) + 1

    # Restore original line order.
    selected.sort(key=lambda x: x[0])
    result = "\n".join(line for _, line in selected)
    return result if result else text[:max_chars].strip()


def _normalize_for_matching(text: str) -> str:
    """Lowercase, strip Vietnamese diacritics, collapse whitespace."""
    text = text.replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", stripped.lower().strip())


def _select_intro(intent: str) -> str:
    if intent == "pricing":
        return _PRICING_INTRO
    if intent == "legal":
        return _LEGAL_INTRO
    if intent == "sales_policy":
        return _SALES_POLICY_INTRO
    return _GENERAL_INTRO


def _extract_intent(classification: Any) -> str:
    if classification is None:
        return ""
    if isinstance(classification, dict):
        return str(classification.get("intent", "") or "")
    return str(getattr(classification, "intent", "") or "")


def _compute_confidence(chunks: list[dict]) -> float:
    """Deterministic confidence in [0, 1] based on rerank scores and diversity."""
    if not chunks:
        return 0.0

    scores = [
        float(c.get("rerank_score", c.get("score", 0.0)) or 0.0)
        for c in chunks
    ]
    avg_score = sum(scores) / len(scores)
    confidence = min(1.0, max(0.0, avg_score))

    # Bonus when all chunks come from the same section (higher coherence).
    sections = {c.get("section", "") for c in chunks}
    if len(sections) == 1:
        confidence = min(1.0, confidence + 0.05)

    # Penalty for single-chunk answers (less cross-referenced, less reliable).
    if len(chunks) < 2:
        confidence = max(0.0, confidence - 0.10)

    return round(confidence, 4)


def _build_concise_legal_answer(question: str, chunks: list[dict], fallback_text: str) -> str:
    question_norm = _normalize_for_matching(question)
    legal_text = "\n".join(
        (chunk.get("text", "") or "").strip()
        for chunk in chunks
        if (chunk.get("section", "") or "") == "legal"
    )
    legal_norm = _normalize_for_matching(legal_text)

    if "mo ban" in question_norm and "chua mo ban" in legal_norm:
        return (
            "Theo tài liệu NEO CITY hiện tại, dự án chưa mở bán chính thức. "
            "Tài liệu cũng nêu đây mới là thông tin định hướng phát triển, chưa phải thông báo giao dịch chính thức."
        )
    if "dat coc" in question_norm:
        if "chua mo ban" in legal_norm:
            return (
                "Theo tài liệu NEO CITY hiện tại, chưa có cơ sở để khẳng định có thể đặt cọc ngay. "
                "Tài liệu đồng thời cho biết dự án chưa mở bán chính thức."
            )
        return (
            "Theo tài liệu NEO CITY hiện tại, tôi chưa thấy căn cứ pháp lý đủ rõ để khẳng định có thể đặt cọc ngay."
        )
    if "huy dong von" in question_norm and "chua huy dong von tu khach hang" in legal_norm:
        return (
            "Theo tài liệu NEO CITY hiện tại, dự án chưa huy động vốn từ khách hàng."
        )
    if "phap ly" in question_norm:
        detail = _extract_relevant_lines(legal_text, question, max_chars=260)
        return (
            "Theo tài liệu pháp lý hiện có của NEO CITY, dự án đang ở giai đoạn định hướng và hoàn thiện hồ sơ pháp lý. "
            f"{_to_sentence(detail)}"
        )
    return _to_sentence(fallback_text)


def _build_concise_pricing_answer(question: str, chunks: list[dict]) -> str:
    pricing_chunk = _first_section_chunk(chunks, {"pricing", "price_sheet"})
    if pricing_chunk is None:
        return FALLBACK_ANSWER
    raw_text = (pricing_chunk.get("text", "") or "").strip()
    detail = _extract_pricing_highlights(raw_text, question, max_chars=260)
    return (
        "Theo tài liệu định hướng hiện tại của NEO CITY, đây là thông tin giá dự kiến, chưa phải giá bán chính thức. "
        f"{_to_sentence(_format_demo_passage(detail))}"
    )


def _build_concise_sales_policy_answer(question: str, chunks: list[dict]) -> str:
    policy_chunk = _first_section_chunk(chunks, {"sales_policy", "price_sheet"})
    if policy_chunk is None:
        return FALLBACK_ANSWER
    detail = _extract_relevant_lines((policy_chunk.get("text", "") or "").strip(), question, max_chars=260)
    return (
        "Theo tài liệu dự kiến của NEO CITY, chính sách này đang ở mức định hướng và có thể thay đổi theo từng đợt mở bán chính thức. "
        f"{_to_sentence(_format_demo_passage(detail))}"
    )


def _build_concise_general_answer(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return FALLBACK_ANSWER
    detail = _extract_relevant_lines((chunks[0].get("text", "") or "").strip(), question, max_chars=2000)
    formatted = _format_demo_passage(detail)
    if "\n" in formatted:
        return f"Theo tài liệu NEO CITY:\n{formatted}"
    return _to_sentence(f"Theo tài liệu NEO CITY, {formatted}")


def _first_section_chunk(chunks: list[dict], sections: set[str]) -> dict | None:
    for chunk in chunks:
        if (chunk.get("section", "") or "") in sections:
            return chunk
    return chunks[0] if chunks else None


def _to_sentence(text: str) -> str:
    cleaned = " ".join(part.strip() for part in text.splitlines() if part.strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""
    return cleaned if cleaned.endswith((".", "!", "?")) else f"{cleaned}."


def _format_demo_passage(text: str) -> str:
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "|" in line:
            cells = [cell.strip() for cell in line.split("|") if cell.strip()]
            if len(cells) >= 4 and cells[0].lower() not in {"loại sản phẩm", "---"}:
                line = f"{cells[0]}: {cells[2]}, tổng giá trị khoảng {cells[3]}"
            elif len(cells) >= 2 and cells[0].lower() not in {"hạng mục", "nội dung", "---"}:
                line = f"{cells[0]}: {cells[1]}"
            else:
                continue
        lines.append(line)
    return "\n".join(lines) if lines else text


def _extract_pricing_highlights(text: str, question: str, max_chars: int = 260) -> str:
    """Prefer lines with actual money/range signals for concise pricing demos."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text[:max_chars].strip()

    question_norm = _normalize_for_matching(question)
    question_tokens = set(question_norm.split()) - _STOP_WORDS
    product_markers = (
        "studio", "1pn", "1pn+1", "2pn", "2pn+1", "3pn",
        "shophouse", "townhouse", "villa", "courtyard", "can ho",
    )
    asked_product_markers = _extract_asked_product_markers(question_norm, product_markers)
    money_markers = ("trieu", "ty", "m²", "m2", "dong/m", "%")

    scored: list[tuple[int, int, int, str]] = []
    for idx, line in enumerate(lines):
        line_norm = _normalize_for_matching(line)
        line_tokens = set(line_norm.split())
        overlap = len(question_tokens.intersection(line_tokens))
        has_money = any(marker in line.lower() for marker in money_markers) or bool(re.search(r"\d", line))
        has_product = any(marker in line_norm for marker in product_markers)
        score = overlap
        if has_money:
            score += 5
        if has_product:
            score += 3
        if asked_product_markers and any(marker in line_norm for marker in asked_product_markers):
            score += 6
        if "du kien" in line_norm or "dinh huong gia" in line_norm:
            score += 2
        if "loai san pham" in line_norm or "dien tich du kien" in line_norm or "tong gia tri du kien" in line_norm:
            score -= 2
        scored.append((score, -idx, idx, line))

    scored.sort(reverse=True)

    selected: list[tuple[int, str]] = []
    total = 0
    for score, _neg_idx, idx, line in scored:
        if score <= 0 and selected:
            continue
        if total > 0 and total + len(line) + 1 > max_chars:
            continue
        selected.append((idx, line))
        total += len(line) + 1
        if len(selected) >= 2:
            break

    if not selected:
        return _extract_relevant_lines(text, question, max_chars=max_chars)

    selected.sort(key=lambda x: x[0])
    if asked_product_markers:
        selected = [
            item for item in selected
            if any(marker in _normalize_for_matching(item[1]) for marker in asked_product_markers)
        ] or selected[:1]
    return "\n".join(line for _, line in selected)


def _extract_asked_product_markers(question_norm: str, product_markers: tuple[str, ...]) -> tuple[str, ...]:
    """Return the most specific product markers mentioned in the question."""
    matched = [marker for marker in product_markers if marker in question_norm]
    if not matched:
        return ()

    matched.sort(key=len, reverse=True)
    specific: list[str] = []
    for marker in matched:
        if any(marker != kept and marker in kept for kept in specific):
            continue
        specific.append(marker)
    return tuple(specific)
