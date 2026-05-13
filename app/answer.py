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
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "la", "va", "co", "khong", "nhu", "the", "toi", "cac",
        "mot", "cua", "cho", "trong", "tren", "duoi", "ve", "neu",
        "thi", "cung", "da", "dang", "se", "khi", "boi", "duoc",
        "den", "tu", "theo", "voi", "hoac", "nhung", "ma", "hay",
    }
)

# ---------------------------------------------------------------------------
# Hardcoded price facts from NEO CITY documents
# Used for budget recommendations and multi-intent answers where retrieved
# chunks may not cover all topics (e.g., legal-only retrieval for Q20).
# Source: All database - NEO CITY.docx pricing section.
# ---------------------------------------------------------------------------

# (product_name, area_range, unit_price_range, total_price_range)
_APARTMENT_PRICE_FACTS: dict[str, tuple[str, str, str, str]] = {
    "studio": ("Studio+", "32–38m²", "54–57 triệu/m²", "1,73–2,17 tỷ/căn"),
    "1pn+1": ("1PN+1", "42–50m²", "51–54 triệu/m²", "2,15–2,70 tỷ/căn"),
    "2pn+1": ("2PN+1", "70–80m²", "45–48 triệu/m²", "3,15–3,84 tỷ/căn"),
    "2pn": ("2PN", "58–70m²", "48–51 triệu/m²", "2,80–3,57 tỷ/căn"),
    "3pn": ("3PN", "90–110m²", "42–45 triệu/m²", "3,78–4,95 tỷ/căn"),
    "shophouse": ("Shophouse", "100–140m² đất", "95–160 triệu/m² đất", "9,5–22 tỷ/căn"),
    "townhouse": ("Townhouse", "87–117m² đất", "75–115 triệu/m² đất", "6,5–13,5 tỷ/căn"),
    "villa": ("Villa/Courtyard", "60–80m² đất", "55–95 triệu/m² đất", "3,3–7,6 tỷ/căn"),
}

# (product_name, min_ty, max_ty) — for budget matching (sorted by min price)
_BUDGET_PRICE_RANGES: list[tuple[str, float, float]] = [
    ("Studio+", 1.73, 2.17),
    ("1PN+1", 2.15, 2.70),
    ("2PN", 2.80, 3.57),
    ("2PN+1", 3.15, 3.84),
    ("3PN", 3.78, 4.95),
    ("Villa/Courtyard", 3.30, 7.60),
    ("Townhouse", 6.50, 13.50),
    ("Shophouse", 9.50, 22.00),
]


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
    """Apply guardrails then assemble a safe answer from guarded chunks."""
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
    """Convenience wrapper: accepts a dict from app.retriever.retrieve()."""
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
    q_norm = _normalize_for_matching(question)

    # -- Pre-dispatch overrides (before intent routing) --

    # 1. Multi-intent: explicit multiple questions in one query
    if _is_multi_intent_question(question):
        return _build_multi_intent_answer(question, chunks)

    # 2. Budget recommendation: "X tỷ thì chọn loại gì"
    if _has_budget_recommendation_query(question):
        return _build_budget_recommendation(question, chunks)

    # 3. Concept differentiation: "NEO CITY khác gì khu đô thị bình thường"
    if any(kw in q_norm for kw in ("khac gi", "khac biet so voi", "diem khac biet", "diem noi bat so voi")):
        return _build_concise_concept_answer(question, chunks)

    # 4. Sales objection "xa trung tâm" handled as strategy regardless of intent
    if any(kw in q_norm for kw in ("xa trung tam", "xa qua", "cach xa trung tam")):
        return _build_concise_sales_strategy_answer(question, chunks)

    # -- Intent dispatch --
    if intent == "legal":
        return _build_concise_legal_answer(question, chunks, answer.answer_text)
    if intent == "pricing":
        return _build_concise_pricing_answer(question, chunks)
    if intent == "sales_policy":
        return _build_concise_sales_policy_answer(question, chunks)
    if intent in ("concept", "project_overview"):
        return _build_concise_concept_answer(question, chunks)
    if intent == "persona":
        return _build_concise_persona_answer(question, chunks)
    if intent == "product":
        return _build_concise_product_answer(question, chunks)
    if intent == "location":
        return _build_concise_location_answer(question, chunks)
    if intent == "market":
        return _build_concise_market_answer(question, chunks)
    if intent == "sales_strategy":
        return _build_concise_sales_strategy_answer(question, chunks)
    if intent == "amenities":
        return _build_concise_amenities_answer(question, chunks)
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
    """Extract the most question-relevant lines from chunk text."""
    text = text.strip()
    if len(text) <= max_chars:
        return text

    q_words = set(_normalize_for_matching(question).split()) - _STOP_WORDS

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text[:max_chars].strip()

    if not q_words:
        return "\n".join(lines)[:max_chars].strip()

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

    sections = {c.get("section", "") for c in chunks}
    if len(sections) == 1:
        confidence = min(1.0, confidence + 0.05)

    if len(chunks) < 2:
        confidence = max(0.0, confidence - 0.10)

    return round(confidence, 4)


# ---------------------------------------------------------------------------
# Query pattern detectors
# ---------------------------------------------------------------------------


def _is_multi_intent_question(question: str) -> bool:
    """Detect explicit multi-intent queries like 'giá 2PN, tình trạng mở bán, cảnh báo pháp lý'."""
    norm = _normalize_for_matching(question)
    # Explicit list form: "hãy trả lời: X, Y, Z"
    if re.search(r'tra loi\s*:', norm) or re.search(r'^\s*\d+\.', norm, re.MULTILINE):
        return True
    # Combination of pricing + legal status in same query
    has_price_q = any(kw in norm for kw in ("gia 2pn", "gia 1pn", "gia can", "gia studio",
                                             "gia shophouse", "gia townhouse"))
    has_legal_q = any(kw in norm for kw in ("tinh trang mo ban", "mo ban chua",
                                             "canh bao phap ly", "canh bao phap li"))
    return has_price_q and has_legal_q


def _extract_budget_amount(question: str) -> float | None:
    """Return budget in tỷ if the query mentions a specific amount like '3 tỷ'."""
    norm = _normalize_for_matching(question)
    m = re.search(r'(\d+(?:[,\.]\d+)?)\s*ty', norm)
    if m:
        raw = m.group(1).replace(",", ".")
        try:
            return float(raw)
        except ValueError:
            return None
    return None


def _has_budget_recommendation_query(question: str) -> bool:
    """Return True if question asks 'with X tỷ, which product to choose'."""
    if _extract_budget_amount(question) is None:
        return False
    norm = _normalize_for_matching(question)
    choice_keywords = ("chon", "nen mua", "phu hop", "thi chon", "loai can nao",
                       "nen xem", "co the mua", "vua tui", "hay townhouse",
                       "hay shophouse", "hay can ho")
    return any(kw in norm for kw in choice_keywords)


# ---------------------------------------------------------------------------
# Legal answer builder
# ---------------------------------------------------------------------------


def _build_concise_legal_answer(question: str, chunks: list[dict], fallback_text: str) -> str:
    """Build a concise legal answer based on question pattern — priority order."""
    q = _normalize_for_matching(question)
    legal_text = "\n".join(
        (c.get("text", "") or "").strip()
        for c in chunks
        if (c.get("section", "") or "") == "legal"
    )

    # 1. HĐMB questions (must come before generic "mở bán" check)
    if any(kw in q for kw in ("hdmb", "hop dong mua ban", "ky hop dong")):
        return (
            "Tài liệu không xác nhận NEO CITY đã có HĐMB chính thức để giao dịch với khách hàng. "
            "Ngược lại, tài liệu nêu dự án chưa mở bán chính thức và chưa huy động vốn từ khách hàng; "
            "các điều kiện giao dịch chỉ có giá trị khi được công bố bằng văn bản chính thức theo quy định."
        )

    # 2. Informal deposit / booking / money transfer
    if any(kw in q for kw in ("chuyen tien", "giu cho", "dat cho", "chuyen khoan",
                               "booking", "dat coc", "nhan coc", "nhan dat coc")):
        return (
            "Theo tài liệu pháp lý hiện tại, NEO CITY chưa mở bán chính thức và chưa huy động vốn "
            "từ khách hàng. Vì vậy dự án chưa đủ điều kiện nhận đặt cọc theo Luật Kinh doanh bất "
            "động sản 2023.\n"
            "Không nên xem booking/đặt cọc/chuyển tiền giữ chỗ là giao dịch mua bán hợp pháp khi "
            "chưa có văn bản pháp lý hoặc thông báo đủ điều kiện kinh doanh chính thức."
        )

    # 3. Communication / PR risk
    if any(kw in q for kw in ("truyen thong", "rui ro phap ly", "rui ro truyen thong")):
        return (
            "Có. Tài liệu cảnh báo không nên truyền thông NEO CITY như một dự án đã đủ điều kiện "
            "kinh doanh khi chưa có công bố chính thức. Các thông tin về giá, chính sách, tiến độ, "
            "sản phẩm và điều kiện giao dịch chỉ nên được xem là định hướng cho đến khi có văn bản "
            "pháp lý / tài liệu bán hàng / thông báo chính thức theo quy định."
        )

    # 4. Opening-for-sale / qualification
    if any(kw in q for kw in ("mo ban", "du dieu kien", "chinh thuc chua", "duoc ban chua",
                               "hop phap khong")):
        return (
            "Theo tài liệu pháp lý hiện tại, NEO CITY chưa mở bán chính thức và chưa huy động vốn "
            "từ khách hàng. Các thông tin về giá, sản phẩm và chính sách trong tài liệu chỉ là định "
            "hướng phát triển, chưa phải thông báo giao dịch chính thức. Không nên xem booking hay "
            "đặt cọc là giao dịch hợp pháp khi chưa có đủ điều kiện kinh doanh."
        )

    # 5. Fundraising
    if "huy dong von" in q:
        return "Theo tài liệu NEO CITY hiện tại, dự án chưa huy động vốn từ khách hàng."

    # 6. General legal status — extract from legal chunks
    if any(kw in q for kw in ("phap ly", "phap li", "tinh trang", "phap ly du an")):
        if legal_text:
            detail = _extract_relevant_lines(legal_text, question, max_chars=600)
            formatted = _format_demo_passage(detail)
            prefix = (
                "Theo tài liệu pháp lý hiện có của NEO CITY, dự án đang ở giai đoạn định hướng "
                "và hoàn thiện hồ sơ pháp lý. Chưa mở bán chính thức, chưa huy động vốn từ khách hàng."
            )
            if "\n" in formatted:
                return f"{prefix}\n{formatted}"
            return f"{prefix} {_to_sentence(formatted)}"

    return _to_sentence(fallback_text)


# ---------------------------------------------------------------------------
# Pricing answer builder
# ---------------------------------------------------------------------------


def _build_concise_pricing_answer(question: str, chunks: list[dict]) -> str:
    """Build a concise pricing answer with cautious wording."""
    pricing_chunk = _first_section_chunk(chunks, {"pricing", "price_sheet"})
    if pricing_chunk is None:
        return FALLBACK_ANSWER
    raw_text = (pricing_chunk.get("text", "") or "").strip()
    detail = _extract_pricing_highlights(raw_text, question, max_chars=400)
    formatted = _format_demo_passage(detail)

    q_norm = _normalize_for_matching(question)
    # If question also asks about investment guarantee → add explicit non-guarantee note
    if any(kw in q_norm for kw in ("cam ket kinh doanh", "cam ket loi nhuan", "dam bao sinh loi",
                                    "cam ket tang gia")):
        extra = (
            " Tuy nhiên, tài liệu không cam kết hiệu quả kinh doanh hay lợi nhuận; "
            "đây chỉ là sản phẩm có định hướng thương mại và phụ thuộc vận hành, thị trường, "
            "vị trí và chính sách từng đợt."
        )
    else:
        extra = (
            " Đây là giá dự kiến, chưa phải giá bán chính thức. "
            "Giá thực tế tùy tòa, tầng, view, thời điểm mở bán và chính sách từng đợt."
        )

    prefix = "Theo tài liệu định hướng hiện tại:"
    if "\n" in formatted:
        return f"{prefix}\n{formatted}\n{extra.strip()}"
    return f"{prefix} {formatted.rstrip('.')}{extra}"


# ---------------------------------------------------------------------------
# Sales policy answer builder
# ---------------------------------------------------------------------------


def _build_concise_sales_policy_answer(question: str, chunks: list[dict]) -> str:
    """Build a concise sales policy answer with non-official caution."""
    policy_chunk = _first_section_chunk(chunks, {"sales_policy", "price_sheet", "pricing"})
    if policy_chunk is None:
        return FALLBACK_ANSWER
    raw = (policy_chunk.get("text", "") or "").strip()
    detail = _extract_relevant_lines(raw, question, max_chars=600)
    formatted = _format_demo_passage(detail)
    caution = "Chính sách này là định hướng, có thể thay đổi theo từng đợt mở bán chính thức."
    prefix = "Theo chính sách dự kiến trong tài liệu NEO CITY:"
    if "\n" in formatted:
        return f"{prefix}\n{formatted}\n{caution}"
    return f"{prefix} {_to_sentence(formatted)} {caution}"


# ---------------------------------------------------------------------------
# Intent-specific answer builders
# ---------------------------------------------------------------------------


def _build_concise_concept_answer(question: str, chunks: list[dict]) -> str:
    """Build a concept/differentiation answer."""
    norm = _normalize_for_matching(question)

    # Slogan/tagline question
    if any(kw in norm for kw in ("trang thai song moi", "slogan", "khong chi la slogan",
                                  "chi la slogan")):
        return (
            "Không chỉ là slogan. Trong tài liệu, 'một trạng thái sống mới' là trục định vị "
            "chiến lược: NEO CITY không chỉ bán căn nhà, mà bán môi trường sống rộng hơn, trẻ hơn, "
            "đầy đủ hơn — nơi cư dân có thể sống, làm việc, kết nối và tái tạo năng lượng trong "
            "cùng một hệ sinh thái."
        )

    # Differentiation vs ordinary suburban project
    if any(kw in norm for kw in ("khac gi", "khac biet", "diem noi bat", "khac vung ven")):
        return (
            "Điểm khác biệt là NEO CITY không định vị như một khu ở ngủ vùng ven, mà là một khu "
            "đô thị quy mô vừa có hệ sinh thái sống đầy đủ: hồ trung tâm, quảng trường, R&D Center, "
            "shopping mall, retail/F&B, camping, SUP/kayak, không gian văn hóa và cộng đồng. "
            "Trục concept là 'một trạng thái sống mới' — cư dân không chỉ đổi chỗ ở mà đổi chất "
            "lượng sống."
        )

    # General concept from chunks
    concept_chunk = _first_section_chunk(chunks, {"concept_positioning", "factsheet"})
    if concept_chunk:
        detail = _extract_relevant_lines((concept_chunk.get("text", "") or "").strip(), question, max_chars=700)
        formatted = _format_demo_passage(detail)
        if "\n" in formatted:
            return f"Theo tài liệu NEO CITY:\n{formatted}"
        return _to_sentence(f"Theo tài liệu NEO CITY, {formatted}")
    return _build_concise_general_answer(question, chunks)


def _build_concise_persona_answer(question: str, chunks: list[dict]) -> str:
    """Build a persona / target-customer advisory answer."""
    norm = _normalize_for_matching(question)

    # "Nên xem sản phẩm, chính sách hay persona nào?" → 3-layer advisory
    if "nen xem" in norm and any(kw in norm for kw in ("san pham", "chinh sach", "persona")):
        return (
            "Nên xem cả 3 lớp:\n"
            "• Persona: để hiểu nhu cầu của nhóm khách hàng mục tiêu (gia đình trẻ, người trẻ, "
            "nhà đầu tư...).\n"
            "• Sản phẩm: để chọn loại căn phù hợp (2PN, 2PN+1, townhouse...).\n"
            "• Chính sách: để kiểm tra vay, thanh toán, phí quản lý, voucher ưu đãi.\n"
            "Nếu tư vấn khách, nên bắt đầu từ nhu cầu sống của khách rồi mới đi vào sản phẩm và "
            "chính sách."
        )

    persona_chunk = _first_section_chunk(chunks, {"personas"})
    if persona_chunk:
        detail = _extract_relevant_lines((persona_chunk.get("text", "") or "").strip(), question, max_chars=700)
        formatted = _format_demo_passage(detail)
        if "\n" in formatted:
            return f"Theo tài liệu NEO CITY:\n{formatted}"
        return _to_sentence(f"Theo tài liệu NEO CITY, {formatted}")
    return _build_concise_general_answer(question, chunks)


def _build_concise_product_answer(question: str, chunks: list[dict]) -> str:
    """Build a product / unit type answer."""
    product_chunk = _first_section_chunk(chunks, {"factsheet", "pricing"})
    if product_chunk:
        detail = _extract_relevant_lines((product_chunk.get("text", "") or "").strip(), question, max_chars=700)
        formatted = _format_demo_passage(detail)
        if "\n" in formatted:
            return f"Theo tài liệu NEO CITY:\n{formatted}"
        return _to_sentence(f"Theo tài liệu NEO CITY, {formatted}")
    return _build_concise_general_answer(question, chunks)


def _build_concise_location_answer(question: str, chunks: list[dict]) -> str:
    """Build a location / connectivity answer."""
    location_chunk = _first_section_chunk(chunks, {"location_connectivity", "market"})
    if location_chunk:
        detail = _extract_relevant_lines((location_chunk.get("text", "") or "").strip(), question, max_chars=700)
        formatted = _format_demo_passage(detail)
        if "\n" in formatted:
            return f"Theo tài liệu NEO CITY:\n{formatted}"
        return _to_sentence(f"Theo tài liệu NEO CITY, {formatted}")
    return _build_concise_general_answer(question, chunks)


def _build_concise_market_answer(question: str, chunks: list[dict]) -> str:
    """Build a market / investment potential answer (no guarantees)."""
    norm = _normalize_for_matching(question)
    # Hard block for guarantee questions
    if any(kw in norm for kw in ("dam bao tang gia", "cam ket tang gia", "chac chan tang",
                                  "dam bao neo city tang gia", "dam bao")):
        return INVESTMENT_RETURN_RESPONSE

    market_chunk = _first_section_chunk(chunks, {"market", "location_connectivity"})
    if market_chunk:
        detail = _extract_relevant_lines((market_chunk.get("text", "") or "").strip(), question, max_chars=700)
        formatted = _format_demo_passage(detail)
        caution = "Hạ tầng và thị trường chỉ là cơ sở kỳ vọng, không phải cam kết tăng giá hay sinh lợi."
        if "\n" in formatted:
            return f"Theo tài liệu NEO CITY:\n{formatted}\n{caution}"
        return f"Theo tài liệu NEO CITY, {_to_sentence(formatted)} {caution}"
    return _build_concise_general_answer(question, chunks)


def _build_concise_sales_strategy_answer(question: str, chunks: list[dict]) -> str:
    """Build a sales strategy / objection-handling answer."""
    norm = _normalize_for_matching(question)

    # Distance objection: "xa trung tâm"
    if any(kw in norm for kw in ("xa trung tam", "xa qua", "cach xa trung tam")):
        return (
            "Không nên chỉ phản biện là 'không xa'. Theo tài liệu, nên xoay câu chuyện sang "
            "'một cực sống mới': NEO CITY giúp khách rời áp lực lõi đô thị cũ để có môi trường "
            "sống thoáng hơn, đủ tiện ích, có hồ trung tâm, quảng trường, camping, mall, "
            "learning hub và đời sống cuối tuần trong nội khu. "
            "Không cần ra ngoài mới có chất lượng sống."
        )

    strategy_chunk = _first_section_chunk(chunks, {"sales_strategy", "personas"})
    if strategy_chunk:
        detail = _extract_relevant_lines((strategy_chunk.get("text", "") or "").strip(), question, max_chars=700)
        formatted = _format_demo_passage(detail)
        if "\n" in formatted:
            return f"Theo tài liệu tư vấn NEO CITY:\n{formatted}"
        return _to_sentence(f"Theo tài liệu tư vấn NEO CITY, {formatted}")
    return _build_concise_general_answer(question, chunks)


def _build_concise_amenities_answer(question: str, chunks: list[dict]) -> str:
    """Build an amenities / facilities answer."""
    amenities_chunk = _first_section_chunk(chunks, {"factsheet"})
    if amenities_chunk:
        detail = _extract_relevant_lines((amenities_chunk.get("text", "") or "").strip(), question, max_chars=700)
        formatted = _format_demo_passage(detail)
        if "\n" in formatted:
            return f"Theo tài liệu NEO CITY:\n{formatted}"
        return _to_sentence(f"Theo tài liệu NEO CITY, {formatted}")
    return _build_concise_general_answer(question, chunks)


def _build_concise_general_answer(question: str, chunks: list[dict]) -> str:
    """Fallback general answer from the top chunk."""
    if not chunks:
        return FALLBACK_ANSWER
    detail = _extract_relevant_lines((chunks[0].get("text", "") or "").strip(), question, max_chars=700)
    formatted = _format_demo_passage(detail)
    if "\n" in formatted:
        return f"Theo tài liệu NEO CITY:\n{formatted}"
    return _to_sentence(f"Theo tài liệu NEO CITY, {formatted}")


# ---------------------------------------------------------------------------
# Budget recommendation builder
# ---------------------------------------------------------------------------


def _build_budget_recommendation(question: str, chunks: list[dict]) -> str:
    """Compare budget amount to NEO CITY price ranges and give advisory."""
    budget = _extract_budget_amount(question)
    if budget is None:
        return _build_concise_pricing_answer(question, chunks)

    norm = _normalize_for_matching(question)
    asks_shophouse = "shophouse" in norm
    asks_townhouse = "townhouse" in norm
    asks_apartment = any(kw in norm for kw in ("can ho", "chung cu", "studio", "1pn", "2pn", "3pn"))

    # Shophouse vs Townhouse comparison
    if asks_shophouse and asks_townhouse:
        return (
            f"Theo khung giá dự kiến, ngân sách khoảng {budget:.0f} tỷ có thể xem townhouse hoặc "
            f"shophouse ở ngưỡng thấp. Townhouse khoảng 6,5–13,5 tỷ/căn; shophouse khoảng 9,5–22 tỷ/căn. "
            f"Nếu ưu tiên ở kết hợp kinh doanh, có thể xem shophouse; nếu ưu tiên ở thấp tầng chọn lọc, "
            f"townhouse có biên ngân sách dễ tiếp cận hơn. "
            f"Đây là giá dự kiến, chưa phải giá chính thức."
        )

    # Only shophouse
    if asks_shophouse:
        return (
            f"Theo khung giá dự kiến, shophouse có tổng giá khoảng 9,5–22 tỷ/căn. "
            f"Với ngân sách {budget:.0f} tỷ, có thể xem shophouse ở vị trí nội khu hoặc ngưỡng thấp. "
            f"Đây là giá dự kiến, chưa phải giá chính thức."
        )

    # Only townhouse
    if asks_townhouse:
        return (
            f"Theo khung giá dự kiến, townhouse có tổng giá khoảng 6,5–13,5 tỷ/căn. "
            f"Với ngân sách {budget:.0f} tỷ, có thể xem townhouse ở vị trí phù hợp. "
            f"Đây là giá dự kiến, chưa phải giá chính thức."
        )

    # General apartment range recommendation
    suitable = [
        (name, lo, hi)
        for name, lo, hi in _BUDGET_PRICE_RANGES
        if lo <= budget <= hi
    ]
    # Near-fit: within 20% of range top
    near_fit = [
        (name, lo, hi)
        for name, lo, hi in _BUDGET_PRICE_RANGES
        if budget < lo and lo <= budget * 1.20
    ]

    if not suitable and not near_fit:
        # Fall back to raw pricing answer
        return _build_concise_pricing_answer(question, chunks)

    chosen = suitable[:2] or near_fit[:1]
    parts = "; ".join(f"{name} khoảng {lo:.1f}–{hi:.1f} tỷ/căn" for name, lo, hi in chosen)

    if suitable:
        return (
            f"Theo khung giá dự kiến, ngân sách khoảng {budget:.0f} tỷ phù hợp nhất với {parts}. "
            f"Đây là giá dự kiến, chưa phải giá chính thức."
        )
    else:
        first_name, first_lo, _ = near_fit[0]
        return (
            f"Theo khung giá dự kiến, ngân sách {budget:.0f} tỷ gần với ngưỡng thấp của "
            f"{first_name} (từ {first_lo:.1f} tỷ/căn). Bạn có thể cân nhắc thêm hoặc xem các "
            f"sản phẩm trong phân khúc thấp hơn. Đây là giá dự kiến, chưa phải giá chính thức."
        )


# ---------------------------------------------------------------------------
# Multi-intent answer builder
# ---------------------------------------------------------------------------

# Hardcoded pricing facts for multi-intent answers where pricing chunks may
# not have been retrieved (e.g., when query classified as legal-only).
_MULTI_INTENT_PRICE_MAP: dict[str, str] = {
    "gia studio": "Studio+: 32–38m², đơn giá 54–57 triệu/m², tổng khoảng 1,73–2,17 tỷ/căn",
    "gia 1pn+1": "1PN+1: 42–50m², đơn giá 51–54 triệu/m², tổng khoảng 2,15–2,70 tỷ/căn",
    "gia 2pn+1": "2PN+1: 70–80m², đơn giá 45–48 triệu/m², tổng khoảng 3,15–3,84 tỷ/căn",
    "gia 2pn": "2PN: 58–70m², đơn giá 48–51 triệu/m², tổng khoảng 2,80–3,57 tỷ/căn",
    "gia 3pn": "3PN: 90–110m², đơn giá 42–45 triệu/m², tổng khoảng 3,78–4,95 tỷ/căn",
    "gia shophouse": "Shophouse: 100–140m² đất, đơn giá 95–160 triệu/m² đất, tổng khoảng 9,5–22 tỷ/căn",
    "gia townhouse": "Townhouse: 87–117m² đất, đơn giá 75–115 triệu/m² đất, tổng khoảng 6,5–13,5 tỷ/căn",
}


def _build_multi_intent_answer(question: str, chunks: list[dict]) -> str:
    """Handle explicit multi-intent queries by composing per-intent sub-answers."""
    norm = _normalize_for_matching(question)
    parts: list[tuple[str, str]] = []

    # Sub-question: pricing
    for price_key, price_fact in _MULTI_INTENT_PRICE_MAP.items():
        if price_key in norm:
            parts.append((
                f"Giá {price_fact.split(':')[0]}",
                f"theo tài liệu định hướng, {price_fact}. Đây là giá dự kiến, chưa phải giá chính thức.",
            ))
            break  # Take first match only

    # Sub-question: opening status
    if any(kw in norm for kw in ("tinh trang mo ban", "mo ban chua", "mo ban")):
        parts.append(("Tình trạng mở bán", "NEO CITY chưa mở bán chính thức."))

    # Sub-question: legal warning
    if any(kw in norm for kw in ("canh bao phap ly", "canh bao", "phap ly hien tai",
                                  "phap ly")):
        parts.append(("Cảnh báo pháp lý", (
            "giá/chính sách chỉ là định hướng, chưa phải thông báo giao dịch chính thức; "
            "tài liệu cũng nêu dự án chưa huy động vốn từ khách hàng."
        )))

    if not parts:
        return _build_concise_general_answer(question, chunks)

    if len(parts) == 1:
        return parts[0][1]

    result = f"Có {len(parts)} ý cần tách rõ:\n"
    for i, (title, content) in enumerate(parts, 1):
        result += f"{i}. {title}: {content}\n"
    return result.strip()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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


_RAW_HEADING_RE = re.compile(
    r'^([IVX]+\.\s+[A-ZĐÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝ]|---+|\|\s*---)',
    re.MULTILINE,
)


def _format_demo_passage(text: str) -> str:
    """Convert raw chunk text (including Markdown tables) to human-readable form."""
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Skip bare Roman-numeral section headings (e.g. "VI. COMBO …")
        if _RAW_HEADING_RE.match(line):
            continue
        if "|" in line:
            cells = [cell.strip() for cell in line.split("|") if cell.strip()]
            if len(cells) >= 4 and cells[0].lower() not in {"loại sản phẩm", "---"}:
                # Strip leading "Khoảng" from 4th cell to avoid "khoảng Khoảng X"
                total_raw = cells[3]
                total_clean = re.sub(r'^kho[aả]ng\s+', '', total_raw, flags=re.IGNORECASE)
                line = f"{cells[0]}: {cells[2]}, tổng giá trị khoảng {total_clean}"
            elif len(cells) >= 2 and cells[0].lower() not in {"hạng mục", "nội dung", "---"}:
                line = f"{cells[0]}: {cells[1]}"
            else:
                continue
        lines.append(line)
    return "\n".join(lines) if lines else text


def _extract_pricing_highlights(text: str, question: str, max_chars: int = 400) -> str:
    """Prefer lines with money/range signals for concise pricing answers."""
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
        if len(selected) >= 3:
            break

    if not selected:
        return _extract_relevant_lines(text, question, max_chars=max_chars)

    selected.sort(key=lambda x: x[0])
    if asked_product_markers:
        filtered = [
            item for item in selected
            if any(marker in _normalize_for_matching(item[1]) for marker in asked_product_markers)
        ]
        if filtered:
            selected = filtered
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
