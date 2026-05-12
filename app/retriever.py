"""Intent-aware Qdrant retriever for NEO CITY.

Public API
----------
retrieve(question, limit=20, min_score=0.15) -> dict
    Full pipeline: classify -> filter -> embed -> search -> score-filter ->
    rerank -> return.

Helper API (also usable standalone / in tests)
-----------------------------------------------
build_filter(project, sections) -> qdrant Filter | None
embed_query(question, model_name) -> list[float]
qdrant_point_to_chunk(point) -> dict
rerank_chunks(question, chunks, classification=None, top_k=5) -> list[dict]

Legacy API (kept for backward-compatibility with earlier tasks)
---------------------------------------------------------------
search_chunks(query, intent, ...)
retrieve_chunks(query, intent, ...)
"""

from __future__ import annotations

import re
import sys
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.intent_classifier import (
    ClassificationResult,
    IntentLabel,
    classify,
    get_section_filter,
)
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


PRODUCT_TOKENS = [
    "studio",
    "1pn",
    "1pn+1",
    "2pn",
    "2pn+1",
    "3pn",
    "shophouse",
    "townhouse",
    "villa",
]
LOWRISE_PRODUCT_TOKENS = ["shophouse", "townhouse", "villa", "thap tang"]
APARTMENT_PRODUCT_TOKENS = [
    "studio",
    "1pn",
    "1pn+1",
    "2pn",
    "2pn+1",
    "3pn",
    "can ho",
    "cao tang",
]
POLICY_HINTS = ["chinh sach", "thanh toan", "chiet khau", "booking", "uu dai", "vay", "an han"]
PRICE_HINTS = ["gia", "bang gia", "don gia", "bao nhieu", "trieu", "ty"]
AMENITIES_HINTS = [
    "tien ich",
    "ho trung tam",
    "neo lake",
    "neo square",
    "quang truong",
    "r&d center",
    "innovation hub",
    "shopping mall",
    "retail street",
    "f&b",
    "camping",
    "picnic",
    "sup",
    "kayak",
    "clubhouse",
    "mam non",
    "learning hub",
]
LEGAL_STATUS_HINTS = [
    "mo ban",
    "du dieu kien",
    "dat coc",
    "nhan coc",
    "huy dong von",
    "duoc ban",
    "thu tien",
    "phap ly",
    "hdmb",
    "hop dong mua ban",
    "giay phep xay dung",
]
PRICE_VALUE_HINTS = [
    "gia",
    "bang gia",
    "gia ban",
    "bao nhieu tien",
    "tong gia tri",
    "don gia",
    "trieu/m2",
    "trieu dong/m2",
    "trieu",
    "ty",
    "price",
    "cost",
    "value",
]
PRODUCT_DETAIL_HINTS = [
    "dien tich",
    "m2",
    "met vuong",
    "loai can",
    "loai san pham",
    "san pham nao",
    "co nhung can",
    "ban giao",
    "hoan thien",
    "thap tang",
    "cao tang",
]
MARKET_HINTS = [
    "thi truong",
    "tiem nang",
    "tang gia",
    "dau tu",
    "sinh loi",
    "loi nhuan",
    "growth",
    "investment",
]
PRICING_KEYWORDS = ["giá", "bao nhiêu", "tổng giá trị", "triệu", "tỷ"]
LEGAL_KEYWORDS = [
    "mở bán",
    "huy động vốn",
    "đặt cọc",
    "đủ điều kiện",
    "pháp lý",
    "hợp đồng mua bán",
]
PERSONA_KEYWORDS = [
    "gia đình trẻ",
    "người trẻ",
    "nhà đầu tư",
    "công nghệ",
    "sáng tạo",
    "nâng cấp",
]

DEFAULT_TOP_K = 5
DEFAULT_LIMIT = 20
DEFAULT_MIN_SCORE = 0.15
DEFAULT_SCORE_THRESHOLD = 0.0
NEO_CITY_PROJECT = "NEO CITY"

SECTION_BOOSTS: dict[str, dict[str, float]] = {
    "pricing": {"pricing": 0.28, "price_sheet": 0.18, "sales_policy": 0.04},
    "legal": {"legal": 0.55},
    "persona": {"personas": 0.28, "sales_strategy": 0.12},
    "sales_strategy": {"sales_strategy": 0.26, "personas": 0.14},
    "amenities": {"factsheet": 0.28, "concept_positioning": -0.08},
    "product": {"factsheet": 0.24, "pricing": 0.18, "concept_positioning": -0.08},
    "location": {"location_connectivity": 0.24, "market": 0.18},
    "market": {"market": 0.24, "location_connectivity": 0.12},
    "sales_policy": {"sales_policy": 0.28, "price_sheet": 0.24, "pricing": -0.06},
    "concept": {"concept_positioning": 0.30, "factsheet": 0.08},
    "project_overview": {"factsheet": 0.28, "concept_positioning": 0.14},
}

TOPIC_BOOST_KEYWORDS: dict[str, tuple[str, ...]] = {
    "product": ("apartment_products", "studio_one_bedroom_policy", "two_bedroom_policy"),
    "location": ("connectivity",),
    "market": ("market_", "positioning", "drivers", "shift", "trend", "gap"),
    "sales_policy": ("payment_policy", "discount_policy", "loan", "pricing_sheet", "policy_"),
    "concept": ("brand_", "concept_", "positioning", "message_", "tagline", "manifesto"),
    "project_overview": ("project_overview", "development_", "brochure_summary"),
}
_EMBEDDER_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class RetrievedChunk:
    """A single chunk returned by the retriever (legacy dataclass)."""

    chunk_id: str
    score: float
    section: str
    topic: str
    source_title: str
    status: str
    legal_sensitivity: str
    text: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """Full result returned by ``search_chunks`` (legacy dataclass)."""

    query: str
    intent: IntentLabel
    section_filter: list[str]
    chunks: list[RetrievedChunk]


def _normalize_text(text: str) -> str:
    return re.sub(r"[^\w\s]", "", str(text).lower())


def _fold_text(text: str) -> str:
    folded = str(text).lower().replace("+", " + ")
    folded = folded.replace("\u0111", "d").replace("\u0110", "d")
    folded = unicodedata.normalize("NFD", folded)
    folded = "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^\w\s+]", " ", folded)


def _chunk_text_value(chunk: dict[str, Any], key: str) -> str:
    value = chunk.get(key, "")
    if value is None:
        return ""
    return str(value)


def _has_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _contains_term(text: str, term: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(term).replace(r"\ ", r"\s+") + r"(?!\w)"
    return re.search(pattern, text) is not None


def _has_any_term(text: str, terms: list[str]) -> bool:
    return any(_contains_term(text, term) for term in terms)


def _classification_to_parts(
    classification: ClassificationResult | dict[str, Any] | None,
) -> tuple[str, set[str], str, bool]:
    if classification is None:
        return "", set(), "", False
    if isinstance(classification, dict):
        return (
            str(classification.get("intent", "") or ""),
            {
                str(section)
                for section in classification.get("target_sections", []) or []
                if section
            },
            str(classification.get("risk_level", "") or ""),
            bool(classification.get("must_use_legal_only", False)),
        )
    return (
        classification.intent,
        set(classification.target_sections),
        classification.risk_level,
        classification.must_use_legal_only,
    )


def _product_topic_bonus(question_lower: str, searchable_text: str, topic_lower: str) -> float:
    if "shophouse" in question_lower and (
        "lowrise_pricing" in topic_lower
        or "lowrise_product_policy" in topic_lower
        or "shophouse_policy" in topic_lower
    ):
        return 0.46
    if "villa" in question_lower and (
        "lowrise_pricing" in topic_lower or "courtyard_villa_policy" in topic_lower
    ):
        return 0.18
    if any(token in question_lower for token in ("2pn", "2pn+1")) and (
        "family_apartment_policy" in topic_lower or "two_bedroom_policy" in topic_lower
    ):
        return 0.16
    if any(token in question_lower for token in ("1pn", "1pn+1", "studio")) and (
        "studio_one_bedroom_policy" in topic_lower or "apartment_pricing" in topic_lower
    ):
        return 0.14
    if any(token in searchable_text for token in PRODUCT_TOKENS):
        return 0.08
    return 0.0


def _topic_contains_any(topic_lower: str, markers: tuple[str, ...]) -> bool:
    return any(marker in topic_lower for marker in markers)


def _get_cached_embedder(model_name: str) -> Any:
    cached = _EMBEDDER_CACHE.get(model_name)
    if cached is not None:
        return cached
    embedder = _create_embedder(model_name)
    _EMBEDDER_CACHE[model_name] = embedder
    return embedder


def rerank_chunks(
    question: str,
    chunks: list[dict],
    classification: ClassificationResult | dict[str, Any] | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Rerank chunks locally using deterministic lexical and metadata boosts."""
    if not chunks:
        return []

    q_lower = question.lower()
    q_norm = _fold_text(question)
    q_words = set(_normalize_text(question).split())
    intent, target_sections, risk_level, must_use_legal_only = _classification_to_parts(
        classification
    )

    q_product_tokens = [token for token in PRODUCT_TOKENS if token in q_lower or token in q_norm]
    q_pricing_kw = [token for token in PRICE_VALUE_HINTS if _contains_term(q_norm, token)]
    q_legal_kw = [token for token in LEGAL_KEYWORDS if _contains_term(q_norm, token)]
    q_persona_kw = [token for token in PERSONA_KEYWORDS if _contains_term(q_norm, token)]
    asks_policy = _has_any_term(q_norm, POLICY_HINTS)
    asks_price = _has_any_term(q_norm, PRICE_HINTS) or _has_any_term(q_norm, PRICE_VALUE_HINTS)
    asks_lowrise = _has_any_term(q_norm, LOWRISE_PRODUCT_TOKENS)
    asks_apartment = _has_any_term(q_norm, APARTMENT_PRODUCT_TOKENS)
    asks_amenities = _has_any_term(q_norm, AMENITIES_HINTS)
    asks_legal_status = _has_any_term(q_norm, LEGAL_STATUS_HINTS)
    asks_market = _has_any_term(q_norm, MARKET_HINTS)
    asks_product_detail = _has_any_term(q_norm, PRODUCT_DETAIL_HINTS)
    asks_product_listing = _has_any_term(
        q_norm,
        ["loai can", "loai san pham", "san pham nao", "can nao", "co nhung can", "gom nhung loai nao"],
    )

    reranked: list[dict[str, Any]] = []
    for chunk in chunks:
        c_text = _chunk_text_value(chunk, "text")
        c_topic = _chunk_text_value(chunk, "topic")
        c_source_title = _chunk_text_value(chunk, "source_title")
        c_section = _chunk_text_value(chunk, "section")

        c_text_lower = c_text.lower()
        c_topic_lower = c_topic.lower()
        c_source_title_lower = c_source_title.lower()
        c_section_lower = c_section.lower()
        searchable_text = " ".join(
            value
            for value in (c_text_lower, c_topic_lower, c_source_title_lower, c_section_lower)
            if value
        )
        searchable_norm = _fold_text(searchable_text)
        c_words = set(_normalize_text(searchable_text).split())

        raw_score = chunk.get("score", 0.0)
        base_score = float(raw_score) if isinstance(raw_score, (int, float)) else 0.0
        rerank_score = base_score
        reasons = [f"base_score: {base_score:.4f}"]

        overlap = len(q_words.intersection(c_words))
        if overlap > 0:
            boost = overlap * 0.01
            rerank_score += boost
            reasons.append(f"lexical_overlap: +{boost:.4f}")

        for token in q_product_tokens:
            if token in searchable_text or token in searchable_norm:
                rerank_score += 0.05
                reasons.append(f"product_token_{token}: +0.05")
        for token in q_pricing_kw:
            if token in searchable_norm:
                rerank_score += 0.02
                reasons.append(f"pricing_kw_{token}: +0.02")
        for token in q_legal_kw:
            if token in searchable_text or token in searchable_norm:
                rerank_score += 0.05
                reasons.append(f"legal_kw_{token}: +0.05")
        for token in q_persona_kw:
            if token in searchable_text or token in searchable_norm:
                rerank_score += 0.05
                reasons.append(f"persona_kw_{token}: +0.05")

        section_boost = SECTION_BOOSTS.get(intent, {}).get(c_section, 0.0)
        if section_boost:
            rerank_score += section_boost
            reasons.append(f"intent_section_{c_section}: {section_boost:+.2f}")

        if target_sections and c_section not in target_sections:
            rerank_score -= 0.08
            reasons.append("outside_target_sections: -0.08")

        if must_use_legal_only or intent == "legal" or risk_level == "critical":
            if c_section == "legal":
                rerank_score += 0.16
                reasons.append("legal_sensitive_legal_section: +0.16")
            else:
                rerank_score -= 0.50
                reasons.append("legal_sensitive_non_legal_penalty: -0.50")

        if intent == "pricing":
            if c_section in ("pricing", "price_sheet"):
                rerank_score += 0.20
                reasons.append("intent_pricing_section_boost: +0.20")
            if not asks_policy and c_section == "sales_policy":
                rerank_score -= 0.35
                reasons.append("intent_pricing_sales_policy_penalty: -0.35")
            if asks_policy and c_section == "sales_policy":
                rerank_score += 0.35
                reasons.append("intent_pricing_asks_policy_boost: +0.35")
            if c_section == "concept_positioning":
                rerank_score -= 0.28
                reasons.append("intent_pricing_concept_penalty: -0.28")
            if asks_lowrise and "apartment_pricing" in c_topic_lower:
                rerank_score -= 0.28
                reasons.append("intent_pricing_shophouse_apartment_penalty: -0.28")
            if asks_lowrise:
                if "lowrise_pricing" in c_topic_lower:
                    rerank_score += 0.48
                    reasons.append("intent_pricing_lowrise_topic: +0.48")
                if c_section == "price_sheet" and any(
                    marker in c_topic_lower for marker in ("lowrise", "shophouse", "townhouse", "villa")
                ):
                    rerank_score += 0.32
                    reasons.append("intent_pricing_lowrise_price_sheet: +0.32")
            elif asks_apartment:
                if "apartment_pricing" in c_topic_lower:
                    rerank_score += 0.42
                    reasons.append("intent_pricing_topic_apartment: +0.42")
                if c_section == "price_sheet" and any(
                    marker in c_topic_lower
                    for marker in ("studio", "one_bedroom", "1pn", "two_bedroom", "2pn", "three_bedroom", "3pn", "family")
                ):
                    rerank_score += 0.28
                    reasons.append("intent_pricing_apartment_price_sheet: +0.28")
            if asks_product_detail or asks_product_listing:
                if c_section == "factsheet":
                    rerank_score -= 0.18
                    reasons.append("intent_pricing_product_detail_factsheet_penalty: -0.18")
                if "apartment_pricing" in c_topic_lower and (asks_apartment or not asks_lowrise):
                    rerank_score += 0.24
                    reasons.append("intent_pricing_product_detail_apartment_boost: +0.24")
                if _topic_contains_any(
                    c_topic_lower,
                    ("studio_one_bedroom_policy", "two_bedroom_policy", "three_bedroom_policy"),
                ):
                    rerank_score += 0.18
                    reasons.append("intent_pricing_product_policy_topic_boost: +0.18")
                if asks_lowrise and (
                    "lowrise_pricing" in c_topic_lower or "lowrise_product_policy" in c_topic_lower
                ):
                    rerank_score += 0.24
                    reasons.append("intent_pricing_product_detail_lowrise_boost: +0.24")
            product_bonus = _product_topic_bonus(q_lower, searchable_norm, c_topic_lower)
            if product_bonus:
                rerank_score += product_bonus
                reasons.append(f"intent_pricing_product_bonus: +{product_bonus:.2f}")
            if q_product_tokens and c_section == "factsheet":
                rerank_score -= 0.22
                reasons.append("intent_pricing_product_factsheet_penalty: -0.22")

        elif intent == "sales_policy":
            if c_section == "sales_policy":
                rerank_score += 0.36
                reasons.append("intent_sales_policy_section_boost: +0.36")
            if c_section == "price_sheet":
                rerank_score += 0.18
                reasons.append("intent_sales_policy_price_sheet_boost: +0.18")
            if c_section == "pricing":
                rerank_score -= 0.24
                reasons.append("intent_sales_policy_pricing_penalty: -0.24")
            if _topic_contains_any(
                c_topic_lower,
                ("payment_policy", "booking_policy", "discount_policy", "combo_", "early_buyer", "supplemental_incentives"),
            ):
                rerank_score += 0.28
                reasons.append("intent_sales_policy_topic_boost: +0.28")
            if not asks_price and "pricing_principles" in c_topic_lower:
                rerank_score -= 0.40
                reasons.append("intent_sales_policy_pricing_principles_penalty: -0.40")

        elif intent == "product":
            if not asks_price:
                if c_section == "factsheet":
                    rerank_score += 0.28
                    reasons.append("intent_product_factsheet_boost: +0.28")
                if "pricing_principles" in c_topic_lower:
                    rerank_score -= 0.32
                    reasons.append("intent_product_pricing_penalty: -0.32")
                # Boost product-related topics in factsheet
                if any(marker in c_topic_lower for marker in ("apartment_products", "product_structure", "product_mix", "lowrise_product")):
                    rerank_score += 0.22
                    reasons.append("intent_product_topic_boost: +0.22")
                if asks_product_detail and c_section == "pricing" and _topic_contains_any(
                    c_topic_lower,
                    ("apartment_pricing", "lowrise_pricing", "studio_one_bedroom_policy", "two_bedroom_policy", "three_bedroom_policy", "lowrise_product_policy"),
                ):
                    rerank_score += 0.22
                    reasons.append("intent_product_detail_pricing_boost: +0.22")
            else:
                if c_section == "pricing":
                    rerank_score += 0.35
                    reasons.append("intent_product_asks_price_pricing_boost: +0.35")
                elif c_section == "factsheet":
                    rerank_score -= 0.25
                    reasons.append("intent_product_asks_price_factsheet_penalty: -0.25")

        elif intent == "location":
            if c_section == "location_connectivity":
                if not asks_market:
                    rerank_score += 0.42
                    reasons.append("intent_location_connectivity_boost: +0.42")
                else:
                    rerank_score += 0.15
                    reasons.append("intent_location_connectivity_weak_boost: +0.15")
            if c_section == "market" and not asks_market:
                rerank_score -= 0.26
                reasons.append("intent_location_market_penalty: -0.26")

        elif intent == "amenities":
            if c_topic_lower == "amenities":
                rerank_score += 0.20
                reasons.append("intent_amenities_topic_boost: +0.20")

        elif intent == "legal":
            if "legal_status_and_warnings" in c_topic_lower:
                rerank_score += 0.65
                reasons.append("intent_legal_topic_match: +0.65")
            if any(
                term in searchable_text or term in searchable_norm
                for term in (
                    "phap ly",
                    "pháp lý",
                    "mo ban",
                    "mở bán",
                    "dat coc",
                    "đặt cọc",
                    "huy dong von",
                    "huy động vốn",
                    "neolab",
                    "chủ đầu tư",
                    "chu dau tu",
                )
            ):
                rerank_score += 0.18
                reasons.append("intent_legal_keyword_match: +0.18")
            if asks_legal_status and c_section != "legal":
                rerank_score -= 0.55
                reasons.append("intent_legal_non_legal_penalty: -0.55")
            if c_section != "legal":
                rerank_score -= 0.60
                reasons.append("intent_legal_non_legal_section_penalty: -0.60")

        elif intent == "persona":
            if "gia đình trẻ" in q_lower and "buyer_persona_family" in c_topic_lower:
                rerank_score += 0.30
                reasons.append("intent_persona_family: +0.30")
            if "người trẻ" in q_lower and "buyer_persona_young_professional" in c_topic_lower:
                rerank_score += 0.20
                reasons.append("intent_persona_young: +0.20")
            if "nhà đầu tư" in q_lower and "investor" in c_topic_lower:
                rerank_score += 0.20
                reasons.append("intent_persona_investor: +0.20")
            if "gia dinh tre" in q_norm and "buyer_persona_family" in c_topic_lower:
                rerank_score += 0.30
                reasons.append("intent_persona_family_norm: +0.30")
            if "nguoi tre" in q_norm and "buyer_persona_young_professional" in c_topic_lower:
                rerank_score += 0.20
                reasons.append("intent_persona_young_norm: +0.20")
            if "nha dau tu" in q_norm and "investor" in c_topic_lower:
                rerank_score += 0.20
                reasons.append("intent_persona_investor_norm: +0.20")
            if any(term in q_norm for term in ("khach hang muc tieu", "persona", "tep khach", "nguoi mua")) and any(
                marker in c_topic_lower for marker in ("core_buyer_persona", "target_customer_segments")
            ):
                rerank_score += 0.24
                reasons.append("intent_persona_core_segments: +0.24")

        elif intent == "sales_strategy":
            if any(
                keyword in searchable_text
                for keyword in ("objection", "objection_handling", "rào cản", "xử lý", "tư vấn")
            ):
                rerank_score += 0.15
                reasons.append("intent_sales_strategy_match: +0.15")
            if any(
                keyword in searchable_norm
                for keyword in ("objection", "objection_handling", "rao can", "xu ly", "tu van", "phan doi", "thuyet phuc", "chot")
            ):
                rerank_score += 0.18
                reasons.append("intent_sales_strategy_match_norm: +0.18")

        elif intent == "amenities":
            if any(
                keyword in searchable_text
                for keyword in ("hồ trung tâm", "neo square", "r&d center", "tiện ích", "amenities")
            ):
                rerank_score += 0.15
                reasons.append("intent_amenities_match: +0.15")
            if "amenities" in c_topic_lower:
                rerank_score += 0.35
                reasons.append("intent_amenities_topic: +0.35")
            if asks_amenities and any(
                keyword in searchable_norm
                for keyword in ("ho trung tam", "neo square", "r&d center", "tien ich", "amenities", "neo lake", "learning hub")
            ):
                rerank_score += 0.15
                reasons.append("intent_amenities_match_norm: +0.15")
            if asks_amenities and any(
                marker in c_topic_lower for marker in ("brochure_summary", "brand_positioning", "strategic_differentiation")
            ):
                rerank_score -= 0.24
                reasons.append("intent_amenities_generic_penalty: -0.24")

        elif intent == "product":
            if not asks_price and c_topic_lower == "pricing_principles":
                rerank_score -= 0.26
                reasons.append("intent_product_pricing_principles_penalty: -0.26")
            if any(term in q_norm for term in ("ban giao", "hoan thien")):
                if c_section == "factsheet":
                    rerank_score += 0.18
                    reasons.append("intent_product_handover_factsheet_boost: +0.18")
                if "pricing_principles" in c_topic_lower:
                    rerank_score -= 0.24
                    reasons.append("intent_product_handover_pricing_penalty: -0.24")
            product_bonus = _product_topic_bonus(q_lower, searchable_norm, c_topic_lower)
            if product_bonus:
                rerank_score += product_bonus
                reasons.append(f"intent_product_topic_bonus: +{product_bonus:.2f}")
            if any(term in q_norm for term in ("co nhung can nao", "loai can", "san pham nao", "gom nhung loai nao", "can nao", "co can")) and (
                "apartment_products" in c_topic_lower
                or "lowrise_product" in c_topic_lower
                or "product_structure" in c_topic_lower
                or "product_mix" in c_topic_lower
            ):
                rerank_score += 0.24
                reasons.append("intent_product_mix_match: +0.24")
            if any(
                keyword in searchable_text
                for keyword in ("diện tích", "loại căn", "loại hình", "mặt bằng", "phòng ngủ")
            ):
                rerank_score += 0.08
                reasons.append("intent_product_detail_match: +0.08")
            if any(
                keyword in searchable_norm
                for keyword in ("dien tich", "loai can", "loai hinh", "mat bang", "phong ngu", "ban giao", "hoan thien", "dual key", "so tang")
            ):
                rerank_score += 0.08
                reasons.append("intent_product_detail_match_norm: +0.08")

        elif intent == "location":
            if "connectivity" in c_topic_lower or "transport_connectivity" in c_topic_lower:
                rerank_score += 0.16
                reasons.append("intent_location_connectivity: +0.16")
            if any(
                keyword in searchable_text
                for keyword in ("nội bài", "sân bay", "vành đai", "giao thông", "kết nối")
            ):
                rerank_score += 0.10
                reasons.append("intent_location_keyword_match: +0.10")
            if any(
                keyword in searchable_norm
                for keyword in ("noi bai", "san bay", "vanh dai", "giao thong", "ket noi", "me linh", "dong anh", "soc son", "vo van kiet")
            ):
                rerank_score += 0.10
                reasons.append("intent_location_keyword_match_norm: +0.10")

        elif intent == "market":
            if c_section == "market":
                rerank_score += 0.10
                reasons.append("intent_market_section: +0.10")
            if any(
                keyword in searchable_text
                for keyword in ("thị trường", "tiềm năng", "xu hướng", "động lực", "dư địa")
            ):
                rerank_score += 0.10
                reasons.append("intent_market_keyword_match: +0.10")
            if any(
                keyword in searchable_norm
                for keyword in ("thi truong", "tiem nang", "xu huong", "dong luc", "du dia", "thanh khoan", "rui ro")
            ):
                rerank_score += 0.10
                reasons.append("intent_market_keyword_match_norm: +0.10")

        elif intent == "sales_policy":
            if c_section == "sales_policy":
                rerank_score += 0.12
                reasons.append("intent_sales_policy_section_bonus: +0.12")
            if any(
                keyword in searchable_text
                for keyword in ("thanh toán", "chiết khấu", "booking", "vay", "ân hạn", "chính sách")
            ):
                rerank_score += 0.12
                reasons.append("intent_sales_policy_keyword_match: +0.12")
            if any(
                keyword in searchable_norm
                for keyword in ("thanh toan", "chiet khau", "booking", "vay", "an han", "chinh sach", "phi quan ly")
            ):
                rerank_score += 0.12
                reasons.append("intent_sales_policy_keyword_match_norm: +0.12")
            if any(
                marker in c_topic_lower
                for marker in ("payment_policy", "booking_policy", "discount_policy", "combo_family", "combo_young_professional")
            ):
                rerank_score += 0.18
                reasons.append("intent_sales_policy_topic_bonus: +0.18")

        elif intent == "concept":
            if c_section == "concept_positioning":
                rerank_score += 0.12
                reasons.append("intent_concept_section: +0.12")

        elif intent == "project_overview":
            if c_section == "factsheet":
                rerank_score += 0.14
                reasons.append("intent_overview_factsheet: +0.14")
            elif c_section == "concept_positioning":
                rerank_score += 0.06
                reasons.append("intent_overview_concept: +0.06")

        for topic_marker in TOPIC_BOOST_KEYWORDS.get(intent, ()):
            if topic_marker in c_topic_lower:
                rerank_score += 0.08
                reasons.append(f"intent_topic_marker_{topic_marker}: +0.08")

        new_chunk = dict(chunk)
        new_chunk["rerank_score"] = rerank_score
        new_chunk["rerank_reasons"] = reasons
        reranked.append(new_chunk)

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return reranked[:top_k]


def build_filter(project: str, sections: list[str]) -> qmodels.Filter | None:
    """Build a Qdrant payload filter that restricts to *project* and *sections*."""
    must: list[qmodels.Condition] = []

    if project:
        must.append(
            qmodels.FieldCondition(
                key="project",
                match=qmodels.MatchValue(value=project),
            )
        )

    if len(sections) == 1:
        must.append(
            qmodels.FieldCondition(
                key="section",
                match=qmodels.MatchValue(value=sections[0]),
            )
        )
    elif len(sections) > 1:
        must.append(
            qmodels.Filter(
                should=[
                    qmodels.FieldCondition(
                        key="section",
                        match=qmodels.MatchValue(value=sec),
                    )
                    for sec in sections
                ]
            )
        )

    if not must:
        return None

    return qmodels.Filter(must=must)


def embed_query(question: str, model_name: str) -> list[float]:
    """Embed *question* using FastEmbed and return the float vector."""
    embedder = _get_cached_embedder(model_name)
    for vec in embedder.embed([question]):
        if hasattr(vec, "tolist"):
            return vec.tolist()
        return list(vec)
    raise RuntimeError("FastEmbed returned no vectors for the query.")


def qdrant_point_to_chunk(point: Any) -> dict[str, Any]:
    """Convert a Qdrant ScoredPoint to a chunk dict preserving payload fields."""
    payload = point.payload or {}
    chunk = dict(payload)
    chunk.update(
        {
            "id": payload.get("id", str(point.id)),
            "score": point.score,
            "project": payload.get("project", ""),
            "section": payload.get("section", ""),
            "topic": payload.get("topic", ""),
            "source_doc": payload.get("source_doc", ""),
            "source_title": payload.get("source_title", ""),
            "status": payload.get("status", ""),
            "legal_sensitivity": payload.get("legal_sensitivity", ""),
            "version": payload.get("version", ""),
            "text": payload.get("text", ""),
        }
    )
    return chunk


def retrieve(
    question: str,
    *,
    limit: int = DEFAULT_LIMIT,
    min_score: float = DEFAULT_MIN_SCORE,
    top_k: int = DEFAULT_TOP_K,
    client: QdrantClient | None = None,
    embedder: Any | None = None,
) -> dict[str, Any]:
    """Full production retrieval path for the chatbot.

    Steps
    -----
    1. Classify *question* to get intent, risk_level, target_sections,
       must_use_legal_only.
    2. If must_use_legal_only -> override target_sections = ["legal"].
    3. If target_sections is empty -> return early with reason="unknown_intent".
    4. Build Qdrant filter: project="NEO CITY" + section in target_sections.
    5. Embed *question* with FastEmbed.
    6. Search Qdrant via ``client.query_points(...).points``.
    7. Drop chunks with score < min_score.
    8. Rerank locally and keep top_k chunks.
    9. If no chunks remain after filtering -> reason="no_chunks_above_min_score".
    10. Return structured result dict.
    """
    settings = get_settings()

    clf = classify(question)
    intent = clf.intent
    risk_level = clf.risk_level
    must_use_legal_only = clf.must_use_legal_only
    target_sections = list(clf.target_sections)

    if must_use_legal_only:
        target_sections = ["legal"]

    if not target_sections:
        return {
            "question": question,
            "intent": intent,
            "risk_level": risk_level,
            "target_sections": [],
            "must_use_legal_only": must_use_legal_only,
            "chunks": [],
            "reason": "unknown_intent",
        }

    qdrant_filter = build_filter(NEO_CITY_PROJECT, target_sections)

    if embedder is None:
        query_vector = embed_query(question, settings.embedding_model)
    else:
        query_vector = _embed_query_from_embedder(embedder, question)

    if client is None:
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )

    raw_points = client.query_points(
        collection_name=settings.qdrant_collection_name,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=limit,
        with_payload=True,
    ).points

    chunks = [qdrant_point_to_chunk(point) for point in raw_points if point.score >= min_score]

    if chunks:
        chunks = rerank_chunks(question, chunks, classification=clf, top_k=top_k)

    reason: str | None = None
    if not chunks:
        reason = "no_chunks_above_min_score"

    return {
        "question": question,
        "intent": intent,
        "risk_level": risk_level,
        "target_sections": target_sections,
        "must_use_legal_only": must_use_legal_only,
        "chunks": chunks,
        "reason": reason,
    }


def _embed_query_from_embedder(embedder: Any, question: str) -> list[float]:
    """Embed *question* using an already-instantiated embedder object."""
    for vec in embedder.embed([question]):
        if hasattr(vec, "tolist"):
            return vec.tolist()
        return list(vec)
    raise RuntimeError("Embedder returned no vectors for the query.")


def _chunk_uuid(chunk_id: str) -> str:
    """Deterministic UUID from chunk id — mirrors the upsert script."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _build_section_filter(sections: list[str]) -> qmodels.Filter | None:
    """Build a section-only Qdrant filter (legacy — no project filter)."""
    if not sections:
        return None
    if len(sections) == 1:
        return qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="section",
                    match=qmodels.MatchValue(value=sections[0]),
                )
            ]
        )
    return qmodels.Filter(
        should=[
            qmodels.FieldCondition(
                key="section",
                match=qmodels.MatchValue(value=sec),
            )
            for sec in sections
        ]
    )


def _point_to_chunk(point: Any) -> RetrievedChunk:
    """Convert a Qdrant ScoredPoint into a RetrievedChunk (legacy)."""
    payload = point.payload or {}
    return RetrievedChunk(
        chunk_id=payload.get("id", str(point.id)),
        score=point.score,
        section=payload.get("section", ""),
        topic=payload.get("topic", ""),
        source_title=payload.get("source_title", ""),
        status=payload.get("status", ""),
        legal_sensitivity=payload.get("legal_sensitivity", ""),
        text=payload.get("text", ""),
        payload=payload,
    )


def _create_embedder(model_name: str) -> Any:
    """Instantiate a FastEmbed TextEmbedding model (legacy)."""
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name)


def _embed_query(embedder: Any, query: str) -> list[float]:
    """Embed a single query string (legacy)."""
    return _embed_query_from_embedder(embedder, query)


def search_chunks(
    query: str,
    intent: IntentLabel,
    *,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    client: QdrantClient | None = None,
    embedder: Any | None = None,
) -> SearchResult:
    """Search Qdrant using a pre-classified legacy intent label.

    This API is preserved for backward compatibility. Production chatbot
    retrieval should use ``retrieve()`` so that rich classification, legal-only
    restrictions, project filtering, and local reranking are applied.
    """
    settings = get_settings()

    if client is None:
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )

    if embedder is None:
        embedder = _create_embedder(settings.embedding_model)

    section_filter_list = get_section_filter(intent)
    qdrant_filter = _build_section_filter(section_filter_list)
    query_vector = _embed_query(embedder, query)

    hits = client.query_points(
        collection_name=settings.qdrant_collection_name,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=top_k,
        score_threshold=score_threshold if score_threshold > 0.0 else None,
        with_payload=True,
    ).points

    chunks = [_point_to_chunk(hit) for hit in hits]

    return SearchResult(
        query=query,
        intent=intent,
        section_filter=section_filter_list,
        chunks=chunks,
    )


def retrieve_chunks(
    query: str,
    intent: IntentLabel,
    *,
    top_k: int = DEFAULT_TOP_K,
    client: QdrantClient | None = None,
    embedder: Any | None = None,
) -> list[dict[str, Any]]:
    """Legacy helper that returns raw payload dicts for earlier tasks."""
    result = search_chunks(
        query,
        intent,
        top_k=top_k,
        client=client,
        embedder=embedder,
    )
    return [chunk.payload for chunk in result.chunks]


def _cli_main() -> None:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="NEO CITY intent-aware retriever CLI")
    parser.add_argument("question", help="User question to retrieve chunks for")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Max chunks to fetch")
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="Minimum similarity score (default 0.15)",
    )
    args = parser.parse_args()

    print(f"\nQuery: {args.question!r}\n{'=' * 60}")
    result = retrieve(args.question, limit=args.limit, min_score=args.min_score)

    print(f"Intent       : {result['intent']}")
    print(f"Risk level   : {result['risk_level']}")
    print(f"Sections     : {result['target_sections']}")
    print(f"Legal only   : {result['must_use_legal_only']}")
    print(f"Reason       : {result['reason']}")
    print(f"Chunks found : {len(result['chunks'])}\n")

    for index, chunk in enumerate(result["chunks"], 1):
        print(
            f"  [{index}] score={chunk.get('score', 0.0):.4f}  "
            f"rerank_score={chunk.get('rerank_score', 0.0):.4f}  "
            f"section={chunk.get('section', '')}"
        )
        print(f"      id={chunk.get('id', '')}")
        print(f"      topic={chunk.get('topic', '')}")
        print(f"      legal_sensitivity={chunk.get('legal_sensitivity', '')}")
        print(f"      text={chunk.get('text', '')[:120]!r}")
        print()


if __name__ == "__main__":
    _cli_main()
