"""app/intent_classifier.py

Rule-based intent classifier for the NEO CITY RAG assistant.

Two public interfaces
---------------------
**Legacy (Task 3/4 backward-compat)**
  classify_intent(query) -> IntentLabel          # "legal" | "pricing" | ...
  get_section_filter(intent) -> list[str]
  INTENT_SECTION_FILTER                          # dict used by retriever

**Task 5 - rich classifier**
  classify(query) -> ClassificationResult
  ClassificationResult.to_dict() -> dict

The legacy interface is kept intact so that app.retriever and its tests
continue to work unchanged.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

# ===========================================================================
# LEGACY interface (unchanged -- backward-compat with app.retriever)
# ===========================================================================

IntentLabel = Literal[
    "general",
    "pricing",
    "legal",
    "sales_policy",
    "market",
]

INTENT_SECTION_FILTER: dict[str, list[str]] = {
    "legal": ["legal"],
    "pricing": ["pricing", "price_sheet"],
    "sales_policy": ["sales_policy"],
    "market": ["market"],
    "general": [],
}

_LEGACY_RULES: list[tuple[str, list[str]]] = [
    (
        "legal",
        [
            "phap ly",
            "phap li",
            "giay phep",
            "so do",
            "so hong",
            "mo ban",
            "dieu kien mo ban",
            "du dieu kien",
            "huy dong von",
            "dat coc",
            "dat cho",
            "gop von",
            "hop dong",
            "legal",
            "license",
            "permit",
            "certificate",
            "ownership",
        ],
    ),
    (
        "pricing",
        [
            "gia",
            "gia ban",
            "gia can",
            "gia ho",
            "gia m2",
            "gia/m2",
            "bang gia",
            "muc gia",
            "gia du kien",
            "gia tham khao",
            "gia thuc",
            "gia tri",
            "price",
            "pricing",
            "cost",
            "value per sqm",
            "psm",
        ],
    ),
    (
        "sales_policy",
        [
            "chinh sach",
            "chiet khau",
            "uu dai",
            "thanh toan",
            "phuong thuc thanh toan",
            "tra gop",
            "ngan hang ho tro",
            "ho tro lai suat",
            "vay von",
            "vay ngan hang",
            "policy",
            "discount",
            "installment",
            "payment plan",
            "bank support",
        ],
    ),
    (
        "market",
        [
            "thi truong",
            "xu huong",
            "tiem nang",
            "trien vong",
            "tang truong",
            "loi nhuan",
            "sinh loi",
            "dau tu",
            "ty suat",
            "market",
            "trend",
            "growth",
            "investment",
            "return",
            "roi",
            "profit",
        ],
    ),
]


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace (no diacritic removal)."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _remove_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics using Unicode NFD decomposition.

    Steps
    -----
    1. Replace ``\u0111`` (đ) and ``\u0110`` (Đ) manually because they do
       not decompose under NFD.
    2. Apply ``unicodedata.normalize("NFD", ...)`` to split characters into
       base + combining marks.
    3. Filter out all Unicode category ``Mn`` (non-spacing combining marks).

    Examples
    --------
    >>> _remove_diacritics("pháp lý")
    'phap ly'
    >>> _remove_diacritics("đặt cọc")
    'dat coc'
    """
    # đ / Đ do not decompose under NFD; replace manually.
    text = text.replace("\u0111", "d").replace("\u0110", "D")
    # NFD decompose then strip combining marks.
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def _norm_legacy(text: str) -> str:
    """Normalize for legacy classifier: lowercase + diacritic removal + collapse."""
    lowered = text.lower()
    removed = _remove_diacritics(lowered)
    return re.sub(r"\s+", " ", removed.strip())


def classify_intent(query: str) -> str:
    """Classify a query into a legacy 5-intent label (backward-compat).

    Rules are checked in priority order; first match wins.
    Falls back to ``"general"`` if no keyword matches.
    """
    if not query or not query.strip():
        return "general"

    normalized = _norm_legacy(query)
    for intent, keywords in _LEGACY_RULES:
        for kw in keywords:
            if kw in normalized:
                return intent
    return "general"


def get_section_filter(intent: str) -> list[str]:
    """Return section list to filter on for the given legacy intent label."""
    return INTENT_SECTION_FILTER.get(intent, [])


# ===========================================================================
# TASK-5 interface -- rich 12-intent classifier
# ===========================================================================


@dataclass
class ClassificationResult:
    """Result returned by :func:`classify`.

    Attributes
    ----------
    intent:
        One of 12 supported intent labels.
    target_sections:
        Qdrant sections to search in. An empty list means the query should
        not trigger retrieval and the caller should fall back.
    risk_level:
        ``"low"`` | ``"medium"`` | ``"high"`` | ``"critical"``.
    must_use_legal_only:
        When ``True`` the retriever MUST restrict to ``section = "legal"``
        only and must not infer from marketing or sales content.
    """

    intent: str
    target_sections: list[str]
    risk_level: str
    must_use_legal_only: bool

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "target_sections": list(self.target_sections),
            "risk_level": self.risk_level,
            "must_use_legal_only": self.must_use_legal_only,
        }


# ---------------------------------------------------------------------------
# Keyword lists for rich classifier -- checked in descending priority order.
# All keywords are already lowercased; matching is done on _normalize(query).
# ---------------------------------------------------------------------------

# Priority 1 -- Legal: deposit, sale opening, fundraising, legal status
_LEGAL_DIRECT: list[str] = [
    "mo ban",
    "dat coc",
    "du dieu kien",
    "huy dong von",
    "phap ly",
    "phap li",
    "so do",
    "so hong",
    "giay phep xay dung",
    "kinh doanh bat dong san",
    "chuyen nhuong hop dong",
    "dieu kien kinh doanh",
    "chinh thuc mo",
    "duoc phep ban",
    "thong bao mo ban",
    "ky hop dong mua ban",
    "hop dong mua ban",
    "giay phep kinh doanh",
    "da du dieu kien",
    "da duoc phep",
    "hoan cong",
    "giay chung nhan",
    "quyen su dung dat",
    "thu tuc phap ly",
    "phap ly du an",
    "hdmb",
    "phe duyet quy hoach",
    "duoc phe duyet quy hoach",
    "booking co hop phap khong",
    "duoc thu tien",
    "nhan coc",
    "co duoc nhan coc khong",
    "du an ma",
    "nguon goc",
    "dam bao dung han",
    "chu dau tu neo city la ai",
    "chu dau tu neolab co uy tin khong",
    "duoc ban chua",
    "duoc thu tien chua",
    "chinh thuc chua",
    "giay phep",
    "so do",
    "so hong",
    "hdmb",
    "hop dong mua ban",
]

# ---------------------------------------------------------------------------
# Compound helper for reversed-order profit+guarantee phrases
# e.g. "loi nhuan dau tu co dam bao khong?" where profit comes before guarantee
# ---------------------------------------------------------------------------
_PROFIT_CORE: list[str] = [
    "loi nhuan",
    "sinh loi",
    "tang gia",
    "loi tuc",
]

_GUARANTEE_CORE: list[str] = [
    "dam bao",
    "bao dam",
    "cam ket",
    "chac chan",
    "chac thang",
    "chac an",
]


def _has_profit_guarantee_compound(norm: str) -> bool:
    """Return True if norm contains BOTH a profit term AND a guarantee term.

    This catches reversed-order phrases such as "loi nhuan ... co dam bao"
    that the single-phrase ``_GUARANTEED_PROFIT`` list would miss.
    """
    return any(kw in norm for kw in _PROFIT_CORE) and any(
        kw in norm for kw in _GUARANTEE_CORE
    )


# Priority 2 -- Guaranteed profit / appreciation (critical but NOT legal-only)
_GUARANTEED_PROFIT: list[str] = [
    "cam ket loi nhuan",
    "cam ket tang gia",
    "cam ket sinh loi",
    "chac tang gia",
    "chac sinh loi",
    "chac thang",
    "chac an",
    "dam bao loi nhuan",
    "dam bao tang gia",
    "dam bao sinh loi",
    "chac chan lai",
    "chac chan tang",
    "guaranteed profit",
    "guaranteed return",
]

# Priority 3 -- Pricing (high risk)
_PRICING: list[str] = [
    "bang gia",
    "don gia",
    "tong gia",
    "gia bao nhieu",
    "gia ban",
    "gia the nao",
    "bao nhieu tien",
    "bao nhieu trieu",
    "bao nhieu ty",
    "gia m2",
    "gia/m",
    "muc gia",
    "gia can",
    "gia du kien",
    "tam gia",
    "gia khoi diem",
    "khoang bao nhieu",
    "price",
    "cost",
    "value",
]

# Priority 4 -- Sales policy, payment, loan, discount (high risk)
_SALES_POLICY_NEW: list[str] = [
    "chinh sach thanh toan",
    "tien do thanh toan",
    "phuong thuc thanh toan",
    "chinh sach ban hang",
    "chinh sach vay",
    "chinh sach uu dai",
    "vay ngan hang",
    "vay von",
    "ho tro vay",
    "lai suat",
    "chiet khau",
    "discount",
    "tra gop",
    "tra cham",
    "an han no goc",
    "an han goc",
    "booking",
    "dat booking",
    "thanh toan nhanh",
    "chinh sach",
    "thanh toan",
    "phi quan ly",
    "mien phi quan ly",
    "uu dai",
    "hỗ trợ vay",
    "ân hạn",
    "voucher",
    "combo",
    "chinh sach",
]

# Priority 5 -- Market / investment potential (medium risk)
_MARKET_NEW: list[str] = [
    "thi truong bat dong san",
    "xu huong thi truong",
    "xu huong",
    "tiem nang",
    "trien vong",
    "tang truong",
    "loi nhuan",
    "sinh loi",
    "ty suat sinh loi",
    "ty suat",
    "hoan von",
    "roi",
    "so sanh du an",
    "bat dong san me linh",
    "du bao",
    "thi truong",
    "thanh khoan",
    "rui ro",
    "xung dang",
]

# Priority 6 -- Location / connectivity / infrastructure (medium risk)
_LOCATION: list[str] = [
    "ket noi san bay",
    "san bay noi bai",
    "san bay",
    "duong vanh dai",
    "vanh dai 4",
    "cau nhat tan",
    "cau thang long",
    "metro",
    "duong sat do thi",
    "ha tang giao thong",
    "co so ha tang",
    "cach trung tam",
    "ket noi me linh",
    "vung ven ha noi",
    "gian dan",
    "huong tay bac",
    "di chuyen tu",
    "di lai",
    "vi tri du an",
    "khoang cach",
    "benh vien",
    "truong hoc",
    "gan trung tam",
    "o dau",
    "vi tri",
    "noi bai",
    "dong anh",
    "soc son",
    "vo van kiet",
    "cau hong ha",
    "lien ket vung",
    "ha tang",
    "khu cong nghiep",
    "ho tay",
    "quoc lo 23",
    "noi thanh",
    "logistics",
    "viec lam",
    "toa lac",
    "ngap lut",
    "giao thong",
    "dong anh va me linh",
    "me linh va dong anh",
]

# Priority 7 -- Sales strategy / advisor script (medium risk)
_SALES_STRATEGY: list[str] = [
    "tu van the nao",
    "tu van ra sao",
    "nen tu van",
    "xu ly tu choi",
    "kich ban ban",
    "kich ban tu van",
    "thuyet phuc",
    "khang cu",
    "phan bac",
    "chot hop dong",
    "ho tro sales",
    "chien luoc ban",
    "nen gioi thieu",
    "cach tu van",
    "tu van sao",
    "xu ly the nao",
    "khach che",
    "noi gi voi",
    "xu ly phan bac",
    "xu ly objection",
    "rao can",
    "phan doi",
    "objection",
    "xu ly",
    "sales noi gi",
    "chot khach",
    "khach lo",
    "xa trung tam",
    "vung ven",
    "thieu tien ich",
]

# Priority 8 -- Persona / target customer (low risk)
_PERSONA: list[str] = [
    "khach hang muc tieu",
    "khach muc tieu",
    "doi tuong khach hang",
    "gia dinh tre",
    "cap vo chong tre",
    "nguoi mua lan dau",
    "nhom khach",
    "phan khuc khach",
    "phu hop voi ai",
    "ai phu hop",
    "ai nen mua",
    "danh cho ai",
    "danh cho doi tuong",
    "muc tieu la ai",
    "khach hang la ai",
    "nha dau tu trung luu",
    "nha dau tu",
    "can dau tien",
    "nguoi tre mua",
    "tep khach",
    "nguoi mua",
    "nguoi tre",
    "cong nghe",
    "sang tao",
    "hybrid work",
    "remote work",
    "nang cap chat luong song",
    "thu nhap 20-30 trieu",
    "persona",
]

# Priority 9 -- Product / unit types / floor plans (medium risk)
_PRODUCT: list[str] = [
    "loai can",
    "dien tich can",
    "can ho bao nhieu phong",
    "1pn",
    "2pn",
    "3pn",
    "studio+",
    "studio plus",
    "can 1 phong",
    "can 2 phong",
    "can 3 phong",
    "mat bang dien hinh",
    "thiet ke can",
    "so phong ngu",
    "penthouse",
    "can goc",
    "can thong tang",
    "loai hinh san pham",
    "san pham nao",
    "san pham",
    "can nao",
    "co nhung can",
    "studio",
    "1pn+1",
    "2pn+1",
    "shophouse",
    "townhouse",
    "villa",
    "thap tang",
    "cao tang",
    "dual key",
    "ban giao",
    "hoan thien",
    "ban giao tho",
    "so tang",
    "dien tich",
    "co can",
    "co nhung",
    "bao nhieu m2",
    "bao nhieu met vuong",
    "m2",
]

# Priority 10 -- Amenities / internal facilities (low risk)
_AMENITIES: list[str] = [
    "tien ich",
    "ho trung tam",
    "ho nuoc",
    "cong vien",
    "gym",
    "be boi",
    "trung tam thuong mai",
    "shophouse",
    "clubhouse",
    "san choi tre em",
    "khu vui choi",
    "tien nghi",
    "noi khu",
    "tien ich noi khu",
    "duong dao",
    "khong gian xanh",
    "neo square",
    "r&d center",
    "khu r&d",
    "cho tre em",
    "khu cho tre em",
    "neo lake",
    "quang truong",
    "shopping mall",
    "retail street",
    "f&b",
    "camping",
    "picnic",
    "sup",
    "kayak",
    "mam non",
    "learning hub",
    "ho dieu hoa",
    "r&d hub",
    "co-working space",
    "an ninh",
    "ty le cay xanh",
]

# Priority 11 -- Concept / brand / vision (low risk)
_CONCEPT: list[str] = [
    "tagline",
    "slogan",
    "dinh vi",
    "concept",
    "thong diep du an",
    "y nghia ten",
    "ten neo city",
    "dinh huong phat trien",
    "tam nhin du an",
    "phong cach thiet ke",
    "tham my",
    "thuong hieu",
]

# Priority 12 -- Project overview / factsheet (low risk)
_PROJECT_OVERVIEW: list[str] = [
    "du an la gi",
    "du an gi",
    "neo city la gi",
    "tong quan du an",
    "gioi thieu du an",
    "quy mo du an",
    "chu dau tu",
    "tong so can",
    "bao nhieu toa",
    "tong dien tich",
    "thong tin du an",
    "factsheet",
    "la gi",
    "tong quan",
    "quy mo",
    "bao nhieu ha",
    "mat do xay dung",
    "chu luc",
    "mo hinh phat trien",
    "khu do thi",
    "neo city co gi",
    "bao nhieu can ho",
    "tong cong",
    "smart home",
    "tien do xay dung du kien",
    "tien do xay dung",
    "tu duy",
    "tu duy phat trien",
    "duoc phat trien theo",
    "xay dung theo tu duy",
    "phat trien theo tu duy",
    "live work play",
    "recharge connect",
]

_LEGAL_OVERRIDE_TERMS: list[str] = [
    "phap ly",
    "phap li",
    "mo ban",
    "du dieu kien",
    "dat coc",
    "nhan coc",
    "huy dong von",
    "giay phep",
    "hdmb",
    "hop dong mua ban",
    "duoc ban chua",
    "duoc thu tien chua",
    "duoc thu tien",
    "chinh thuc chua",
    "chu dau tu neo city la ai",
    "chu dau tu neolab co uy tin khong",
    "chu dau tu neolab la ai",
]

_PRICING_VALUE_TERMS: list[str] = [
    "gia",
    "bang gia",
    "gia ban",
    "bao nhieu tien",
    "tong gia tri",
    "don gia",
    "trieu/m2",
    "trieu dong/m2",
    "trieu dong",
    "trieu",
    "ty",
    "khoang bao nhieu",
    "price",
    "cost",
    "value",
]

_SALES_POLICY_OVERRIDE_TERMS: list[str] = [
    "chinh sach",
    "thanh toan",
    "chiet khau",
    "booking",
    "uu dai",
    "ho tro vay",
    "an han",
    "lai suat",
    "phi quan ly",
    "voucher",
    "combo",
    "tien do thanh toan",
]

_LOCATION_DIRECT_TERMS: list[str] = [
    "vi tri",
    "o dau",
    "ket noi",
    "lien ket vung",
    "san bay",
    "noi bai",
    "dong anh",
    "soc son",
    "vo van kiet",
    "vanh dai",
    "cau thang long",
    "cau nhat tan",
    "cau hong ha",
    "giao thong",
]

_MARKET_INVESTMENT_TERMS: list[str] = [
    "thi truong",
    "dau tu",
    "tiem nang",
    "tang truong",
    "tang gia",
    "sinh loi",
    "loi nhuan",
    "profit",
    "return",
    "investment",
    "growth",
]

_AMENITIES_POLICY_TERMS: list[str] = [
    "voucher",
    "combo",
    "uu dai",
    "chinh sach",
    "booking",
]


# ---------------------------------------------------------------------------
# Helper: match any keyword in the normalized query string
# ---------------------------------------------------------------------------


def _any_match(normalized: str, keywords: list[str]) -> bool:
    return any(kw in normalized for kw in keywords)


def _contains_term(normalized: str, term: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(term).replace(r"\ ", r"\s+") + r"(?!\w)"
    return re.search(pattern, normalized) is not None


def _match_terms(normalized: str, keywords: list[str]) -> bool:
    return any(_contains_term(normalized, keyword) for keyword in keywords)


# ---------------------------------------------------------------------------
# Rich classify() -- main Task-5 function
# ---------------------------------------------------------------------------

_LEGAL_SECTIONS = ["legal"]
_PRICING_SECTIONS = ["pricing", "price_sheet"]
_SALES_POLICY_SECTIONS = ["sales_policy", "price_sheet"]
_MARKET_SECTIONS = ["market"]
_LOCATION_SECTIONS = ["location_connectivity", "market"]
_SALES_STRATEGY_SECTIONS = ["sales_strategy", "personas"]
_PERSONA_SECTIONS = ["personas"]
_PRODUCT_SECTIONS = ["factsheet", "pricing"]
_AMENITIES_SECTIONS = ["factsheet"]
_CONCEPT_SECTIONS = ["concept_positioning"]
_PROJECT_OVERVIEW_SECTIONS = ["factsheet", "concept_positioning"]


def classify(query: str) -> ClassificationResult:
    """Classify a Vietnamese (or English) user query into a rich intent result.

    Parameters
    ----------
    query:
        The raw user question.

    Returns
    -------
    ClassificationResult
        Contains intent, target_sections, risk_level, must_use_legal_only.

    Notes
    -----
    Rules are evaluated in strict priority order (highest priority first).
    The first matching rule wins.
    """
    if not query or not query.strip():
        return ClassificationResult(
            intent="unknown",
            target_sections=[],
            risk_level="low",
            must_use_legal_only=False,
        )

    norm = _norm_legacy(query)
    has_sales_policy = _match_terms(norm, _SALES_POLICY_OVERRIDE_TERMS) or _match_terms(
        norm, _SALES_POLICY_NEW
    )
    has_area_terms = _match_terms(norm, ["dien tich", "m2", "met vuong"])
    has_product = _match_terms(norm, _PRODUCT)
    has_amenities = _match_terms(norm, _AMENITIES)
    has_market = _match_terms(norm, _MARKET_NEW)
    has_market_investment = _match_terms(norm, _MARKET_INVESTMENT_TERMS)
    has_location = _match_terms(norm, _LOCATION_DIRECT_TERMS) or _match_terms(norm, _LOCATION)
    has_explicit_price_value = _match_terms(
        norm,
        [
            "bang gia",
            "gia ban",
            "tong gia tri",
            "don gia",
            "muc gia",
            "trieu/m2",
            "trieu dong/m2",
            "price",
            "cost",
            "value",
        ],
    ) or (
        _contains_term(norm, "gia")
        and _match_terms(
            norm,
            ["bao nhieu", "du kien", "tham khao", "khoi diem", "cao hon", "thap hon", "m2"],
        )
    )
    has_soft_price_value = _match_terms(
        norm,
        ["bao nhieu tien", "khoang bao nhieu"],
    )
    has_pricing = has_explicit_price_value or (
        has_soft_price_value and not has_area_terms and not has_sales_policy
    )
    has_legal = _match_terms(norm, _LEGAL_OVERRIDE_TERMS) or _match_terms(norm, _LEGAL_DIRECT)

    # ------------------------------------------------------------------
    # Priority 1 -- Legal (critical, must_use_legal_only=True)
    # ------------------------------------------------------------------
    if has_legal:
        return ClassificationResult(
            intent="legal",
            target_sections=_LEGAL_SECTIONS,
            risk_level="critical",
            must_use_legal_only=True,
        )

    # ------------------------------------------------------------------
    # Priority 2 -- Guaranteed profit (critical, must_use_legal_only=False)
    # ------------------------------------------------------------------
    if _any_match(norm, _GUARANTEED_PROFIT) or _has_profit_guarantee_compound(norm):
        return ClassificationResult(
            intent="legal",
            target_sections=["legal", "market"],
            risk_level="critical",
            must_use_legal_only=False,
        )

    # ------------------------------------------------------------------
    # Priority 3 -- Sales policy (high)
    # ------------------------------------------------------------------
    if has_amenities and _any_match(norm, _AMENITIES_POLICY_TERMS):
        return ClassificationResult(
            intent="sales_policy",
            target_sections=_SALES_POLICY_SECTIONS,
            risk_level="high",
            must_use_legal_only=False,
        )

    if has_sales_policy:
        return ClassificationResult(
            intent="sales_policy",
            target_sections=_SALES_POLICY_SECTIONS,
            risk_level="high",
            must_use_legal_only=False,
        )

    # ------------------------------------------------------------------
    # Priority 4 -- Pricing (high)
    # ------------------------------------------------------------------
    if (
        has_market_investment
        and _contains_term(norm, "gia")
        and _contains_term(norm, "thi truong")
        and _match_terms(norm, ["cao hon", "thap hon", "xung dang"])
    ):
        return ClassificationResult(
            intent="market",
            target_sections=_MARKET_SECTIONS,
            risk_level="medium",
            must_use_legal_only=False,
        )

    if has_product and has_pricing:
        return ClassificationResult(
            intent="pricing",
            target_sections=_PRICING_SECTIONS,
            risk_level="high",
            must_use_legal_only=False,
        )

    if has_pricing:
        return ClassificationResult(
            intent="pricing",
            target_sections=_PRICING_SECTIONS,
            risk_level="high",
            must_use_legal_only=False,
        )

    # ------------------------------------------------------------------
    # Priority 5 -- Market / investment (medium)
    # ------------------------------------------------------------------
    if has_market and not has_location:
        return ClassificationResult(
            intent="market",
            target_sections=_MARKET_SECTIONS,
            risk_level="medium",
            must_use_legal_only=False,
        )

    # ------------------------------------------------------------------
    # Priority 6 -- Location / connectivity (medium)
    # ------------------------------------------------------------------
    if has_location and not has_market_investment:
        return ClassificationResult(
            intent="location",
            target_sections=_LOCATION_SECTIONS,
            risk_level="medium",
            must_use_legal_only=False,
        )

    if has_market:
        return ClassificationResult(
            intent="market",
            target_sections=_MARKET_SECTIONS,
            risk_level="medium",
            must_use_legal_only=False,
        )

    # ------------------------------------------------------------------
    # Priority 7 -- Sales strategy (medium)
    # ------------------------------------------------------------------
    if _any_match(norm, _SALES_STRATEGY):
        return ClassificationResult(
            intent="sales_strategy",
            target_sections=_SALES_STRATEGY_SECTIONS,
            risk_level="medium",
            must_use_legal_only=False,
        )

    # ------------------------------------------------------------------
    # Priority 8 -- Persona / target customer (low)
    # ------------------------------------------------------------------
    if _any_match(norm, _PERSONA):
        return ClassificationResult(
            intent="persona",
            target_sections=_PERSONA_SECTIONS,
            risk_level="low",
            must_use_legal_only=False,
        )

    # ------------------------------------------------------------------
    # Priority 9 -- Amenities / internal facilities (low)
    # "tien ich" is a specific signal that must not be overridden by
    # broad product terms like "co nhung" that appear in amenity queries.
    # ------------------------------------------------------------------
    if has_amenities:
        return ClassificationResult(
            intent="amenities",
            target_sections=_AMENITIES_SECTIONS,
            risk_level="low",
            must_use_legal_only=False,
        )

    # ------------------------------------------------------------------
    # Priority 10 -- Product / unit type (medium)
    # ------------------------------------------------------------------
    if has_product:
        return ClassificationResult(
            intent="product",
            target_sections=_PRODUCT_SECTIONS,
            risk_level="medium",
            must_use_legal_only=False,
        )

    # ------------------------------------------------------------------
    # Priority 11 -- Concept / brand / vision (low)
    # ------------------------------------------------------------------
    if _any_match(norm, _CONCEPT):
        return ClassificationResult(
            intent="concept",
            target_sections=_CONCEPT_SECTIONS,
            risk_level="low",
            must_use_legal_only=False,
        )

    # ------------------------------------------------------------------
    # Priority 12 -- Project overview / factsheet (low)
    # ------------------------------------------------------------------
    if _any_match(norm, _PROJECT_OVERVIEW):
        return ClassificationResult(
            intent="project_overview",
            target_sections=_PROJECT_OVERVIEW_SECTIONS,
            risk_level="low",
            must_use_legal_only=False,
        )

    # ------------------------------------------------------------------
    # Fallback -- unknown
    # ------------------------------------------------------------------
    return ClassificationResult(
        intent="unknown",
        target_sections=[],
        risk_level="low",
        must_use_legal_only=False,
    )


# ---------------------------------------------------------------------------
# CLI demo: python -m app.intent_classifier "query here"
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    _query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "NEO CITY là dự án gì?"
    _result = classify(_query)
    print(json.dumps(_result.to_dict(), ensure_ascii=False, indent=2))
