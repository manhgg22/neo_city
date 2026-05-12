from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = ROOT_DIR / "data" / "processed" / "neo_city_sections.json"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "neo_city_chunks.jsonl"
DEFAULT_SCHEMA_PATH = ROOT_DIR / "data" / "schema" / "neo_city_schema.json"

PROJECT_NAME = "NEO CITY"
SOURCE_DOC_NAME = "All database - NEO CITY.docx"
VERSION = "2026-05"

STATUS_BY_SECTION = {
    "factsheet": "estimated",
    "location_connectivity": "market_reference",
    "personas": "strategy_data",
    "concept_positioning": "marketing_core",
    "sales_strategy": "strategy_data",
    "sales_policy": "hypothetical_policy",
    "legal": "legal_sensitive",
    "pricing": "estimated",
    "market": "market_reference",
    "price_sheet": "draft",
}

BASE_LEGAL_SENSITIVITY_BY_SECTION = {
    "legal": "critical",
    "pricing": "high",
    "sales_policy": "high",
    "price_sheet": "high",
    "market": "medium",
    "location_connectivity": "medium",
    "factsheet": "medium",
    "personas": "low",
    "concept_positioning": "low",
    "sales_strategy": "medium",
}

LEGAL_SENSITIVITY_PRIORITY = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}

LEGAL_SENSITIVITY_TERMS = (
    "mở bán",
    "huy động vốn",
    "đặt cọc",
    "pháp lý",
    "giấy phép",
    "đủ điều kiện",
    "cam kết",
    "lợi nhuận",
)

ROMAN_PATTERN = re.compile(
    r"^(?P<roman>I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII)\.\s+"
)
SUBNUMBER_PATTERN = re.compile(r"^(?P<number>\d+\.\d+)\.\s+")
SIMPLE_NUMBER_PATTERN = re.compile(r"^(?P<number>\d+)\.\s+")
ALPHA_PATTERN = re.compile(r"^(?P<alpha>[A-Z])\.\s+")
PERSONA_PATTERN = re.compile(r"^Persona\s+\d+\b", re.IGNORECASE)
POLICY_PATTERN = re.compile(r"^Chính sách\s+\d+\s*[–-]\s*", re.IGNORECASE)
COMBO_PATTERN = re.compile(r"^Combo\s+[A-Z]\b", re.IGNORECASE)

PERSONA_SPECIAL_TITLES = {
    "tóm tắt 5 personas theo logic chiến lược": "persona_strategy_summary",
    "persona trung tâm nhất của neo city là ai?": "core_buyer_persona",
}

CONCEPT_BRAND_TOPICS = {
    "brand core": "brand_core",
    "brand essence": "brand_essence",
    "brand purpose": "brand_purpose",
    "brand vision": "brand_vision",
    "brand values": "brand_values",
    "brand personality": "brand_personality",
    "tone of voice": "tone_of_voice",
    "strategic positioning": "brand_strategic_positioning",
    "brand promise": "brand_promise",
    "reasons to believe": "reasons_to_believe",
    "message house": "message_house",
    "content pillars": "content_pillars",
    "sales narrative": "sales_narrative",
    "visual direction": "visual_direction",
    "tagline options": "brand_platform_tagline",
    "brand manifesto": "brand_platform_manifesto",
    "kết luận chiến lược": "strategic_conclusion",
}

TOPIC_OVERRIDES = {
    ("factsheet", "1. tổng quan dự án"): "project_overview",
    ("factsheet", "2. tư duy phát triển dự án"): "development_vision",
    ("factsheet", "3. vị trí & vai trò chiến lược"): "location_strategy",
    ("factsheet", "4. quy mô phát triển"): "development_scale",
    ("factsheet", "5. cơ cấu khách hàng mục tiêu"): "target_customer_segments",
    ("factsheet", "6. cơ cấu sản phẩm chi tiết"): "apartment_products",
    ("factsheet", "7. hệ tiện ích cốt lõi"): "amenities",
    ("factsheet", "8. điểm khác biệt chiến lược của neo city"): "strategic_differentiation",
    ("factsheet", "9. định vị truyền thông gợi ý"): "brand_positioning",
    ("factsheet", "10. giá trị dự án mang lại cho thị trường"): "market_value",
    ("factsheet", "11. dữ liệu gợi ý để feed cho hệ ai demo"): "ai_input_assets",
    ("factsheet", "12. mô tả ngắn để dùng trong brochure / deck"): "brochure_summary",
    ("location_connectivity", "1. hệ thống y tế kế cận dự án"): "medical_connectivity",
    ("location_connectivity", "2. kết nối giao thông chiến lược"): "transport_connectivity",
    ("location_connectivity", "3. kết nối hạ tầng tương lai"): "future_infrastructure_connectivity",
    ("location_connectivity", "4. kết nối tới các trung tâm việc làm và công nghiệp"): "employment_connectivity",
    ("location_connectivity", "5. liên kết vùng theo nhu cầu sống hằng ngày"): "daily_life_connectivity",
    ("location_connectivity", "6. đoạn viết hoàn chỉnh để đưa vào hồ sơ khách hàng"): "customer_profile_narrative",
    ("location_connectivity", "7. phiên bản ngắn để đưa vào factsheet"): "factsheet_connectivity_summary",
    ("personas", "persona 1"): "buyer_persona_young_professional",
    ("personas", "persona 2"): "buyer_persona_family",
    ("personas", "persona 3"): "buyer_persona_tech_creative",
    ("personas", "persona 4"): "buyer_persona_investor",
    ("personas", "persona 5"): "buyer_persona_upgrader",
    ("personas", "tóm tắt 5 personas theo logic chiến lược"): "persona_strategy_summary",
    ("personas", "persona trung tâm nhất của neo city là ai?"): "core_buyer_persona",
    ("sales_strategy", "1. persona"): "buyer_persona_young_professional",
    ("sales_strategy", "2. persona"): "buyer_persona_family",
    ("sales_strategy", "3. persona"): "buyer_persona_tech_creative",
    ("sales_strategy", "4. persona"): "buyer_persona_investor",
    ("sales_strategy", "5. persona"): "buyer_persona_upgrader",
    ("sales_strategy", "vi. chiến lược bán hàng tổng thể theo phân lớp persona"): "persona_sales_framework",
    ("sales_strategy", "vii. nguyên tắc chung cho đội sales"): "sales_principles",
    ("sales_strategy", "viii. câu chốt cho sales toàn dự án"): "sales_closing_message",
    ("sales_policy", "i. tư duy chính sách bán hàng tổng thể"): "sales_policy_framework",
    ("sales_policy", "1. chính sách “khởi đầu nhịp sống mới”"): "umbrella_sales_policy",
    ("sales_policy", "2. chính sách thanh toán chuẩn"): "payment_policy",
    ("sales_policy", "3. chính sách “ưu tiên người đi trước”"): "early_buyer_policy",
    ("sales_policy", "chính sách 1 – “an cư nhẹ bước”"): "apartment_policy_light_entry",
    ("sales_policy", "chính sách 2 – “ở trước, trả sau nhẹ hơn”"): "apartment_policy_pay_later",
    ("sales_policy", "chính sách 1 – “nhà phố mở nhịp kinh doanh”"): "lowrise_policy_business_rhythm",
    ("sales_policy", "chính sách 2 – “cam kết hỗ trợ khai trương”"): "lowrise_policy_launch_support",
    ("sales_policy", "1. persona: người trẻ mua căn đầu tiên"): "buyer_persona_young_professional_policy",
    ("sales_policy", "2. persona: gia đình trẻ"): "buyer_persona_family_policy",
    ("sales_policy", "3. persona: người làm công nghệ / sáng tạo / hybrid work"): "buyer_persona_tech_creative_policy",
    ("sales_policy", "4. persona: nhà đầu tư trung lưu"): "buyer_persona_investor_policy",
    ("sales_policy", "5. persona: người mua nâng cấp chất lượng sống"): "buyer_persona_upgrader_policy",
    ("sales_policy", "1. “mở bán hồ trung tâm”"): "campaign_central_lake_launch",
    ("sales_policy", "2. “mùa lễ hội neo square”"): "campaign_neo_square_festival",
    ("sales_policy", "3. “neo weekend discovery”"): "campaign_weekend_discovery",
    ("sales_policy", "vi. combo chính sách theo sản phẩm + persona"): "combo_policy_overview",
    ("sales_policy", "combo a"): "combo_young_professional",
    ("sales_policy", "combo b"): "combo_family",
    ("sales_policy", "combo c"): "combo_investor",
    ("sales_policy", "combo d"): "combo_upgrader",
    ("sales_policy", "1. neo start"): "top_policy_neo_start",
    ("sales_policy", "2. neo family move"): "top_policy_family_move",
    ("sales_policy", "3. ở trước, trả sau nhẹ hơn"): "top_policy_pay_later",
    ("sales_policy", "viii. câu chốt chính sách"): "sales_policy_closing_message",
    ("legal", "1. thông tin pháp nhân phát triển dự án"): "legal_entity_info",
    ("legal", "2. thông tin pháp lý tổng quan dự án"): "legal_overview",
    ("legal", "3. danh mục hồ sơ pháp lý cần hoàn thiện"): "required_legal_documents",
    ("legal", "4. tình trạng pháp lý hiện tại"): "legal_status_and_warnings",
    ("pricing", "1. nguyên tắc xây dựng giá bán"): "pricing_principles",
    ("pricing", "2.1. sản phẩm căn hộ cao tầng"): "apartment_pricing",
    ("pricing", "2.2. sản phẩm thấp tầng"): "lowrise_pricing",
    ("pricing", "3.1. chính sách đặt chỗ / đăng ký nguyện vọng"): "reservation_policy",
    ("pricing", "3.2. chính sách thanh toán tiêu chuẩn"): "payment_policy",
    ("pricing", "3.3. chính sách vay ngân hàng"): "bank_loan_policy",
    ("pricing", "3.4. chính sách ưu đãi theo nhóm khách hàng"): "customer_group_incentives",
    ("pricing", "3.5. chính sách chiết khấu"): "discount_policy",
    ("pricing", "4.1. căn hộ studio+ và 1pn+1"): "studio_one_bedroom_policy",
    ("pricing", "4.2. căn hộ 2pn và 2pn+1"): "family_apartment_policy",
    ("pricing", "4.3. căn hộ 3pn"): "three_bedroom_policy",
    ("pricing", "4.4. townhouse / shophouse / courtyard villa"): "lowrise_product_policy",
    ("pricing", "5. ghi chú pháp lý về giá bán và chính sách bán hàng"): "pricing_legal_notes",
    ("market", "1. tóm tắt luận điểm thị trường"): "market_overview",
    ("market", "2. vị thế mới của mê linh trong cấu trúc phát triển hà nội"): "melinh_positioning",
    ("market", "3. động lực hạ tầng: yếu tố củng cố niềm tin dài hạn"): "infrastructure_drivers",
    ("market", "4. áp lực giá nhà nội đô tạo nhu cầu dịch chuyển thật"): "affordability_shift",
    ("market", "5. xu hướng dịch chuyển ra vùng đô thị mới đã rõ hơn"): "suburban_migration_trend",
    ("market", "6. khoảng trống thị trường mà neo city có thể nắm giữ"): "market_gap",
    ("price_sheet", "1. thông tin chung trên phiếu tính giá"): "pricing_sheet_overview",
    ("price_sheet", "2.1. đối tượng áp dụng"): "standard_policy_eligibility",
    ("price_sheet", "2.2. lịch thanh toán dự kiến"): "standard_payment_schedule",
    ("price_sheet", "2.3. cơ cấu thanh toán tổng hợp"): "standard_payment_structure",
    ("price_sheet", "3.1. đối tượng áp dụng"): "early_payment_eligibility",
    ("price_sheet", "3.2. các gói thanh toán sớm dự kiến"): "early_payment_packages",
    ("price_sheet", "3.3. cách tính giá trị sau chiết khấu"): "discount_calculation",
    ("price_sheet", "3.4. ưu đãi bổ sung có thể áp dụng"): "additional_incentives",
    ("price_sheet", "4.1. đối tượng áp dụng"): "bank_loan_eligibility",
    ("price_sheet", "4.2. cấu trúc vay dự kiến"): "loan_structure",
    ("price_sheet", "4.3. lịch thanh toán dự kiến với phương án vay"): "loan_payment_schedule",
    ("price_sheet", "4.4. các khoản khách hàng cần chuẩn bị"): "loan_preparation_costs",
    ("price_sheet", "5. chính sách ưu đãi bổ sung"): "supplemental_incentives",
    ("price_sheet", "6.1. căn hộ studio+ và 1pn+1"): "studio_one_bedroom_policy",
    ("price_sheet", "6.2. căn hộ 2pn và 2pn+1"): "two_bedroom_policy",
    ("price_sheet", "6.3. căn hộ 3pn"): "three_bedroom_policy",
    ("price_sheet", "6.4. townhouse"): "townhouse_policy",
    ("price_sheet", "6.5. shophouse"): "shophouse_policy",
    ("price_sheet", "6.6. courtyard villa"): "courtyard_villa_policy",
    ("price_sheet", "7. bảng so sánh 3 chính sách bán hàng"): "policy_comparison",
    ("concept_positioning", "1. giải nghĩa tên dự án: neo city là gì?"): "name_meaning",
    ("concept_positioning", "2. strategic concept tổng thể"): "strategic_concept",
    ("concept_positioning", "3. tư tưởng chiến lược cốt lõi của concept"): "core_concept",
    ("concept_positioning", "4. bài toán thị trường mà concept này giải quyết"): "market_problem",
    ("concept_positioning", "5. big strategic promise của dự án"): "big_promise",
    ("concept_positioning", "6. core insight của khách hàng mục tiêu"): "customer_insight",
    ("concept_positioning", "7. định vị chiến lược của neo city"): "strategic_positioning",
    ("concept_positioning", "8. 5 trụ giá trị của concept"): "concept_value_pillars",
    ("concept_positioning", "9. narrative chuẩn để kể dự án"): "narrative_framework",
    ("concept_positioning", "10. tuyên ngôn thương hiệu dự án"): "brand_manifesto",
    ("concept_positioning", "11. key message system cho kinh doanh & marketing"): "message_system",
    ("concept_positioning", "12. hướng ứng dụng cho bộ phận kinh doanh"): "sales_application",
    ("concept_positioning", "13. hướng ứng dụng cho marketing"): "marketing_application",
    ("concept_positioning", "2. hình ảnh truyền thông"): "visual_communication",
    ("concept_positioning", "3. trải nghiệm chiến dịch"): "campaign_experience",
    ("concept_positioning", "14. big idea gợi ý từ concept mũ"): "big_idea_options",
    ("concept_positioning", "14. visual direction gợi ý"): "brand_creative_output",
    ("concept_positioning", "15. tagline gợi ý"): "tagline_options",
    ("concept_positioning", "16. kết luận concept"): "concept_conclusion",
    ("concept_positioning", "i. định vị rõ ràng"): "clear_positioning",
    ("concept_positioning", "ii. thông điệp chiến lược cốt lõi"): "core_strategic_message",
    ("concept_positioning", "iii. tagline thật hay"): "tagline_recommendation",
    ("concept_positioning", "iv. usp mạnh nhất của dự án"): "primary_usp",
    ("concept_positioning", "v. bộ ksp (key selling points)"): "key_selling_points",
    ("concept_positioning", "vi. message house cho sales kit"): "sales_message_house",
    ("concept_positioning", "vii. sales conversion messages"): "sales_conversion_messages",
    ("concept_positioning", "viii. objection handling angles"): "objection_handling",
    ("concept_positioning", "ix. các câu headline dùng trong sales kit"): "sales_headlines",
    ("concept_positioning", "x. kết luận chốt"): "positioning_conclusion",
}


@dataclass(frozen=True)
class SourceSection:
    section_key: str
    source_title: str
    raw_text: str


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    project: str
    section: str
    topic: str
    source_doc: str
    source_title: str
    status: str
    legal_sensitivity: str
    version: str
    text: str
    chunk_index: int


@dataclass(frozen=True)
class SchemaRules:
    required_fields: tuple[str, ...]
    allowed_sections: frozenset[str]
    allowed_statuses: frozenset[str]
    allowed_legal_sensitivity: frozenset[str]
    project_name: str
    version_pattern: re.Pattern[str]


@dataclass(frozen=True)
class HeadingInfo:
    text: str
    level: int
    kind: str
    start_chunk: bool
    number: int | None = None


@dataclass
class ChunkUnit:
    section_key: str
    source_title: str
    primary_heading: str | None
    blocks: list[str]


def normalize_text(text: str) -> str:
    stripped = unicodedata.normalize("NFKD", text).strip()
    stripped = stripped.replace("đ", "d").replace("Đ", "D")
    without_marks = "".join(
        character for character in stripped if not unicodedata.combining(character)
    )
    return " ".join(without_marks.casefold().split())


PERSONA_SPECIAL_TITLES = {
    normalize_text(key): value for key, value in PERSONA_SPECIAL_TITLES.items()
}

CONCEPT_BRAND_TOPICS = {
    normalize_text(key): value for key, value in CONCEPT_BRAND_TOPICS.items()
}

TOPIC_OVERRIDES = {
    (section_key, normalize_text(heading_prefix)): topic
    for (section_key, heading_prefix), topic in TOPIC_OVERRIDES.items()
}

# Brand platform headings (normalized, stripped of number prefix) that should be
# appended as context blocks into the preceding chunk rather than starting their own.
CONCEPT_BRAND_MERGE_INTO_PREV = frozenset({
    "brand essence",
    "brand purpose",
    "brand vision",
    "brand promise",
    "tagline options",
    "brand manifesto",
    "ket luan chien luoc",
})


def slugify(text: str) -> str:
    normalized = normalize_text(text)
    slug = re.sub(r"[^a-z0-9]+", "_", normalized)
    slug = slug.strip("_")
    return slug or "general"


def load_sections(input_path: Path) -> list[SourceSection]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    return [SourceSection(**item) for item in payload]


def load_schema_rules(schema_path: Path) -> SchemaRules:
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    properties = payload["properties"]
    project_name = properties["project"]["const"]
    version_pattern = re.compile(properties["version"]["pattern"])
    return SchemaRules(
        required_fields=tuple(payload["required"]),
        allowed_sections=frozenset(properties["section"]["enum"]),
        allowed_statuses=frozenset(properties["status"]["enum"]),
        allowed_legal_sensitivity=frozenset(properties["legal_sensitivity"]["enum"]),
        project_name=project_name,
        version_pattern=version_pattern,
    )


def split_blocks(raw_text: str) -> list[str]:
    return [block.strip() for block in raw_text.split("\n\n") if block.strip()]


def is_table_block(block: str) -> bool:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    return bool(lines) and all(line.startswith("|") for line in lines)


def dedupe_blocks(blocks: list[str]) -> list[str]:
    deduped: list[str] = []
    for block in blocks:
        if not block:
            continue
        if deduped and deduped[-1] == block:
            continue
        deduped.append(block)
    return deduped


def parse_roman_heading(block: str) -> HeadingInfo | None:
    match = ROMAN_PATTERN.match(block)
    if match is None:
        return None
    return HeadingInfo(
        text=block,
        level=1,
        kind="roman",
        start_chunk=False,
    )


def parse_subnumber_heading(block: str) -> HeadingInfo | None:
    match = SUBNUMBER_PATTERN.match(block)
    if match is None:
        return None
    return HeadingInfo(
        text=block,
        level=3,
        kind="subnumber",
        start_chunk=False,
    )


def parse_simple_number_heading(block: str) -> HeadingInfo | None:
    if SUBNUMBER_PATTERN.match(block):
        return None
    match = SIMPLE_NUMBER_PATTERN.match(block)
    if match is None:
        return None
    return HeadingInfo(
        text=block,
        level=2,
        kind="simple",
        start_chunk=False,
        number=int(match.group("number")),
    )


def parse_alpha_heading(block: str) -> HeadingInfo | None:
    match = ALPHA_PATTERN.match(block)
    if match is None:
        return None
    return HeadingInfo(
        text=block,
        level=3,
        kind="alpha",
        start_chunk=False,
    )


def parse_persona_heading(block: str) -> HeadingInfo | None:
    if PERSONA_PATTERN.match(block) is None:
        return None
    match = re.search(r"Persona\s+(?P<number>\d+)", block, re.IGNORECASE)
    return HeadingInfo(
        text=block,
        level=2,
        kind="persona",
        start_chunk=True,
        number=int(match.group("number")) if match else None,
    )


def parse_policy_heading(block: str) -> HeadingInfo | None:
    if POLICY_PATTERN.match(block) is None:
        return None
    return HeadingInfo(
        text=block,
        level=4,
        kind="policy",
        start_chunk=True,
    )


def parse_combo_heading(block: str) -> HeadingInfo | None:
    if COMBO_PATTERN.match(block) is None:
        return None
    return HeadingInfo(
        text=block,
        level=3,
        kind="combo",
        start_chunk=True,
    )


def current_root_number(active_headings: dict[int, HeadingInfo]) -> int | None:
    heading = active_headings.get(2)
    if heading is None or heading.number is None:
        return None
    return heading.number


def current_nested_number(active_headings: dict[int, HeadingInfo]) -> int | None:
    heading = active_headings.get(4)
    if heading is None or heading.number is None:
        return None
    return heading.number


def concept_brand_heading_topic(block: str) -> str | None:
    normalized = normalize_text(re.sub(r"^\d+\.\s+", "", block))
    for keyword, topic in CONCEPT_BRAND_TOPICS.items():
        if normalized == normalize_text(keyword):
            return topic
    return None


def build_context_heading(
    block: str,
    level: int,
    kind: str,
    number: int | None = None,
) -> HeadingInfo:
    return HeadingInfo(
        text=block,
        level=level,
        kind=kind,
        start_chunk=False,
        number=number,
    )


def detect_heading(
    section: SourceSection,
    block: str,
    active_headings: dict[int, HeadingInfo],
) -> HeadingInfo | None:
    if is_table_block(block):
        return None

    normalized = normalize_text(block)

    if section.section_key == "personas":
        if block in {"Concept", "Định vị"}:
            return None
        if normalized in PERSONA_SPECIAL_TITLES:
            return HeadingInfo(block, 1, "special", True)
        persona_heading = parse_persona_heading(block)
        if persona_heading is not None:
            return persona_heading
        simple_heading = parse_simple_number_heading(block)
        if simple_heading is not None:
            return build_context_heading(
                block,
                level=4,
                kind="simple_nested",
                number=simple_heading.number,
            )
        return None

    if section.section_key == "sales_strategy":
        roman_heading = parse_roman_heading(block)
        if roman_heading is not None:
            return HeadingInfo(block, 1, "roman", True)
        if "persona:" in normalized:
            simple_heading = parse_simple_number_heading(block)
            if simple_heading is not None:
                return HeadingInfo(block, 2, "simple", True, simple_heading.number)
        simple_heading = parse_simple_number_heading(block)
        if simple_heading is not None:
            return build_context_heading(
                block,
                level=4,
                kind="simple_nested",
                number=simple_heading.number,
            )
        return None

    if section.section_key == "sales_policy":
        roman_heading = parse_roman_heading(block)
        if roman_heading is not None:
            roman_value = ROMAN_PATTERN.match(block).group("roman")
            if roman_value in {"I", "VI", "VIII"}:
                return HeadingInfo(block, 1, "roman", True)
            return build_context_heading(block, 1, "roman")

        combo_heading = parse_combo_heading(block)
        if combo_heading is not None:
            return build_context_heading(block, 3, "combo")

        policy_heading = parse_policy_heading(block)
        if policy_heading is not None:
            return policy_heading

        simple_heading = parse_simple_number_heading(block)
        if simple_heading is None:
            alpha_heading = parse_alpha_heading(block)
            if alpha_heading is not None:
                return alpha_heading
            return None

        roman_parent = active_headings.get(1)
        roman_parent_value = None
        if roman_parent is not None:
            match = ROMAN_PATTERN.match(roman_parent.text)
            roman_parent_value = match.group("roman") if match else None

        if roman_parent_value in {"II", "IV", "V"}:
            return HeadingInfo(block, 2, "simple", True, simple_heading.number)

        return build_context_heading(
            block,
            level=4,
            kind="simple_nested",
            number=simple_heading.number,
        )

    if section.section_key == "legal":
        simple_heading = parse_simple_number_heading(block)
        if simple_heading is None:
            subnumber_heading = parse_subnumber_heading(block)
            if subnumber_heading is not None:
                return subnumber_heading
            return None
        if simple_heading.number in {1, 2, 3, 4}:
            return HeadingInfo(block, 2, "simple", True, simple_heading.number)
        return build_context_heading(block, 2, "simple", number=simple_heading.number)

    if section.section_key == "pricing":
        subnumber_heading = parse_subnumber_heading(block)
        if subnumber_heading is not None:
            return HeadingInfo(block, 3, "subnumber", True)
        simple_heading = parse_simple_number_heading(block)
        if simple_heading is None:
            return None
        if simple_heading.number in {1, 5}:
            return HeadingInfo(block, 2, "simple", True, simple_heading.number)
        if simple_heading.number in {2, 3, 4}:
            return build_context_heading(block, 2, "simple", number=simple_heading.number)
        return None

    if section.section_key == "price_sheet":
        subnumber_heading = parse_subnumber_heading(block)
        if subnumber_heading is not None:
            return HeadingInfo(block, 3, "subnumber", True)
        simple_heading = parse_simple_number_heading(block)
        if simple_heading is None:
            return None
        if simple_heading.number in {1, 5, 7}:
            return HeadingInfo(block, 2, "simple", True, simple_heading.number)
        if simple_heading.number in {2, 3, 4, 6}:
            return build_context_heading(block, 2, "simple", number=simple_heading.number)
        return None

    if section.section_key == "concept_positioning" and section.source_title == "Định vị":
        roman_heading = parse_roman_heading(block)
        if roman_heading is not None:
            return HeadingInfo(block, 1, "roman", True)
        simple_heading = parse_simple_number_heading(block)
        if simple_heading is not None:
            return build_context_heading(
                block,
                level=4,
                kind="simple_nested",
                number=simple_heading.number,
            )
        alpha_heading = parse_alpha_heading(block)
        if alpha_heading is not None:
            return alpha_heading
        return None

    subnumber_heading = parse_subnumber_heading(block)
    if subnumber_heading is not None:
        return subnumber_heading

    alpha_heading = parse_alpha_heading(block)
    if alpha_heading is not None:
        return alpha_heading

    simple_heading = parse_simple_number_heading(block)
    if simple_heading is None:
        return None

    if (
        section.section_key == "concept_positioning"
        and section.source_title == "Concept"
        and concept_brand_heading_topic(block) is not None
    ):
        stripped_label = re.sub(r"^\d+\.\s+", "", block).strip()
        if stripped_label != stripped_label.upper():
            return build_context_heading(block, level=4, kind="simple_nested", number=simple_heading.number)
        if normalize_text(stripped_label) in CONCEPT_BRAND_MERGE_INTO_PREV:
            return build_context_heading(block, level=4, kind="simple_nested", number=simple_heading.number)
        return HeadingInfo(block, 2, "simple", True, simple_heading.number)

    nested_number = current_nested_number(active_headings)
    if nested_number is not None and simple_heading.number == nested_number + 1:
        return build_context_heading(
            block,
            level=4,
            kind="simple_nested",
            number=simple_heading.number,
        )

    root_number = current_root_number(active_headings)
    if root_number is None:
        return HeadingInfo(block, 2, "simple", True, simple_heading.number)

    if simple_heading.number > root_number:
        return HeadingInfo(block, 2, "simple", True, simple_heading.number)

    return build_context_heading(
        block,
        level=4,
        kind="simple_nested",
        number=simple_heading.number,
    )


def collect_parent_headings(
    active_headings: dict[int, HeadingInfo],
    level: int,
) -> list[str]:
    return [
        active_headings[key].text
        for key in sorted(active_headings)
        if key < level
    ]


def update_active_headings(
    active_headings: dict[int, HeadingInfo],
    heading: HeadingInfo,
) -> None:
    for level in sorted(list(active_headings), reverse=True):
        if level >= heading.level:
            del active_headings[level]
    active_headings[heading.level] = heading


def finalize_unit(
    units: list[ChunkUnit],
    current_unit: ChunkUnit | None,
) -> None:
    if current_unit is None:
        return
    current_unit.blocks = dedupe_blocks(current_unit.blocks)
    if any(block.strip() for block in current_unit.blocks):
        units.append(current_unit)


def build_chunk_units(section: SourceSection) -> list[ChunkUnit]:
    blocks = split_blocks(section.raw_text)
    active_headings: dict[int, HeadingInfo] = {}
    lead_in_blocks: list[str] = []
    units: list[ChunkUnit] = []
    current_unit: ChunkUnit | None = None

    for block in blocks:
        heading = detect_heading(section, block, active_headings)
        if heading is None:
            if current_unit is not None:
                current_unit.blocks.append(block)
            else:
                lead_in_blocks.append(block)
            continue

        if heading.start_chunk:
            finalize_unit(units, current_unit)
            parent_headings = collect_parent_headings(active_headings, heading.level)
            context_blocks = dedupe_blocks(lead_in_blocks + parent_headings)
            current_unit = ChunkUnit(
                section_key=section.section_key,
                source_title=section.source_title,
                primary_heading=heading.text,
                blocks=context_blocks + [heading.text],
            )
            lead_in_blocks = []
        else:
            if current_unit is not None:
                current_unit.blocks.append(heading.text)
            else:
                lead_in_blocks.append(heading.text)

        update_active_headings(active_headings, heading)

    if current_unit is None and lead_in_blocks:
        current_unit = ChunkUnit(
            section_key=section.section_key,
            source_title=section.source_title,
            primary_heading=None,
            blocks=lead_in_blocks,
        )

    finalize_unit(units, current_unit)
    return units


def infer_topic(unit: ChunkUnit) -> str:
    heading = unit.primary_heading or unit.blocks[0]
    normalized_heading = normalize_text(heading)

    for (section_key, heading_prefix), topic in TOPIC_OVERRIDES.items():
        if unit.section_key == section_key and normalized_heading.startswith(heading_prefix):
            return topic

    if unit.section_key == "personas":
        for keyword, topic in PERSONA_SPECIAL_TITLES.items():
            if normalized_heading == keyword:
                return topic

    if unit.section_key == "concept_positioning" and unit.source_title == "Concept":
        brand_topic = concept_brand_heading_topic(heading)
        if brand_topic is not None:
            return brand_topic

    if unit.primary_heading is None:
        return f"{unit.section_key}_overview"

    trimmed_heading = re.sub(
        r"^(?:\d+(?:\.\d+)?|I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|[A-Z])\.\s+",
        "",
        heading,
    )
    trimmed_heading = re.sub(r"^Chính sách\s+\d+\s*[–-]\s*", "", trimmed_heading)
    trimmed_heading = re.sub(r"^Combo\s+[A-Z]\s+[—-]\s*", "", trimmed_heading)
    topic_slug = slugify(trimmed_heading)
    if topic_slug:
        return topic_slug
    return f"{unit.section_key}_detail"


def infer_legal_sensitivity(section_key: str, text: str) -> str:
    base_sensitivity = BASE_LEGAL_SENSITIVITY_BY_SECTION[section_key]
    normalized_text = normalize_text(text)
    normalized_terms = tuple(normalize_text(term) for term in LEGAL_SENSITIVITY_TERMS)
    if any(term in normalized_text for term in normalized_terms):
        elevated = "critical" if section_key == "legal" else "high"
        if LEGAL_SENSITIVITY_PRIORITY[elevated] > LEGAL_SENSITIVITY_PRIORITY[base_sensitivity]:
            return elevated
    return base_sensitivity


def render_unit_text(unit: ChunkUnit) -> str:
    return "\n\n".join(block for block in unit.blocks if block.strip()).strip()


def build_chunk_records(sections: list[SourceSection]) -> list[ChunkRecord]:
    counters: defaultdict[str, int] = defaultdict(int)
    chunks: list[ChunkRecord] = []
    global_index = 0

    for section in sections:
        units = build_chunk_units(section)
        if not units:
            raise ValueError(
                f"No chunks generated for source section {section.section_key} / {section.source_title}"
            )

        for unit in units:
            text = render_unit_text(unit)
            counters[section.section_key] += 1
            global_index += 1
            chunk_id = f"neo_city_{section.section_key}_{counters[section.section_key]:03d}"
            chunks.append(
                ChunkRecord(
                    id=chunk_id,
                    project=PROJECT_NAME,
                    section=section.section_key,
                    topic=infer_topic(unit),
                    source_doc=SOURCE_DOC_NAME,
                    source_title=section.source_title,
                    status=STATUS_BY_SECTION[section.section_key],
                    legal_sensitivity=infer_legal_sensitivity(section.section_key, text),
                    version=VERSION,
                    text=text,
                    chunk_index=global_index,
                )
            )

    return chunks


def validate_chunk(chunk: ChunkRecord, schema_rules: SchemaRules) -> None:
    payload = asdict(chunk)

    missing_fields = [
        field_name for field_name in schema_rules.required_fields if field_name not in payload
    ]
    if missing_fields:
        raise ValueError(f"Chunk {chunk.id} is missing fields: {missing_fields}")

    if payload["section"] not in schema_rules.allowed_sections:
        raise ValueError(f"Chunk {chunk.id} has invalid section: {payload['section']}")

    if payload["status"] not in schema_rules.allowed_statuses:
        raise ValueError(f"Chunk {chunk.id} has invalid status: {payload['status']}")

    if payload["legal_sensitivity"] not in schema_rules.allowed_legal_sensitivity:
        raise ValueError(
            f"Chunk {chunk.id} has invalid legal_sensitivity: {payload['legal_sensitivity']}"
        )

    if payload["project"] != schema_rules.project_name:
        raise ValueError(f"Chunk {chunk.id} has invalid project: {payload['project']}")

    if not schema_rules.version_pattern.match(payload["version"]):
        raise ValueError(f"Chunk {chunk.id} has invalid version: {payload['version']}")

    for field_name in ("id", "topic", "source_doc", "source_title", "text"):
        if not str(payload[field_name]).strip():
            raise ValueError(f"Chunk {chunk.id} has empty field: {field_name}")


def validate_chunks(
    sections: list[SourceSection],
    chunks: list[ChunkRecord],
    schema_rules: SchemaRules,
) -> None:
    for chunk in chunks:
        validate_chunk(chunk, schema_rules)

    source_pairs = {(section.section_key, section.source_title) for section in sections}
    generated_pairs = {(chunk.section, chunk.source_title) for chunk in chunks}
    missing_pairs = source_pairs - generated_pairs
    if missing_pairs:
        raise ValueError(f"Missing chunks for source sections: {sorted(missing_pairs)}")


def write_chunks_jsonl(chunks: list[ChunkRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        for chunk in chunks:
            file_handle.write(json.dumps(asdict(chunk), ensure_ascii=False))
            file_handle.write("\n")


def create_chunks_jsonl(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
) -> list[ChunkRecord]:
    sections = load_sections(input_path)
    schema_rules = load_schema_rules(schema_path)
    chunks = build_chunk_records(sections)
    validate_chunks(sections, chunks, schema_rules)
    write_chunks_jsonl(chunks, output_path)
    return chunks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert parsed NEO CITY sections into validated JSONL chunks.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the parsed sections JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the output chunks JSONL file.",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help="Path to the chunk schema JSON file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    chunks = create_chunks_jsonl(args.input, args.output, args.schema)
    counts = Counter(chunk.section for chunk in chunks)
    print(f"Created {len(chunks)} chunks at {args.output}")
    for section, count in sorted(counts.items()):
        print(f"- {section}: {count}")


if __name__ == "__main__":
    main()
