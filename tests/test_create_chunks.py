from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT_DIR / "scripts" / "02_create_chunks.py"
INPUT_PATH = ROOT_DIR / "data" / "processed" / "neo_city_sections.json"
SCHEMA_PATH = ROOT_DIR / "data" / "schema" / "neo_city_schema.json"


def load_create_chunks_module():
    spec = importlib.util.spec_from_file_location("create_chunks_script", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load chunker module from {SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


create_chunks = load_create_chunks_module()


@lru_cache(maxsize=1)
def build_actual_chunks() -> tuple[list[object], object]:
    sections = create_chunks.load_sections(INPUT_PATH)
    schema_rules = create_chunks.load_schema_rules(SCHEMA_PATH)
    chunks = create_chunks.build_chunk_records(sections)
    create_chunks.validate_chunks(sections, chunks, schema_rules)
    return chunks, schema_rules


def chunk_payloads() -> list[dict]:
    chunks, _ = build_actual_chunks()
    return [asdict(chunk) for chunk in chunks]


def test_output_file_creation(tmp_path: Path) -> None:
    output_path = tmp_path / "neo_city_chunks.jsonl"
    chunks = create_chunks.create_chunks_jsonl(INPUT_PATH, output_path, SCHEMA_PATH)

    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(chunks)
    assert all(json.loads(line) for line in lines)


def test_every_chunk_has_required_fields() -> None:
    payloads = chunk_payloads()
    _, schema_rules = build_actual_chunks()
    required_fields = set(schema_rules.required_fields)

    for payload in payloads:
        assert required_fields.issubset(payload)


def test_every_chunk_section_is_allowed() -> None:
    payloads = chunk_payloads()
    _, schema_rules = build_actual_chunks()

    for payload in payloads:
        assert payload["section"] in schema_rules.allowed_sections


def test_legal_chunks_have_legal_sensitive_status() -> None:
    payloads = chunk_payloads()
    legal_chunks = [payload for payload in payloads if payload["section"] == "legal"]

    assert legal_chunks
    assert all(payload["status"] == "legal_sensitive" for payload in legal_chunks)


def test_legal_chunks_have_critical_legal_sensitivity() -> None:
    payloads = chunk_payloads()
    legal_chunks = [payload for payload in payloads if payload["section"] == "legal"]

    assert legal_chunks
    assert all(payload["legal_sensitivity"] == "critical" for payload in legal_chunks)


def test_pricing_chunks_have_high_legal_sensitivity() -> None:
    payloads = chunk_payloads()
    pricing_chunks = [payload for payload in payloads if payload["section"] == "pricing"]

    assert pricing_chunks
    assert all(payload["legal_sensitivity"] == "high" for payload in pricing_chunks)


def test_sales_policy_chunks_have_high_legal_sensitivity() -> None:
    payloads = chunk_payloads()
    policy_chunks = [payload for payload in payloads if payload["section"] == "sales_policy"]

    assert policy_chunks
    assert all(payload["legal_sensitivity"] == "high" for payload in policy_chunks)


def test_concept_positioning_chunks_preserve_source_title() -> None:
    payloads = chunk_payloads()
    concept_chunks = [
        payload for payload in payloads if payload["section"] == "concept_positioning"
    ]
    source_titles = {payload["source_title"] for payload in concept_chunks}

    assert concept_chunks
    assert source_titles == {"Concept", "Định vị"}


def test_no_chunk_has_empty_text() -> None:
    payloads = chunk_payloads()
    assert all(payload["text"].strip() for payload in payloads)


def test_expected_sections_are_present_in_chunks() -> None:
    payloads = chunk_payloads()
    sections = {payload["section"] for payload in payloads}

    assert sections == {
        "factsheet",
        "location_connectivity",
        "personas",
        "concept_positioning",
        "sales_strategy",
        "sales_policy",
        "legal",
        "pricing",
        "market",
        "price_sheet",
    }


def test_no_chunk_is_heading_only() -> None:
    payloads = chunk_payloads()
    short_chunks = [p for p in payloads if len(p["text"]) < 50]
    assert not short_chunks, (
        f"Found heading-only chunks (< 50 chars): "
        f"{[(p['id'], len(p['text'])) for p in short_chunks]}"
    )


def test_location_connectivity_chunks_use_market_reference() -> None:
    payloads = chunk_payloads()
    loc_chunks = [p for p in payloads if p["section"] == "location_connectivity"]

    assert loc_chunks
    assert all(p["status"] == "market_reference" for p in loc_chunks), (
        f"Expected all location_connectivity chunks to have status='market_reference', "
        f"got: {[p['status'] for p in loc_chunks]}"
    )


def test_no_vietnamese_topic_slugs() -> None:
    payloads = chunk_payloads()
    bad_slugs = {"hinh_anh_truyen_thong", "trai_nghiem_chien_dich"}
    offenders = [p for p in payloads if p["topic"] in bad_slugs]
    assert not offenders, (
        f"Found chunks with Vietnamese topic slugs: "
        f"{[(p['id'], p['topic']) for p in offenders]}"
    )


def test_chunk_index_is_present_sequential_and_unique() -> None:
    payloads = chunk_payloads()

    assert all("chunk_index" in p for p in payloads), "chunk_index missing from some chunks"
    assert all(isinstance(p["chunk_index"], int) for p in payloads), "chunk_index must be int"

    indices = [p["chunk_index"] for p in payloads]
    assert indices == list(range(1, len(payloads) + 1)), (
        "chunk_index must be sequential starting from 1"
    )


def test_marketing_application_chunk_has_content() -> None:
    payloads = chunk_payloads()
    marketing_chunks = [
        p for p in payloads
        if p["section"] == "concept_positioning" and p["topic"] == "marketing_application"
    ]
    assert marketing_chunks, "No marketing_application chunk found"
    assert all(len(p["text"]) > 100 for p in marketing_chunks), (
        "marketing_application chunk should have real content, not just a heading"
    )


# ---------------------------------------------------------------------------
# Task 2: concept_positioning deduplication and brand platform consolidation
# ---------------------------------------------------------------------------


def test_concept_positioning_has_no_duplicate_topics() -> None:
    payloads = chunk_payloads()
    concept_topics = [p["topic"] for p in payloads if p["section"] == "concept_positioning"]
    duplicates = [t for t in set(concept_topics) if concept_topics.count(t) > 1]
    assert not duplicates, (
        f"concept_positioning has duplicate topic slugs: {duplicates}"
    )


def test_concept_positioning_chunk_count_is_reduced() -> None:
    payloads = chunk_payloads()
    concept_chunks = [p for p in payloads if p["section"] == "concept_positioning"]
    assert len(concept_chunks) <= 36, (
        f"Expected at most 36 concept_positioning chunks after Task 2 merges, "
        f"got {len(concept_chunks)}"
    )


def test_brand_platform_topics_are_disambiguated() -> None:
    payloads = chunk_payloads()
    concept_topics = {p["topic"] for p in payloads if p["section"] == "concept_positioning"}

    # brand_strategic_positioning is the standalone brand-platform chunk
    # (renamed from "strategic_positioning" to avoid collision with the Định vị doc chunk).
    # brand_platform_tagline and brand_platform_manifesto are merged into brand_creative_output,
    # so they do not appear as standalone topics.
    assert "brand_strategic_positioning" in concept_topics, (
        "Expected 'brand_strategic_positioning' topic in concept_positioning"
    )

    # brand_creative_output absorbs visual_direction + tagline + manifesto + conclusion
    assert "brand_creative_output" in concept_topics, (
        "Expected 'brand_creative_output' topic in concept_positioning"
    )


def test_merged_brand_platform_chunks_have_substantial_content() -> None:
    payloads = chunk_payloads()
    merged_topics = {"brand_core", "brand_strategic_positioning", "brand_creative_output"}

    for topic in merged_topics:
        matching = [
            p for p in payloads
            if p["section"] == "concept_positioning" and p["topic"] == topic
        ]
        assert matching, f"No concept_positioning chunk with topic='{topic}' found"
        for p in matching:
            assert len(p["text"]) >= 300, (
                f"Merged chunk '{topic}' ({p['id']}) is too short ({len(p['text'])} chars); "
                "expected >= 300 chars after merging brand platform sub-items"
            )


def test_factsheet_chunk_count_is_stable() -> None:
    payloads = chunk_payloads()
    factsheet_chunks = [p for p in payloads if p["section"] == "factsheet"]
    assert len(factsheet_chunks) == 12, (
        f"Expected 12 factsheet chunks, got {len(factsheet_chunks)}"
    )
