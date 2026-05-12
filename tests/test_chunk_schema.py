from __future__ import annotations

import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT_DIR / "data" / "schema" / "neo_city_schema.json"

REQUIRED_FIELDS = {
    "id",
    "project",
    "section",
    "topic",
    "source_doc",
    "source_title",
    "status",
    "legal_sensitivity",
    "version",
    "text",
    "chunk_index",
}

ALLOWED_SECTIONS = {
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

ALLOWED_STATUS_VALUES = {
    "marketing_core",
    "strategy_data",
    "estimated",
    "hypothetical_policy",
    "legal_sensitive",
    "market_reference",
    "draft",
}

ALLOWED_LEGAL_SENSITIVITY_VALUES = {
    "low",
    "medium",
    "high",
    "critical",
}


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_schema_file_exists() -> None:
    assert SCHEMA_PATH.exists(), f"Missing schema file: {SCHEMA_PATH}"


def test_schema_includes_required_fields() -> None:
    schema = load_schema()
    assert set(schema["required"]) == REQUIRED_FIELDS
    assert REQUIRED_FIELDS.issubset(schema["properties"])


def test_schema_includes_allowed_values() -> None:
    schema = load_schema()
    properties = schema["properties"]
    assert set(properties["section"]["enum"]) == ALLOWED_SECTIONS
    assert set(properties["status"]["enum"]) == ALLOWED_STATUS_VALUES
    assert (
        set(properties["legal_sensitivity"]["enum"])
        == ALLOWED_LEGAL_SENSITIVITY_VALUES
    )
