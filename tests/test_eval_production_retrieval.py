from __future__ import annotations

import importlib.util
from io import StringIO
from pathlib import Path
from contextlib import redirect_stdout


def _load_eval_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "06_eval_production_retrieval.py"
    spec = importlib.util.spec_from_file_location("eval_production_retrieval", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_module = _load_eval_module()


def test_extract_chunks_from_dict_payload() -> None:
    result = {
        "chunks": [
            {"id": "1", "section": "pricing"},
            {"id": "2", "section": "price_sheet"},
        ]
    }

    chunks = eval_module.extract_chunks(result)

    assert [chunk["id"] for chunk in chunks] == ["1", "2"]


def test_extract_chunks_from_list_result() -> None:
    chunks = eval_module.extract_chunks(
        [{"id": "1", "section": "legal"}, {"id": "2", "section": "pricing"}]
    )

    assert [chunk["section"] for chunk in chunks] == ["legal", "pricing"]


def test_extract_chunks_handles_none_or_non_chunk_payloads() -> None:
    assert eval_module.extract_chunks(None) == []
    assert eval_module.extract_chunks({"chunks": None}) == []
    assert eval_module.extract_chunks({"data": [{"id": "1"}, "x", None]}) == [{"id": "1"}]


def test_evaluate_case_detects_section_accuracy_and_forbidden_violation() -> None:
    case = {
        "id": "legal_001",
        "query": "Dự án đã mở bán chưa?",
        "expected_section": "legal",
        "forbidden_sections": ["pricing"],
        "must_contain": ["mở bán"],
    }
    result = {
        "intent": "legal",
        "risk_level": "critical",
        "chunks": [
            {
                "section": "pricing",
                "topic": "apartment_pricing",
                "text": "Mở bán giá dự kiến",
                "score": 0.4,
                "rerank_score": 0.8,
            },
            {
                "section": "legal",
                "topic": "legal_status_and_warnings",
                "text": "Chưa mở bán",
                "score": 0.3,
                "rerank_score": 0.7,
            },
        ],
    }

    evaluated = eval_module.evaluate_case(case, result)

    assert evaluated["top1_section_ok"] is False
    assert evaluated["top3_section_ok"] is True
    assert evaluated["forbidden_violation"] is True
    assert evaluated["high_risk_failure"] is True


def test_summarize_results_counts_failures() -> None:
    results = [
        {
            "top1_section_ok": True,
            "top3_section_ok": True,
            "top5_section_ok": True,
            "topic_hit5": None,
            "must_contain_ok": True,
            "forbidden_violation": False,
            "no_result": False,
            "high_risk_failure": False,
        },
        {
            "top1_section_ok": False,
            "top3_section_ok": True,
            "top5_section_ok": True,
            "topic_hit5": False,
            "must_contain_ok": False,
            "forbidden_violation": True,
            "no_result": False,
            "high_risk_failure": True,
        },
    ]

    summary = eval_module.summarize_results(results)

    assert summary["total"] == 2
    assert summary["top1_ok"] == 1
    assert summary["forbidden"] == 1
    assert summary["risk_failures"] == 1
    assert summary["pass_target"] is False


def test_evaluate_case_handles_missing_expected_fields_and_missing_rerank_score() -> None:
    case = {
        "id": "misc_001",
        "query": "Query without explicit expectations",
    }
    result = {"chunks": [{"section": "factsheet", "topic": None, "text": None, "score": "0.4"}]}

    evaluated = eval_module.evaluate_case(case, result)

    assert evaluated["expected_sections"] == []
    assert evaluated["expected_topics"] == []
    assert evaluated["top1_rerank"] == 0.0
    assert evaluated["must_contain_ok"] is True
    assert evaluated["forbidden_violation"] is False


def test_print_case_result_includes_top5_lists() -> None:
    result = {
        "id": "case_001",
        "top1_section_ok": True,
        "top3_section_ok": True,
        "top5_section_ok": True,
        "top1_section": "pricing",
        "top1_topic": "apartment_pricing",
        "top1_rerank": 0.91,
        "sections_top5": ["pricing", "price_sheet"],
        "topics_top5": ["apartment_pricing", "two_bedroom_policy"],
        "error": "",
    }
    buffer = StringIO()

    with redirect_stdout(buffer):
        eval_module.print_case_result(1, result)

    output = buffer.getvalue()
    assert "top5_sections=['pricing', 'price_sheet']" in output
    assert "top5_topics=['apartment_pricing', 'two_bedroom_policy']" in output


def test_print_grouped_failures_includes_expected_and_got_section_groups() -> None:
    results = [
        {
            "id": "pricing_001",
            "intent": "pricing",
            "expected_sections": ["pricing", "price_sheet"],
            "top1_section": "factsheet",
            "top1_section_ok": False,
            "forbidden_violation": False,
            "high_risk_failure": True,
            "no_result": False,
        },
        {
            "id": "legal_001",
            "intent": "legal",
            "expected_sections": ["legal"],
            "top1_section": "",
            "top1_section_ok": False,
            "forbidden_violation": False,
            "high_risk_failure": True,
            "no_result": True,
        },
    ]
    buffer = StringIO()

    with redirect_stdout(buffer):
        eval_module.print_grouped_failures(results)

    output = buffer.getvalue()
    assert "By Intent:" in output
    assert "By Expected Section:" in output
    assert "By Got Section:" in output
    assert "No-result cases: legal_001" in output
