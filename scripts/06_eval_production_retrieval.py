"""Production retrieval evaluation for the real Task 5-7 pipeline.

This script evaluates ``app.retriever.retrieve()`` against the retrieval
evaluation set. It is diagnostic only: it does not bypass the retriever,
does not call raw Qdrant, and does not alter retrieval behavior.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.retriever import retrieve

DEFAULT_EVAL_FILE = ROOT_DIR / "data" / "eval" / "retrieval_eval.jsonl"
DEFAULT_LIMIT = 20
DEFAULT_MIN_SCORE = 0.15
DEFAULT_TOP_K = 5

TARGET_TOP1 = 0.80
TARGET_TOP3 = 0.90
TARGET_TOP5 = 0.95

HIGH_RISK_LEVELS = {"high", "critical"}
SAFETY_INTENTS = {"legal", "pricing", "sales_policy"}

_SEP = "=" * 72
_SEP_THIN = "-" * 72


def load_eval_cases(path: Path) -> list[dict[str, Any]]:
    """Load JSONL evaluation cases from *path*."""
    cases: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Line {line_no}: {exc}", file=sys.stderr)
                continue
            if isinstance(parsed, dict):
                cases.append(parsed)
            else:
                print(f"[WARN] Line {line_no}: expected object, got {type(parsed).__name__}", file=sys.stderr)
    return cases


def extract_chunks(result: Any) -> list[dict[str, Any]]:
    """Extract returned chunks from either a list result or a dict payload."""
    if result is None:
        return []
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    if isinstance(result, dict):
        for key in ("chunks", "results", "retrieved_chunks", "data"):
            value = result.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _safe_result_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    return {}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    if value == "":
        return []
    return [str(value)]


def _safe_get_float(chunk: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = chunk.get(key, default)
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _chunk_section(chunk: dict[str, Any]) -> str:
    return str(chunk.get("section", "") or "")


def _chunk_topic(chunk: dict[str, Any]) -> str:
    return str(chunk.get("topic", "") or "")


def _chunk_text(chunk: dict[str, Any]) -> str:
    return str(chunk.get("text", "") or "")


def _top1(chunk_list: list[dict[str, Any]]) -> dict[str, Any] | None:
    return chunk_list[0] if chunk_list else None


def _expected_sections(case: dict[str, Any]) -> list[str]:
    sections = _as_list(case.get("expected_sections"))
    if not sections:
        sections = _as_list(case.get("expected_section"))

    expanded: list[str] = []
    for section in sections:
        if section not in expanded:
            expanded.append(section)
        if section == "pricing" and "price_sheet" not in expanded:
            expanded.append("price_sheet")
        if section == "sales_policy" and "price_sheet" not in expanded:
            expanded.append("price_sheet")
        if section == "factsheet" and "concept_positioning" not in expanded:
            expanded.append("concept_positioning")
        if section == "location_connectivity" and "market" not in expanded:
            expanded.append("market")
        if section == "sales_strategy" and "personas" not in expanded:
            expanded.append("personas")
    return expanded


def _expected_topics(case: dict[str, Any]) -> list[str]:
    return _as_list(case.get("expected_topics"))


def _must_not_sections(case: dict[str, Any]) -> list[str]:
    sections = _as_list(case.get("must_not_sections"))
    if sections:
        return sections
    return _as_list(case.get("forbidden_sections"))


def _must_contain_tokens(case: dict[str, Any]) -> list[str]:
    return _as_list(case.get("must_contain"))


def evaluate_case(case: dict[str, Any], result: Any) -> dict[str, Any]:
    """Evaluate one case against the production retriever output."""
    chunks = extract_chunks(result)
    expected_sections = _expected_sections(case)
    expected_topics = _expected_topics(case)
    forbidden_sections = _must_not_sections(case)
    must_contain_tokens = _must_contain_tokens(case)

    top1 = _top1(chunks)
    sections_top5 = [_chunk_section(chunk) for chunk in chunks[:5]]
    topics_top5 = [_chunk_topic(chunk) for chunk in chunks[:5]]
    all_text_top5 = " ".join(_chunk_text(chunk) for chunk in chunks[:5]).lower()

    top1_section = _chunk_section(top1 or {})
    top1_topic = _chunk_topic(top1 or {})
    top1_rerank = _safe_get_float(top1 or {}, "rerank_score", 0.0)
    top1_score = _safe_get_float(top1 or {}, "score", 0.0)

    top1_section_ok = bool(expected_sections) and top1_section in expected_sections
    top3_section_ok = any(section in expected_sections for section in sections_top5[:3])
    top5_section_ok = any(section in expected_sections for section in sections_top5[:5])

    topic_hit5: bool | None = None
    if expected_topics:
        topic_hit5 = any(topic in expected_topics for topic in topics_top5)

    must_contain_ok = True
    if expected_sections:
        must_contain_ok = must_contain_ok and top5_section_ok
    if expected_topics:
        must_contain_ok = must_contain_ok and bool(topic_hit5)
    if must_contain_tokens:
        must_contain_ok = must_contain_ok and all(
            token.lower() in all_text_top5 for token in must_contain_tokens
        )

    forbidden_violation = bool(top1_section and top1_section in forbidden_sections)
    no_result = not chunks

    result_dict = _safe_result_dict(result)
    case_intent = str(case.get("intent", "") or "")
    case_risk_level = str(case.get("risk_level", "") or "")
    intent = str(result_dict.get("intent", case_intent) or case_intent)
    risk_level = str(result_dict.get("risk_level", case_risk_level) or case_risk_level)

    high_risk_failure = False
    effective_risk = risk_level or case_risk_level
    effective_intent = intent or case_intent
    if effective_risk in HIGH_RISK_LEVELS or effective_intent in SAFETY_INTENTS:
        high_risk_failure = (
            no_result
            or not top1_section_ok
            or forbidden_violation
            or not must_contain_ok
            or (topic_hit5 is False)
        )

    return {
        "id": str(case.get("id", "") or ""),
        "query": str(case.get("query", "") or ""),
        "intent": intent,
        "risk_level": risk_level,
        "expected_sections": expected_sections,
        "expected_topics": expected_topics,
        "top1_section": top1_section,
        "top1_topic": top1_topic,
        "top1_score": top1_score,
        "top1_rerank": top1_rerank,
        "top1_section_ok": top1_section_ok,
        "top3_section_ok": top3_section_ok,
        "top5_section_ok": top5_section_ok,
        "topic_hit5": topic_hit5,
        "must_contain_ok": must_contain_ok,
        "forbidden_violation": forbidden_violation,
        "no_result": no_result,
        "high_risk_failure": high_risk_failure,
        "sections_top5": sections_top5,
        "topics_top5": topics_top5,
        "chunks": chunks,
        "error": str(result_dict.get("error", "") or ""),
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate evaluation metrics across all cases."""
    total = len(results)
    topic_cases = [result for result in results if result["topic_hit5"] is not None]

    top1_ok = sum(1 for result in results if result["top1_section_ok"])
    top3_ok = sum(1 for result in results if result["top3_section_ok"])
    top5_ok = sum(1 for result in results if result["top5_section_ok"])
    must_ok = sum(1 for result in results if result["must_contain_ok"])
    forbidden = sum(1 for result in results if result["forbidden_violation"])
    no_result = sum(1 for result in results if result["no_result"])
    risk_failures = sum(1 for result in results if result["high_risk_failure"])
    topic_hits = sum(1 for result in topic_cases if result["topic_hit5"])

    return {
        "total": total,
        "no_result": no_result,
        "top1_ok": top1_ok,
        "top3_ok": top3_ok,
        "top5_ok": top5_ok,
        "must_ok": must_ok,
        "forbidden": forbidden,
        "risk_failures": risk_failures,
        "topic_cases": len(topic_cases),
        "topic_hits": topic_hits,
        "top1_pct": (top1_ok / total) if total else 0.0,
        "top3_pct": (top3_ok / total) if total else 0.0,
        "top5_pct": (top5_ok / total) if total else 0.0,
        "must_pct": (must_ok / total) if total else 0.0,
        "topic_pct": (topic_hits / len(topic_cases)) if topic_cases else None,
        "pass_target": (
            total > 0
            and (top1_ok / total) >= TARGET_TOP1
            and (top3_ok / total) >= TARGET_TOP3
            and (top5_ok / total) >= TARGET_TOP5
            and forbidden == 0
            and risk_failures == 0
        ),
    }


def _marker(ok: bool) -> str:
    return "OK" if ok else "XX"


def print_case_result(index: int, result: dict[str, Any]) -> None:
    """Print one compact case line plus diagnostic top-5 context."""
    case_id = result["id"]
    top1 = _marker(result["top1_section_ok"])
    top3 = _marker(result["top3_section_ok"])
    top5 = _marker(result["top5_section_ok"])
    got_section = result["top1_section"] or "-"
    got_topic = result["top1_topic"] or "-"
    rerank = result["top1_rerank"]
    print(
        f"[{index:03d}] {top1}top1 {top3}top3 {top5}top5 "
        f"{case_id:<16s} got={got_section:<22s} "
        f"topic={got_topic:<30s} rerank={rerank:.3f}"
    )
    print(
        f"      top5_sections={result['sections_top5']} "
        f"top5_topics={result['topics_top5']}"
    )
    if result.get("error"):
        print(f"      error={result['error']}")


def print_summary(summary: dict[str, Any], elapsed: float) -> None:
    """Print final aggregate metrics."""
    total = summary["total"]
    topic_line = "n/a"
    if summary["topic_pct"] is not None:
        topic_line = (
            f"{summary['topic_hits']}/{summary['topic_cases']} = "
            f"{summary['topic_pct'] * 100:.1f}%"
        )

    print()
    print(_SEP)
    print("NEO CITY - Production Retrieval Evaluation Summary")
    print(_SEP)
    print(f"Total cases                 : {total}")
    print(
        f"Top-1 section accuracy      : {summary['top1_ok']}/{total} = "
        f"{summary['top1_pct'] * 100:.1f}%"
    )
    print(
        f"Top-3 section accuracy      : {summary['top3_ok']}/{total} = "
        f"{summary['top3_pct'] * 100:.1f}%"
    )
    print(
        f"Top-5 section accuracy      : {summary['top5_ok']}/{total} = "
        f"{summary['top5_pct'] * 100:.1f}%"
    )
    print(f"Topic hit@5                 : {topic_line}")
    print(
        f"Must-contain rate           : {summary['must_ok']}/{total} = "
        f"{summary['must_pct'] * 100:.1f}%"
    )
    print(f"Forbidden violations        : {summary['forbidden']}")
    print(f"No-result count             : {summary['no_result']}")
    print(f"High/critical risk failures : {summary['risk_failures']}")
    print(
        "Pass/fail against target    : "
        f"{'PASS' if summary['pass_target'] else 'FAIL'}"
    )
    print(f"Elapsed                     : {elapsed:.1f}s")
    print(_SEP)
    print(
        "Targets                     : "
        "top1 >= 80%, top3 >= 90%, top5 >= 95%, "
        "forbidden = 0, high/critical failures = 0"
    )


def print_failures(results: list[dict[str, Any]], show_top: int = 25) -> None:
    """Print the worst failing cases for debugging."""
    failed = [
        result
        for result in results
        if (
            not result["top1_section_ok"]
            or result["forbidden_violation"]
            or result["high_risk_failure"]
            or result["no_result"]
        )
    ]
    if not failed:
        print("All cases cleared the failure checks.")
        return

    print()
    print(f"Worst failing cases ({len(failed)} total, showing up to {show_top}):")
    print(_SEP_THIN)
    for result in failed[:show_top]:
        print(
            f"[{result['id']}] intent={result['intent'] or '-'} "
            f"risk={result['risk_level'] or '-'}"
        )
        print(
            f"  expected_sections={result['expected_sections']} "
            f"got={result['top1_section']!r} topic={result['top1_topic']!r}"
        )
        print(
            f"  top5_sections={result['sections_top5']} "
            f"top5_topics={result['topics_top5']}"
        )
        print(
            f"  flags: top1={result['top1_section_ok']} "
            f"forbidden={result['forbidden_violation']} "
            f"no_result={result['no_result']} "
            f"risk_failure={result['high_risk_failure']}"
        )
        if result.get("error"):
            print(f"  error: {result['error']}")
        print()


def print_grouped_failures(results: list[dict[str, Any]]) -> None:
    """Print grouped failure analysis by intent, expected/got section, and risk."""
    failed = [
        result
        for result in results
        if (
            not result["top1_section_ok"]
            or result["forbidden_violation"]
            or result["high_risk_failure"]
            or result["no_result"]
        )
    ]
    if not failed:
        return

    print()
    print("Grouped Failure Analysis")
    print(_SEP_THIN)

    by_intent: dict[str, list[str]] = {}
    by_expected_section: dict[str, list[str]] = {}
    by_got_section: dict[str, list[str]] = {}
    by_expected_got: dict[tuple[str, str], list[str]] = {}

    for result in failed:
        intent = result["intent"] or "unknown"
        expected_label = ",".join(result["expected_sections"]) if result["expected_sections"] else "missing"
        got_label = result["top1_section"] or "NONE"
        by_intent.setdefault(intent, []).append(result["id"])
        by_expected_section.setdefault(expected_label, []).append(result["id"])
        by_got_section.setdefault(got_label, []).append(result["id"])
        by_expected_got.setdefault((expected_label, got_label), []).append(result["id"])

    print("By Intent:")
    for intent, ids in sorted(by_intent.items()):
        print(f"  {intent}: {len(ids)} cases - {', '.join(ids[:5])}{'...' if len(ids) > 5 else ''}")
    print()

    print("By Expected Section:")
    for expected, ids in sorted(by_expected_section.items()):
        print(f"  {expected}: {len(ids)} cases - {', '.join(ids[:5])}{'...' if len(ids) > 5 else ''}")
    print()

    print("By Got Section:")
    for got, ids in sorted(by_got_section.items()):
        print(f"  {got}: {len(ids)} cases - {', '.join(ids[:5])}{'...' if len(ids) > 5 else ''}")
    print()

    print("By Expected -> Got Section:")
    for (expected, got), ids in sorted(by_expected_got.items()):
        print(f"  {expected} -> {got}: {len(ids)} cases - {', '.join(ids[:3])}{'...' if len(ids) > 3 else ''}")
    print()

    no_result_cases = [result["id"] for result in failed if result["no_result"]]
    if no_result_cases:
        print(f"No-result cases: {', '.join(no_result_cases)}")
        print()

    high_risk_cases = [result["id"] for result in failed if result["high_risk_failure"]]
    if high_risk_cases:
        print(f"High/critical risk failures: {', '.join(high_risk_cases[:10])}")
        if len(high_risk_cases) > 10:
            print(f"  ... and {len(high_risk_cases) - 10} more")
        print()


def main() -> None:
    eval_path = DEFAULT_EVAL_FILE
    if not eval_path.exists():
        print(f"[ERROR] Eval file not found: {eval_path}", file=sys.stderr)
        sys.exit(1)

    cases = load_eval_cases(eval_path)
    if not cases:
        print("[ERROR] No valid eval cases found.", file=sys.stderr)
        sys.exit(1)

    print(_SEP)
    print("NEO CITY - Production Retrieval Evaluation")
    print(f"Eval file : {eval_path}")
    print(f"Cases     : {len(cases)}")
    print(
        f"Retrieve  : limit={DEFAULT_LIMIT}, min_score={DEFAULT_MIN_SCORE}, "
        f"top_k={DEFAULT_TOP_K}"
    )
    print(_SEP)

    results: list[dict[str, Any]] = []
    start = time.time()
    for index, case in enumerate(cases, start=1):
        question = str(case.get("query", "") or "")
        try:
            result = retrieve(
                question,
                limit=DEFAULT_LIMIT,
                min_score=DEFAULT_MIN_SCORE,
                top_k=DEFAULT_TOP_K,
            )
        except Exception as exc:  # pragma: no cover
            result = {"chunks": [], "error": f"{type(exc).__name__}: {exc}"}
        case_result = evaluate_case(case, result)
        results.append(case_result)
        print_case_result(index, case_result)

    elapsed = time.time() - start
    summary = summarize_results(results)
    print_summary(summary, elapsed)
    print_failures(results)
    print_grouped_failures(results)

    if not summary["pass_target"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
