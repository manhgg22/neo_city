"""Task 5 — Retrieval Evaluation Script.

Runs 100-question evaluation against the NEO CITY Qdrant collection
and reports top-1/top-3/top-5 accuracy, must_contain hit rate, and
forbidden-section violations.

Usage
-----
    # Section-filtered mode: each query filtered to its expected_section
    python scripts/05_eval_retrieval.py --mode section-filtered

    # Global mode: only project filter applied; tests intent/ranking quality
    python scripts/05_eval_retrieval.py --mode global

    # Custom eval file / collection / limit
    python scripts/05_eval_retrieval.py --eval-file data/eval/retrieval_eval.jsonl \\
        --collection neo_city_chunks --limit 5 --mode global

Requirements
------------
- Qdrant must be running (QDRANT_URL from .env).
- Collection must already be populated.
- .env must be present in the project root.
"""

from __future__ import annotations

import argparse
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

from dotenv import load_dotenv

load_dotenv(ROOT_DIR / ".env", override=False)

from app.config import get_settings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

DEFAULT_EVAL_FILE = ROOT_DIR / "data" / "eval" / "retrieval_eval.jsonl"
DEFAULT_COLLECTION = "neo_city_chunks"
DEFAULT_LIMIT = 5
PROJECT_NAME = "NEO CITY"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_eval_cases(path: Path) -> list[dict[str, Any]]:
    """Load JSONL evaluation cases from *path*."""
    cases: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] Line {line_no}: {exc}", file=sys.stderr)
    return cases


# ---------------------------------------------------------------------------
# Embedding helper (shared with 04_test_retrieval.py)
# ---------------------------------------------------------------------------


def embed_query(query: str, model_name: str) -> list[float]:
    """Embed a query string using FastEmbed."""
    from fastembed import TextEmbedding  # type: ignore

    embedder = TextEmbedding(model_name=model_name)
    vectors = list(embedder.embed([query]))
    vec = vectors[0]
    if hasattr(vec, "tolist"):
        return vec.tolist()
    return list(vec)


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------


def build_filter(project: str, section: str | None) -> qmodels.Filter:
    """Build a Qdrant payload filter."""
    must: list[qmodels.FieldCondition] = [
        qmodels.FieldCondition(
            key="project",
            match=qmodels.MatchValue(value=project),
        )
    ]
    if section:
        must.append(
            qmodels.FieldCondition(
                key="section",
                match=qmodels.MatchValue(value=section),
            )
        )
    return qmodels.Filter(must=must)


def retrieve(
    query: str,
    *,
    client: QdrantClient,
    collection_name: str,
    model_name: str,
    section: str | None = None,
    limit: int = DEFAULT_LIMIT,
) -> list[qmodels.ScoredPoint]:
    """Embed query and search Qdrant, returning top *limit* hits."""
    query_vector = embed_query(query, model_name)
    query_filter = build_filter(PROJECT_NAME, section)
    hits: list[qmodels.ScoredPoint] = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    ).points
    return hits


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------


def _section_of(hit: qmodels.ScoredPoint) -> str:
    return (hit.payload or {}).get("section", "")


def _text_of(hit: qmodels.ScoredPoint) -> str:
    return (hit.payload or {}).get("text", "")


def evaluate_case(
    case: dict[str, Any],
    hits: list[qmodels.ScoredPoint],
) -> dict[str, Any]:
    """Evaluate a single test case against retrieved hits.

    Returns a result dict with pass/fail flags and diagnostics.
    """
    expected_section = case.get("expected_section", "")
    must_contain: list[str] = case.get("must_contain", [])
    forbidden_sections: list[str] = case.get("forbidden_sections", [])

    # -- section accuracy --
    sections_top5 = [_section_of(h) for h in hits]
    top1_section = sections_top5[0] if sections_top5 else ""
    top1_section_ok = top1_section == expected_section
    top3_section_ok = any(s == expected_section for s in sections_top5[:3])
    top5_section_ok = any(s == expected_section for s in sections_top5[:5])

    # -- must_contain: any hit in top-5 has all required keywords --
    all_text_top5 = " ".join(_text_of(h) for h in hits)
    must_contain_ok = all(
        kw.lower() in all_text_top5.lower() for kw in must_contain
    ) if must_contain else True

    # -- forbidden section violations: top-1 is in forbidden list --
    forbidden_violation = top1_section in forbidden_sections if forbidden_sections else False

    # -- no result --
    no_result = len(hits) == 0

    return {
        "id": case["id"],
        "query": case["query"],
        "expected_section": expected_section,
        "top1_section": top1_section,
        "top1_score": hits[0].score if hits else 0.0,
        "top1_id": (hits[0].payload or {}).get("id", "") if hits else "",
        "top1_section_ok": top1_section_ok,
        "top3_section_ok": top3_section_ok,
        "top5_section_ok": top5_section_ok,
        "must_contain_ok": must_contain_ok,
        "forbidden_violation": forbidden_violation,
        "no_result": no_result,
        "sections_top5": sections_top5,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_SEP = "\u2550" * 72
_SEP_THIN = "\u2500" * 72


def _print_summary(
    results: list[dict[str, Any]],
    mode: str,
    elapsed: float,
) -> None:
    total = len(results)
    top1_ok = sum(1 for r in results if r["top1_section_ok"])
    top3_ok = sum(1 for r in results if r["top3_section_ok"])
    top5_ok = sum(1 for r in results if r["top5_section_ok"])
    must_ok = sum(1 for r in results if r["must_contain_ok"])
    forbidden = sum(1 for r in results if r["forbidden_violation"])
    no_result = sum(1 for r in results if r["no_result"])

    print(_SEP)
    print(f"  NEO CITY — Retrieval Evaluation Summary  [{mode.upper()} mode]")
    print(_SEP)
    print(f"  Total cases           : {total}")
    print(f"  Top-1 section accuracy: {top1_ok}/{total} = {top1_ok/total*100:.1f}%")
    print(f"  Top-3 section accuracy: {top3_ok}/{total} = {top3_ok/total*100:.1f}%")
    print(f"  Top-5 section accuracy: {top5_ok}/{total} = {top5_ok/total*100:.1f}%")
    print(f"  Must-contain hit rate : {must_ok}/{total} = {must_ok/total*100:.1f}%")
    print(f"  Forbidden violations  : {forbidden}")
    print(f"  No-result cases       : {no_result}")
    print(f"  Elapsed               : {elapsed:.1f}s")
    print(_SEP)


def _print_failed(results: list[dict[str, Any]], show_top: int = 30) -> None:
    failed = [r for r in results if not r["top1_section_ok"] or r["forbidden_violation"]]
    if not failed:
        print("  ✓ All cases passed top-1 section check.")
        return

    print(f"\n  Failed cases ({len(failed)} total, showing up to {show_top}):")
    print(_SEP_THIN)
    for r in failed[:show_top]:
        reason_parts = []
        if not r["top1_section_ok"]:
            reason_parts.append(
                f"top1_section={r['top1_section']!r} ≠ expected={r['expected_section']!r}"
            )
        if r["forbidden_violation"]:
            reason_parts.append(f"forbidden section violation: {r['top1_section']!r}")
        reason = " | ".join(reason_parts)

        print(
            f"  [{r['id']}] {r['query'][:60]}"
            f"\n    Expected: {r['expected_section']!r}"
            f"  Got: {r['top1_section']!r} (score={r['top1_score']:.4f})"
            f"\n    Top-5 sections: {r['sections_top5']}"
            f"\n    Reason: {reason}"
        )
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NEO CITY — Task 5 Retrieval Evaluation."
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=DEFAULT_EVAL_FILE,
        help=f"Path to JSONL eval file (default: {DEFAULT_EVAL_FILE}).",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Qdrant collection name (default: from .env QDRANT_COLLECTION_NAME).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Number of results per query (default: {DEFAULT_LIMIT}).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["section-filtered", "global"],
        default="global",
        help=(
            "section-filtered: filter project+expected_section per case. "
            "global: filter only by project (tests end-to-end ranking)."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()

    collection_name = args.collection or settings.qdrant_collection_name
    model_name = settings.embedding_model

    # -- load eval cases --
    eval_path = args.eval_file
    if not eval_path.exists():
        print(f"[ERROR] Eval file not found: {eval_path}", file=sys.stderr)
        sys.exit(1)

    cases = load_eval_cases(eval_path)
    if not cases:
        print("[ERROR] No valid eval cases found.", file=sys.stderr)
        sys.exit(1)

    print(_SEP)
    print(f"  NEO CITY — Task 5 Retrieval Evaluation")
    print(f"  Mode       : {args.mode}")
    print(f"  Eval file  : {eval_path}")
    print(f"  Collection : {collection_name}")
    print(f"  Model      : {model_name}")
    print(f"  Limit      : {args.limit}")
    print(f"  Cases      : {len(cases)}")
    print(_SEP)

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )

    results: list[dict[str, Any]] = []
    start_time = time.time()

    for i, case in enumerate(cases, start=1):
        query = case["query"]
        expected_section = case.get("expected_section")

        # In section-filtered mode, pass expected_section as filter
        section_filter = expected_section if args.mode == "section-filtered" else None

        hits = retrieve(
            query,
            client=client,
            collection_name=collection_name,
            model_name=model_name,
            section=section_filter,
            limit=args.limit,
        )

        result = evaluate_case(case, hits)
        results.append(result)

        top1_marker = "✓" if result["top1_section_ok"] else "✗"
        top3_marker = "✓" if result["top3_section_ok"] else "✗"
        print(
            f"  [{i:03d}] {top1_marker}top1 {top3_marker}top3  "
            f"{case['id']:<30s}  "
            f"got={result['top1_section']:<25s} "
            f"score={result['top1_score']:.3f}"
        )

    elapsed = time.time() - start_time

    print()
    _print_summary(results, args.mode, elapsed)
    _print_failed(results)

    # -- exit code: non-zero if top-3 < 70% --
    total = len(results)
    top3_pct = sum(1 for r in results if r["top3_section_ok"]) / total * 100
    if top3_pct < 70.0:
        print(
            f"\n  [WARN] top-3 accuracy {top3_pct:.1f}% is below 70% target.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()