"""Task 4C — Search Smoke Test.

Standalone smoke test for the upserted NEO CITY Qdrant collection.
Embeds queries locally with FastEmbed and searches Qdrant directly.
Does NOT depend on app.retriever or app.intent_classifier.

Usage
-----
    python scripts/04_test_retrieval.py
    python scripts/04_test_retrieval.py --query "Căn 2PN giá bao nhiêu?" --section pricing
    python scripts/04_test_retrieval.py --query "Dự án đã mở bán chưa?" --section legal
    python scripts/04_test_retrieval.py --query "Gia đình trẻ phù hợp sản phẩm nào?" --section personas
    python scripts/04_test_retrieval.py --limit 3
    python scripts/04_test_retrieval.py --query "Căn 2PN giá bao nhiêu?" --section pricing --preview 800

Requirements
------------
- Qdrant must be running locally (QDRANT_URL, default http://localhost:6333).
- Collection must already be populated (run step 4B first).
- .env must be present in the project root.
"""

from __future__ import annotations

import argparse
import sys
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

# ---------------------------------------------------------------------------
# Default smoke-test queries
# ---------------------------------------------------------------------------

DEFAULT_QUERIES: list[str] = [
    "NEO CITY là dự án gì?",
    "Căn 2PN giá bao nhiêu?",
    "Dự án đã mở bán chưa?",
    "Có được đặt cọc mua NEO CITY chưa?",
    "Gia đình trẻ phù hợp sản phẩm nào?",
    "NEO CITY có tiện ích gì?",
    "Khách hàng mục tiêu của NEO CITY là ai?",
    "Mê Linh kết nối sân bay Nội Bài thế nào?",
]

DEFAULT_LIMIT = 5
DEFAULT_PREVIEW = 800

# Project name constant used for mandatory payload filter
PROJECT_NAME = "NEO CITY"


# ---------------------------------------------------------------------------
# Pure helper functions (testable without live Qdrant)
# ---------------------------------------------------------------------------


def build_section_filter(section: str | None) -> qmodels.Filter:
    """Return a Qdrant payload filter that always restricts to project='NEO CITY'.

    If *section* is provided, also restricts to that section.

    Parameters
    ----------
    section:
        Section name string (e.g. "pricing", "legal") or None for no section
        restriction (only the project filter is applied).

    Returns
    -------
    qmodels.Filter
        Always returns a Filter (never None), because we always filter by project.
    """
    must_conditions: list[qmodels.FieldCondition] = [
        qmodels.FieldCondition(
            key="project",
            match=qmodels.MatchValue(value=PROJECT_NAME),
        )
    ]

    if section:
        must_conditions.append(
            qmodels.FieldCondition(
                key="section",
                match=qmodels.MatchValue(value=section),
            )
        )

    return qmodels.Filter(must=must_conditions)


def format_search_result(
    rank: int,
    hit: qmodels.ScoredPoint,
    preview: int = DEFAULT_PREVIEW,
) -> str:
    """Format a single ScoredPoint into a human-readable string.

    Parameters
    ----------
    rank:
        1-based result rank.
    hit:
        A Qdrant ScoredPoint with payload.
    preview:
        Number of characters to show from the text field.

    Returns
    -------
    str
        Multi-line formatted string.
    """
    payload: dict[str, Any] = hit.payload or {}
    text_preview = (payload.get("text") or "")[:preview].replace("\n", " ")
    lines = [
        f"  #{rank}",
        f"     score            : {hit.score:.4f}",
        f"     id               : {payload.get('id', '')}",
        f"     section          : {payload.get('section', '')}",
        f"     topic            : {payload.get('topic', '')}",
        f"     status           : {payload.get('status', '')}",
        f"     legal_sensitivity: {payload.get('legal_sensitivity', '')}",
        f"     source_title     : {payload.get('source_title', '')}",
        f"     text ({preview}c)    : {text_preview}",
    ]
    return "\n".join(lines)


def embed_query(query: str, model_name: str) -> list[float]:
    """Embed a single query string using FastEmbed.

    Parameters
    ----------
    query:
        The text to embed.
    model_name:
        FastEmbed model name (e.g. "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").

    Returns
    -------
    list[float]
        The embedding vector.
    """
    from fastembed import TextEmbedding  # type: ignore

    embedder = TextEmbedding(model_name=model_name)
    vectors = list(embedder.embed([query]))
    vec = vectors[0]
    if hasattr(vec, "tolist"):
        return vec.tolist()
    return list(vec)


def search_qdrant(
    query: str,
    *,
    client: QdrantClient,
    collection_name: str,
    model_name: str,
    section: str | None = None,
    limit: int = DEFAULT_LIMIT,
) -> list[qmodels.ScoredPoint]:
    """Embed query and search the Qdrant collection.

    Parameters
    ----------
    query:
        User query text.
    client:
        Connected QdrantClient.
    collection_name:
        Target collection name.
    model_name:
        FastEmbed model name used during upsert.
    section:
        Optional section filter (project filter is always applied).
    limit:
        Maximum number of results to return.

    Returns
    -------
    list[qmodels.ScoredPoint]
        Sorted list of scored points (highest score first).
    """
    query_vector = embed_query(query, model_name)
    query_filter = build_section_filter(section)

    hits: list[qmodels.ScoredPoint] = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    ).points
    return hits


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _separator(char: str = "\u2500", width: int = 72) -> None:
    print(char * width)


def _print_query_results(
    query: str,
    hits: list[qmodels.ScoredPoint],
    section: str | None,
    preview: int = DEFAULT_PREVIEW,
) -> None:
    _separator("\u2550")
    print(f"Query  : {query}")
    if section:
        print(f"Filter : project='NEO CITY', section={section!r}")
    else:
        print(f"Filter : project='NEO CITY'")
    _separator()

    if not hits:
        print("  \u26a0  No results returned.")
        return

    for rank, hit in enumerate(hits, start=1):
        print(format_search_result(rank, hit, preview=preview))
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task 4C \u2014 NEO CITY Qdrant search smoke test.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single custom query instead of the default query set.",
    )
    parser.add_argument(
        "--section",
        type=str,
        default=None,
        choices=[
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
        ],
        help="Optional section filter for all queries.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Number of results per query (default: {DEFAULT_LIMIT}).",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=DEFAULT_PREVIEW,
        help=f"Number of characters to preview from text field (default: {DEFAULT_PREVIEW}).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )

    queries = [args.query] if args.query else DEFAULT_QUERIES

    _separator("\u2550")
    print("NEO CITY \u2014 Task 4C Search Smoke Test")
    print(f"Collection : {settings.qdrant_collection_name}")
    print(f"Model      : {settings.embedding_model}")
    print(f"Limit      : {args.limit}")
    print(f"Preview    : {args.preview} chars")
    print(f"Project    : {PROJECT_NAME}  (always applied)")
    if args.section:
        print(f"Section    : {args.section}")
    _separator("\u2550")

    zero_result_queries: list[str] = []

    for query in queries:
        hits = search_qdrant(
            query,
            client=client,
            collection_name=settings.qdrant_collection_name,
            model_name=settings.embedding_model,
            section=args.section,
            limit=args.limit,
        )
        _print_query_results(query, hits, args.section, preview=args.preview)

        if not hits:
            zero_result_queries.append(query)

    _separator("\u2550")
    total = len(queries)
    passed = total - len(zero_result_queries)
    print(f"Smoke test complete: {passed}/{total} queries returned at least 1 result.")
    if zero_result_queries:
        print("Queries with zero results:")
        for q in zero_result_queries:
            print(f"  - {q!r}")
    _separator("\u2550")

    if zero_result_queries:
        sys.exit(1)


if __name__ == "__main__":
    main()