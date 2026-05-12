from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import QdrantSettings, get_qdrant_settings
from qdrant_client import QdrantClient
from qdrant_client.http import models


TEST_COLLECTION_NAME = "neo_city_qdrant_healthcheck"
TEST_VECTOR_SIZE = 4
TEST_QUERY_VECTOR = [0.1, 0.2, 0.3, 0.4]


@dataclass(frozen=True)
class HealthcheckResult:
    qdrant_url: str
    connection_status: str
    collection_status: str
    dummy_points_upserted: int
    search_result_count: int
    cleaned_up: bool


def build_vector_config() -> models.VectorParams:
    """Return the fixed vector config for the healthcheck collection."""

    return models.VectorParams(
        size=TEST_VECTOR_SIZE,
        distance=models.Distance.COSINE,
    )


def build_dummy_points() -> list[models.PointStruct]:
    """Return the dummy points used for the local Qdrant healthcheck."""

    return [
        models.PointStruct(
            id=1,
            vector=[0.1, 0.2, 0.3, 0.4],
            payload={"text": "NEO CITY overview"},
        ),
        models.PointStruct(
            id=2,
            vector=[0.2, 0.1, 0.4, 0.3],
            payload={"text": "pricing policy"},
        ),
        models.PointStruct(
            id=3,
            vector=[0.4, 0.3, 0.2, 0.1],
            payload={"text": "legal status"},
        ),
    ]


def create_qdrant_client(settings: QdrantSettings) -> QdrantClient:
    """Create a Qdrant client from repository configuration."""

    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )


def ensure_test_collection(client: QdrantClient) -> str:
    """Create the test collection if needed and return its status."""

    if client.collection_exists(TEST_COLLECTION_NAME):
        return "reused"

    client.create_collection(
        collection_name=TEST_COLLECTION_NAME,
        vectors_config=build_vector_config(),
    )
    return "created"


def run_healthcheck(
    client: QdrantClient,
    qdrant_url: str,
    cleanup: bool = False,
) -> HealthcheckResult:
    """Run the local Qdrant healthcheck against the temporary collection."""

    client.get_collections()
    collection_status = ensure_test_collection(client)

    points = build_dummy_points()
    client.upsert(
        collection_name=TEST_COLLECTION_NAME,
        points=points,
        wait=True,
    )

    search_response = client.query_points(
        collection_name=TEST_COLLECTION_NAME,
        query=TEST_QUERY_VECTOR,
        limit=3,
        with_payload=True,
    )

    if cleanup:
        client.delete_collection(TEST_COLLECTION_NAME)

    return HealthcheckResult(
        qdrant_url=qdrant_url,
        connection_status="ok",
        collection_status=collection_status,
        dummy_points_upserted=len(points),
        search_result_count=len(search_response.points),
        cleaned_up=cleanup,
    )


def print_result(result: HealthcheckResult) -> None:
    """Print a concise human-readable healthcheck summary."""

    print(f"Qdrant URL: {result.qdrant_url}")
    print(f"Connection status: {result.connection_status}")
    print(f"Collection created/reused: {result.collection_status}")
    print(f"Dummy points upserted: {result.dummy_points_upserted}")
    print(f"Search result count: {result.search_result_count}")
    if result.cleaned_up:
        print(f"Cleanup: deleted {TEST_COLLECTION_NAME}")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the local Qdrant healthcheck."""

    parser = argparse.ArgumentParser(
        description="Verify the local Qdrant connection with a temporary collection.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete the temporary healthcheck collection after the test.",
    )
    return parser


def main() -> None:
    """CLI entrypoint for the local Qdrant healthcheck."""

    args = build_parser().parse_args()
    settings = get_qdrant_settings()
    client = create_qdrant_client(settings)
    result = run_healthcheck(client, settings.qdrant_url, cleanup=args.cleanup)
    print_result(result)


if __name__ == "__main__":
    main()
