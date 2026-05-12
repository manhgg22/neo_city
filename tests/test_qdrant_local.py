from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT_DIR / "scripts" / "03_test_qdrant_local.py"


def load_qdrant_healthcheck_module():
    spec = importlib.util.spec_from_file_location("qdrant_healthcheck_script", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load healthcheck module from {SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


qdrant_healthcheck = load_qdrant_healthcheck_module()


class FakeQdrantClient:
    def __init__(self, exists: bool = False) -> None:
        self.exists = exists
        self.create_collection_calls: list[tuple[str, object]] = []
        self.upsert_calls: list[tuple[str, list[object], bool]] = []
        self.query_points_calls: list[tuple[str, list[float], int, bool]] = []
        self.deleted_collections: list[str] = []
        self.connected = False

    def get_collections(self) -> SimpleNamespace:
        self.connected = True
        return SimpleNamespace(collections=[])

    def collection_exists(self, collection_name: str) -> bool:
        return self.exists

    def create_collection(self, collection_name: str, vectors_config: object) -> bool:
        self.create_collection_calls.append((collection_name, vectors_config))
        self.exists = True
        return True

    def upsert(self, collection_name: str, points: list[object], wait: bool) -> SimpleNamespace:
        self.upsert_calls.append((collection_name, points, wait))
        return SimpleNamespace(status="ok")

    def query_points(
        self,
        collection_name: str,
        query: list[float],
        limit: int,
        with_payload: bool,
    ) -> SimpleNamespace:
        self.query_points_calls.append((collection_name, query, limit, with_payload))
        return SimpleNamespace(points=[1, 2, 3])

    def delete_collection(self, collection_name: str) -> bool:
        self.deleted_collections.append(collection_name)
        return True


def test_build_vector_config_uses_expected_shape() -> None:
    vector_config = qdrant_healthcheck.build_vector_config()

    assert vector_config.size == 4
    assert vector_config.distance == qdrant_healthcheck.models.Distance.COSINE


def test_build_dummy_points_returns_expected_points() -> None:
    points = qdrant_healthcheck.build_dummy_points()

    assert len(points) == 3
    assert points[0].payload["text"] == "NEO CITY overview"
    assert points[1].payload["text"] == "pricing policy"
    assert points[2].payload["text"] == "legal status"


def test_run_healthcheck_creates_collection_and_cleans_up() -> None:
    client = FakeQdrantClient(exists=False)

    result = qdrant_healthcheck.run_healthcheck(
        client,
        qdrant_url="http://localhost:6333",
        cleanup=True,
    )

    assert client.connected is True
    assert result.connection_status == "ok"
    assert result.collection_status == "created"
    assert result.dummy_points_upserted == 3
    assert result.search_result_count == 3
    assert client.create_collection_calls
    assert client.upsert_calls
    assert client.query_points_calls
    assert client.deleted_collections == [qdrant_healthcheck.TEST_COLLECTION_NAME]


def test_run_healthcheck_reuses_existing_collection_without_cleanup() -> None:
    client = FakeQdrantClient(exists=True)

    result = qdrant_healthcheck.run_healthcheck(
        client,
        qdrant_url="http://localhost:6333",
        cleanup=False,
    )

    assert result.collection_status == "reused"
    assert client.create_collection_calls == []
    assert client.deleted_collections == []
