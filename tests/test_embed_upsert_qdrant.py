from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from app.config import Settings


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT_DIR / "scripts" / "03_embed_upsert_qdrant.py"


def load_embed_upsert_module():
    spec = importlib.util.spec_from_file_location("embed_upsert_script", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load embed/upsert module from {SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


embed_upsert = load_embed_upsert_module()


def sample_chunk() -> dict:
    return {
        "id": "neo_city_factsheet_001",
        "project": "NEO CITY",
        "section": "factsheet",
        "topic": "project_overview",
        "source_doc": "All database - NEO CITY.docx",
        "source_title": "FACTSHEET",
        "status": "estimated",
        "legal_sensitivity": "medium",
        "version": "2026-05",
        "text": "NEO CITY overview text.",
        "chunk_index": 1,
    }


class FakeEmbedder:
    def embed(self, texts, batch_size=16):
        for index, _ in enumerate(texts, start=1):
            yield [float(index), 0.0, 0.0, 0.0]


def test_load_chunks_jsonl(tmp_path: Path) -> None:
    input_path = tmp_path / "chunks.jsonl"
    chunks = [sample_chunk(), {**sample_chunk(), "id": "neo_city_factsheet_002"}]
    input_path.write_text(
        "\n".join(json.dumps(chunk, ensure_ascii=False) for chunk in chunks) + "\n",
        encoding="utf-8",
    )

    loaded = embed_upsert.load_chunks_jsonl(input_path)

    assert loaded == chunks


def test_payload_conversion_preserves_all_fields() -> None:
    payload = embed_upsert.chunk_to_payload(sample_chunk())

    assert payload == sample_chunk()


def test_chunk_uuid_is_deterministic() -> None:
    chunk_id = sample_chunk()["id"]

    assert embed_upsert.chunk_uuid(chunk_id) == embed_upsert.chunk_uuid(chunk_id)
    assert embed_upsert.chunk_uuid(chunk_id) != embed_upsert.chunk_uuid(
        "neo_city_factsheet_002"
    )


def test_dry_run_does_not_call_qdrant_upsert(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "chunks.jsonl"
    input_path.write_text(
        json.dumps(sample_chunk(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    settings = Settings.model_validate(
        {
            "EMBEDDING_PROVIDER": "fastembed",
            "QDRANT_URL": "http://localhost:6333",
            "QDRANT_API_KEY": "",
            "QDRANT_COLLECTION_NAME": "neo_city_chunks",
            "EMBEDDING_MODEL": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "RERANKER_MODEL": "none",
        }
    )

    monkeypatch.setattr(embed_upsert, "create_fastembed_model", lambda model_name: FakeEmbedder())
    monkeypatch.setattr(
        embed_upsert,
        "create_qdrant_client",
        lambda settings: (_ for _ in ()).throw(AssertionError("Qdrant client must not be created in dry-run")),
    )

    result = embed_upsert.run_dry_run(
        settings=settings,
        input_path=input_path,
        schema_path=ROOT_DIR / "data" / "schema" / "neo_city_schema.json",
        batch_size=2,
    )

    assert result.chunk_count == 1
    assert result.vector_size == 4


def test_create_vector_config_uses_cosine_distance() -> None:
    vector_config = embed_upsert.create_vector_config(384)

    assert vector_config.size == 384
    assert vector_config.distance == embed_upsert.models.Distance.COSINE


def test_build_batches_splits_items_by_batch_size() -> None:
    batches = list(embed_upsert.build_batches([1, 2, 3, 4, 5], batch_size=2))

    assert batches == [[1, 2], [3, 4], [5]]


def test_openai_api_key_is_not_required(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "")
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "neo_city_chunks")
    monkeypatch.setenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    monkeypatch.setenv("RERANKER_MODEL", "none")

    settings = Settings.from_env()

    assert settings.embedding_provider == "fastembed"
    assert settings.qdrant_url == "http://localhost:6333"
