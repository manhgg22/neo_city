from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.config import Settings, get_settings
from qdrant_client import QdrantClient
from qdrant_client.http import models


DEFAULT_INPUT_PATH = ROOT_DIR / "data" / "processed" / "neo_city_chunks.jsonl"
DEFAULT_SCHEMA_PATH = ROOT_DIR / "data" / "schema" / "neo_city_schema.json"
DEFAULT_BATCH_SIZE = 16
PAYLOAD_INDEX_FIELDS = (
    "project",
    "section",
    "topic",
    "status",
    "legal_sensitivity",
    "source_title",
    "version",
)


@dataclass(frozen=True)
class SchemaRules:
    required_fields: tuple[str, ...]
    allowed_sections: frozenset[str]
    allowed_statuses: frozenset[str]
    allowed_legal_sensitivity: frozenset[str]
    project_name: str


@dataclass(frozen=True)
class DryRunResult:
    chunk_count: int
    embedding_provider: str
    embedding_model: str
    vector_size: int
    sample_payload: dict[str, Any]


@dataclass(frozen=True)
class UpsertResult:
    chunk_count: int
    embedding_provider: str
    embedding_model: str
    vector_size: int
    collection_name: str
    collection_status: str
    points_upserted: int


def load_schema_rules(schema_path: Path) -> SchemaRules:
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    properties = payload["properties"]
    return SchemaRules(
        required_fields=tuple(payload["required"]),
        allowed_sections=frozenset(properties["section"]["enum"]),
        allowed_statuses=frozenset(properties["status"]["enum"]),
        allowed_legal_sensitivity=frozenset(properties["legal_sensitivity"]["enum"]),
        project_name=properties["project"]["const"],
    )


def load_chunks_jsonl(input_path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            payload = json.loads(stripped_line)
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_number} is not a JSON object.")
            chunks.append(payload)
    return chunks


def validate_chunk_payload(chunk: dict[str, Any], schema_rules: SchemaRules) -> None:
    missing_fields = [
        field_name for field_name in schema_rules.required_fields if field_name not in chunk
    ]
    if missing_fields:
        raise ValueError(f"Chunk is missing required fields: {missing_fields}")

    if chunk["project"] != schema_rules.project_name:
        raise ValueError(f"Invalid project value: {chunk['project']}")

    if chunk["section"] not in schema_rules.allowed_sections:
        raise ValueError(f"Invalid section value: {chunk['section']}")

    if chunk["status"] not in schema_rules.allowed_statuses:
        raise ValueError(f"Invalid status value: {chunk['status']}")

    if chunk["legal_sensitivity"] not in schema_rules.allowed_legal_sensitivity:
        raise ValueError(
            f"Invalid legal_sensitivity value: {chunk['legal_sensitivity']}"
        )

    for field_name in schema_rules.required_fields:
        value = chunk[field_name]
        if isinstance(value, str) and not value.strip():
            raise ValueError(f"Chunk field '{field_name}' must not be empty.")


def validate_chunks(chunks: Sequence[dict[str, Any]], schema_rules: SchemaRules) -> None:
    if not chunks:
        raise ValueError("No chunks were loaded from the JSONL file.")

    for chunk in chunks:
        validate_chunk_payload(chunk, schema_rules)


def chunk_to_payload(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": chunk["id"],
        "project": chunk["project"],
        "section": chunk["section"],
        "topic": chunk["topic"],
        "source_doc": chunk["source_doc"],
        "source_title": chunk["source_title"],
        "status": chunk["status"],
        "legal_sensitivity": chunk["legal_sensitivity"],
        "version": chunk["version"],
        "text": chunk["text"],
        "chunk_index": chunk["chunk_index"],
    }


def chunk_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def create_fastembed_model(model_name: str):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name)


def create_sparse_model():
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name="Qdrant/bm25")


def embed_texts(
    embedder: Any,
    texts: Sequence[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    embeddings = embedder.embed(texts, batch_size=batch_size)
    result: list[list[float]] = []
    for embedding in embeddings:
        if hasattr(embedding, "tolist"):
            result.append(embedding.tolist())
        else:
            result.append(list(embedding))
    return result


def embed_sparse_texts(
    sparse_embedder: Any,
    texts: Sequence[str],
) -> list[models.SparseVector]:
    result: list[models.SparseVector] = []
    for sv in sparse_embedder.embed(texts):
        indices = sv.indices.tolist() if hasattr(sv.indices, "tolist") else list(sv.indices)
        values = sv.values.tolist() if hasattr(sv.values, "tolist") else list(sv.values)
        result.append(models.SparseVector(indices=indices, values=values))
    return result


def build_batches(items: Sequence[Any], batch_size: int) -> Iterator[list[Any]]:
    for index in range(0, len(items), batch_size):
        yield list(items[index : index + batch_size])


def create_vector_config(vector_size: int) -> models.VectorParams:
    return models.VectorParams(
        size=vector_size,
        distance=models.Distance.COSINE,
    )


def create_hybrid_vector_config(vector_size: int) -> tuple[dict, dict]:
    dense_cfg = {"dense": models.VectorParams(size=vector_size, distance=models.Distance.COSINE)}
    sparse_cfg = {"sparse": models.SparseVectorParams()}
    return dense_cfg, sparse_cfg


def create_qdrant_client(settings: Settings) -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )


def extract_collection_vector_size(collection_info: models.CollectionInfo) -> int:
    vectors_config = collection_info.config.params.vectors
    if isinstance(vectors_config, models.VectorParams):
        return int(vectors_config.size)

    if isinstance(vectors_config, dict):
        first_config = next(iter(vectors_config.values()))
        return int(first_config.size)

    raise ValueError(f"Unsupported Qdrant vectors config: {type(vectors_config)!r}")


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    recreate: bool = False,
) -> str:
    dense_cfg, sparse_cfg = create_hybrid_vector_config(vector_size)

    if client.collection_exists(collection_name):
        if recreate:
            client.delete_collection(collection_name)
            client.create_collection(
                collection_name=collection_name,
                vectors_config=dense_cfg,
                sparse_vectors_config=sparse_cfg,
            )
            return "recreated"

        collection_info = client.get_collection(collection_name)
        existing_vector_size = extract_collection_vector_size(collection_info)
        if existing_vector_size != vector_size:
            raise ValueError(
                f"Existing collection '{collection_name}' uses vector size "
                f"{existing_vector_size}, but the FastEmbed model produced size "
                f"{vector_size}. Use --recreate to replace the collection explicitly."
            )
        return "reused"

    client.create_collection(
        collection_name=collection_name,
        vectors_config=dense_cfg,
        sparse_vectors_config=sparse_cfg,
    )
    return "created"


def create_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    for field_name in PAYLOAD_INDEX_FIELDS:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def build_points(
    chunks: Sequence[dict[str, Any]],
    vectors: Sequence[Sequence[float]],
    sparse_vectors: Sequence[models.SparseVector] | None = None,
) -> list[models.PointStruct]:
    if len(chunks) != len(vectors):
        raise ValueError("Chunk count and vector count must match.")

    points: list[models.PointStruct] = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
        if sparse_vectors is not None:
            vec = {"dense": list(vector), "sparse": sparse_vectors[i]}
        else:
            vec = list(vector)
        points.append(
            models.PointStruct(
                id=chunk_uuid(chunk["id"]),
                vector=vec,
                payload=chunk_to_payload(chunk),
            )
        )
    return points


def validate_provider(settings: Settings) -> None:
    if settings.embedding_provider.strip().lower() != "fastembed":
        raise ValueError(
            f"Unsupported EMBEDDING_PROVIDER '{settings.embedding_provider}'. "
            "This script only supports 'fastembed'."
        )


def run_dry_run(
    settings: Settings,
    input_path: Path,
    schema_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> DryRunResult:
    validate_provider(settings)
    chunks = load_chunks_jsonl(input_path)
    schema_rules = load_schema_rules(schema_path)
    validate_chunks(chunks, schema_rules)

    sample_chunks = chunks[: min(3, len(chunks))]
    embedder = create_fastembed_model(settings.embedding_model)
    sample_vectors = embed_texts(
        embedder,
        [chunk["text"] for chunk in sample_chunks],
        batch_size=min(batch_size, len(sample_chunks)) or 1,
    )
    vector_size = len(sample_vectors[0])

    return DryRunResult(
        chunk_count=len(chunks),
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
        vector_size=vector_size,
        sample_payload=chunk_to_payload(sample_chunks[0]),
    )


def run_upsert(
    settings: Settings,
    input_path: Path,
    schema_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    recreate: bool = False,
) -> UpsertResult:
    validate_provider(settings)
    chunks = load_chunks_jsonl(input_path)
    schema_rules = load_schema_rules(schema_path)
    validate_chunks(chunks, schema_rules)

    client = create_qdrant_client(settings)

    # --- Phase 1: dense embeddings (load e5-large alone to minimise peak RAM) ---
    all_texts = [chunk["text"] for chunk in chunks]
    print("Phase 1/2: computing dense embeddings...")
    embedder = create_fastembed_model(settings.embedding_model)
    all_dense: list[list[float]] = []
    for batch in build_batches(all_texts, batch_size):
        all_dense.extend(embed_texts(embedder, batch, batch_size=len(batch) or 1))
    vector_size = len(all_dense[0])
    del embedder  # free e5-large before loading sparse model
    import gc; gc.collect()

    collection_status = ensure_collection(
        client,
        settings.qdrant_collection_name,
        vector_size,
        recreate=recreate,
    )
    create_payload_indexes(client, settings.qdrant_collection_name)

    # --- Phase 2: sparse BM25 embeddings + upsert ---
    print("Phase 2/2: computing sparse embeddings and upserting...")
    sparse_embedder = create_sparse_model()
    points_upserted = 0
    all_batches = list(build_batches(list(range(len(chunks))), batch_size))
    for idx_batch in all_batches:
        chunk_batch = [chunks[i] for i in idx_batch]
        batch_texts = [all_texts[i] for i in idx_batch]
        dense_batch = [all_dense[i] for i in idx_batch]
        sparse_vecs = embed_sparse_texts(sparse_embedder, batch_texts)

        points = build_points(chunk_batch, dense_batch, sparse_vecs)
        client.upsert(
            collection_name=settings.qdrant_collection_name,
            points=points,
            wait=True,
        )
        points_upserted += len(points)

    return UpsertResult(
        chunk_count=len(chunks),
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
        vector_size=vector_size,
        collection_name=settings.qdrant_collection_name,
        collection_status=collection_status,
        points_upserted=points_upserted,
    )


def print_dry_run_result(result: DryRunResult) -> None:
    print(f"Chunks loaded: {result.chunk_count}")
    print(f"Embedding provider: {result.embedding_provider}")
    print(f"Embedding model: {result.embedding_model}")
    print(f"Vector size: {result.vector_size}")
    print("Sample payload:")
    print(json.dumps(result.sample_payload, ensure_ascii=False, indent=2))


def print_upsert_result(result: UpsertResult) -> None:
    print(f"Chunks loaded: {result.chunk_count}")
    print(f"Embedding provider: {result.embedding_provider}")
    print(f"Embedding model: {result.embedding_model}")
    print(f"Vector size: {result.vector_size}")
    print(f"Qdrant collection name: {result.collection_name}")
    print(f"Collection created or reused: {result.collection_status}")
    print(f"Number of points upserted: {result.points_upserted}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate local FastEmbed embeddings and upsert NEO CITY chunks into Qdrant.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the chunk JSONL file.",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help="Path to the chunk schema JSON file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size used for local embedding and Qdrant upsert.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate chunks and test local FastEmbed on the first 1-3 chunks without upserting to Qdrant.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the target Qdrant collection before upserting.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()

    if args.dry_run:
        result = run_dry_run(
            settings=settings,
            input_path=args.input,
            schema_path=args.schema,
            batch_size=args.batch_size,
        )
        print_dry_run_result(result)
        return

    result = run_upsert(
        settings=settings,
        input_path=args.input,
        schema_path=args.schema,
        batch_size=args.batch_size,
        recreate=args.recreate,
    )
    print_upsert_result(result)


if __name__ == "__main__":
    main()
