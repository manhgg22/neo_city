"""NEO CITY API Server

Loads all models once on startup, then serves queries via HTTP.
Existing code (retriever, answer, intent_classifier) is untouched.

Endpoints
---------
GET  /health          — server + model status
POST /ask             — concise customer-facing answer
POST /ask/debug       — full retrieval detail (intent, chunks, scores)
GET  /stats           — request count, avg latency
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

TEMPLATES_DIR = ROOT_DIR / "app" / "templates"

from app.answer import answer_from_retrieval, chatbot_answer_from_retrieval
from app.config import get_settings
from app.retriever import (
    _DEFAULT_CROSS_ENCODER,
    _DEFAULT_SPARSE_MODEL,
    _CROSS_ENCODER_CACHE,
    _EMBEDDER_CACHE,
    _SPARSE_EMBEDDER_CACHE,
    _embed_sparse_query,
    _get_cached_cross_encoder,
    embed_query,
    retrieve,
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NEO CITY RAG API",
    description="Intent-aware retrieval + cross-encoder reranking + BM25 hybrid search",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory stats
# ---------------------------------------------------------------------------

_stats: dict[str, Any] = {
    "total_requests": 0,
    "total_latency_ms": 0.0,
    "models_loaded": False,
    "startup_time_ms": 0.0,
}


# ---------------------------------------------------------------------------
# Startup — load all three models once
# ---------------------------------------------------------------------------

@app.on_event("startup")
def load_models() -> None:
    settings = get_settings()
    t0 = time.perf_counter()
    print("NEO CITY server — loading models, please wait...")

    embed_query("khởi động", settings.embedding_model)
    _get_cached_cross_encoder(_DEFAULT_CROSS_ENCODER)
    _embed_sparse_query("khởi động")

    elapsed = (time.perf_counter() - t0) * 1000
    _stats["models_loaded"] = True
    _stats["startup_time_ms"] = round(elapsed, 1)
    print(f"Models loaded in {elapsed:.0f} ms. Server ready.")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    query: str
    limit: int = 20
    min_score: float = 0.15
    top_k: int = 5


class AskResponse(BaseModel):
    answer: str
    intent: str
    latency_ms: float


class DebugChunk(BaseModel):
    id: str
    section: str
    topic: str
    score: float
    rerank_score: float
    cross_encoder_score: float
    text_preview: str


class DebugResponse(BaseModel):
    answer: str
    intent: str
    risk_level: str
    target_sections: list[str]
    must_use_legal_only: bool
    answer_mode: str
    chunks: list[DebugChunk]
    latency_ms: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_retrieve(req: AskRequest) -> tuple[dict, float]:
    t0 = time.perf_counter()
    result = retrieve(
        req.query,
        limit=req.limit,
        min_score=req.min_score,
        top_k=req.top_k,
    )
    latency = (time.perf_counter() - t0) * 1000
    _stats["total_requests"] += 1
    _stats["total_latency_ms"] += latency
    return result, round(latency, 1)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def dashboard_ui() -> HTMLResponse:
    html = (TEMPLATES_DIR / "dashboard.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "models_loaded": _stats["models_loaded"],
        "startup_time_ms": _stats["startup_time_ms"],
        "embedding_model": settings.embedding_model,
        "cross_encoder": _DEFAULT_CROSS_ENCODER,
        "sparse_model": _DEFAULT_SPARSE_MODEL,
        "embedder_cached": settings.embedding_model in _EMBEDDER_CACHE,
        "cross_encoder_cached": _DEFAULT_CROSS_ENCODER in _CROSS_ENCODER_CACHE,
        "sparse_cached": _DEFAULT_SPARSE_MODEL in _SPARSE_EMBEDDER_CACHE,
        "qdrant_url": settings.qdrant_url,
        "collection": settings.qdrant_collection_name,
    }


@app.get("/stats")
def stats() -> dict:
    total = _stats["total_requests"]
    avg = (_stats["total_latency_ms"] / total) if total > 0 else 0.0
    return {
        "total_requests": total,
        "avg_latency_ms": round(avg, 1),
        "startup_time_ms": _stats["startup_time_ms"],
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    result, latency = _run_retrieve(req)
    answer = chatbot_answer_from_retrieval(result)

    return AskResponse(
        answer=answer,
        intent=result.get("intent", ""),
        latency_ms=latency,
    )


@app.post("/ask/debug", response_model=DebugResponse)
def ask_debug(req: AskRequest) -> DebugResponse:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    result, latency = _run_retrieve(req)
    answer_result = answer_from_retrieval(result)
    concise = chatbot_answer_from_retrieval(result)

    chunks = [
        DebugChunk(
            id=str(c.get("id", "")),
            section=str(c.get("section", "")),
            topic=str(c.get("topic", "")),
            score=round(float(c.get("score", 0.0)), 4),
            rerank_score=round(float(c.get("rerank_score", 0.0)), 4),
            cross_encoder_score=round(float(c.get("cross_encoder_score", 0.0)), 4),
            text_preview=(c.get("text", "") or "")[:200],
        )
        for c in result.get("chunks", [])
    ]

    return DebugResponse(
        answer=concise,
        intent=result.get("intent", ""),
        risk_level=result.get("risk_level", ""),
        target_sections=result.get("target_sections", []),
        must_use_legal_only=bool(result.get("must_use_legal_only", False)),
        answer_mode=answer_result.answer_mode,
        chunks=chunks,
        latency_ms=latency,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="NEO CITY API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Dev auto-reload (disables startup preload)")
    args = parser.parse_args()

    uvicorn.run(
        "scripts.server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
