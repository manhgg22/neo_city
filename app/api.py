"""app/api.py

FastAPI application for the NEO CITY RAG chatbot.

Endpoints
---------
GET  /         → serves the local chat demo UI (app/templates/chat.html)
GET  /health   → liveness check
POST /chat     → run the full pipeline and return an answer + debug metadata
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

app = FastAPI(title="neo-city-ai", version="0.1.0")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


class ChunkInfo(BaseModel):
    id: str
    section: str
    topic: str
    rerank_score: float


class ChatResponse(BaseModel):
    answer: str
    intent: str
    risk_level: str
    answer_mode: str
    confidence: float
    used_sections: list[str]
    target_sections: list[str]
    top_chunks: list[ChunkInfo]
    error: str = ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def chat_ui() -> FileResponse:
    """Serve the chat demo UI."""
    html_path = TEMPLATES_DIR / "chat.html"
    return FileResponse(str(html_path), media_type="text/html")


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> Any:
    """Run the full NEO CITY pipeline and return a structured response."""
    from app.answer import answer_from_retrieval, chatbot_answer_from_retrieval
    from app.retriever import retrieve

    question = request.question.strip()
    if not question:
        return ChatResponse(
            answer="Vui lòng nhập câu hỏi.",
            intent="",
            risk_level="",
            answer_mode="fallback",
            confidence=0.0,
            used_sections=[],
            target_sections=[],
            top_chunks=[],
        )

    try:
        retrieval_result = retrieve(
            question,
            limit=20,
            min_score=0.15,
            top_k=5,
        )
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content=ChatResponse(
                answer="Xin lỗi, hệ thống tìm kiếm tạm thời không khả dụng. Vui lòng thử lại.",
                intent="",
                risk_level="",
                answer_mode="fallback",
                confidence=0.0,
                used_sections=[],
                target_sections=[],
                top_chunks=[],
                error=f"{type(exc).__name__}: {exc}",
            ).model_dump(),
        )

    try:
        answer = answer_from_retrieval(retrieval_result)
        demo_answer = chatbot_answer_from_retrieval(retrieval_result)
    except Exception as exc:
        traceback.print_exc()
        from app.guardrails import FALLBACK_ANSWER
        answer_text = FALLBACK_ANSWER
        return ChatResponse(
            answer=answer_text,
            intent=str(retrieval_result.get("intent", "") or ""),
            risk_level=str(retrieval_result.get("risk_level", "") or ""),
            answer_mode="fallback",
            confidence=0.0,
            used_sections=[],
            target_sections=list(retrieval_result.get("target_sections", []) or []),
            top_chunks=[],
            error=f"{type(exc).__name__}: {exc}",
        )

    raw_chunks: list[dict] = list(retrieval_result.get("chunks", []) or [])
    top_chunks = [
        ChunkInfo(
            id=str(c.get("id", "") or ""),
            section=str(c.get("section", "") or ""),
            topic=str(c.get("topic", "") or ""),
            rerank_score=round(float(c.get("rerank_score", c.get("score", 0.0)) or 0.0), 4),
        )
        for c in raw_chunks[:5]
    ]

    return ChatResponse(
        answer=demo_answer,
        intent=str(retrieval_result.get("intent", "") or ""),
        risk_level=str(retrieval_result.get("risk_level", "") or ""),
        answer_mode=answer.answer_mode,
        confidence=answer.confidence,
        used_sections=list(answer.used_sections),
        target_sections=list(retrieval_result.get("target_sections", []) or []),
        top_chunks=top_chunks,
    )
