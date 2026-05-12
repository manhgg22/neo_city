from __future__ import annotations

from fastapi import FastAPI


app = FastAPI(title="neo-city-ai", version="0.1.0")


@app.get("/health")
def healthcheck() -> dict[str, str]:
    """Basic health endpoint for the API skeleton."""

    return {"status": "ok"}
