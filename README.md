# neo-city-ai

Initial project skeleton for a Qdrant-based RAG assistant for the NEO CITY knowledge base.

## Current scope

This repository currently includes:

- project structure and environment-based configuration
- DOCX parsing into section records
- structured chunk generation with schema validation
- local embedding and Qdrant upsert utilities
- intent classification
- Qdrant retrieval with metadata filters
- local deterministic reranking
- baseline and integration-oriented tests

## Project layout

```text
data/
  raw/
  processed/
  schema/
scripts/
app/
tests/
```

## Setup

1. Create a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Copy `.env.example` to `.env` and fill in real values.
4. Run `pytest`.

## Notes

- The source document is expected at `data/raw/All database - NEO CITY.docx`.
- Do not commit `.env`.
- Follow the guardrails and schema constraints defined in `AGENTS.md`.

## Local Qdrant healthcheck

1. Create `.env` from `.env.example`.
2. Set `QDRANT_URL` to your local server, for example `http://localhost:6333`.
3. Set `QDRANT_API_KEY` only if your local Qdrant instance requires one.
4. `OPENAI_API_KEY` is not required for this healthcheck.
5. Run `python scripts/03_test_qdrant_local.py --cleanup`.

The script creates a temporary collection named `neo_city_qdrant_healthcheck`, upserts three dummy points, runs a vector query, and optionally deletes the collection at the end.
