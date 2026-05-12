# NEO CITY AI Assistant - Coding Instructions

## Goal

Build a Qdrant-based RAG assistant for the NEO CITY project with a target answer accuracy above 90%.

The assistant must answer only from the provided NEO CITY knowledge base. It must avoid hallucination, especially for pricing, sales policy, legal status, project launch status, deposit/payment, and investment return claims.

## Core Architecture

The pipeline must be:

1. Parse source documents
2. Split content into structured chunks by heading
3. Add metadata to every chunk
4. Embed chunks
5. Upsert chunks into Qdrant
6. Classify user intent before retrieval
7. Retrieve from Qdrant using metadata filters
8. Rerank retrieved chunks
9. Apply guardrails
10. Generate answer only from retrieved context
11. If context is insufficient, return a fallback answer
12. Log queries, retrieved chunks, answer, and confidence

## Required Chunk Schema

Every chunk must follow this schema:

```json
{
  "id": "neo_city_factsheet_001",
  "project": "NEO CITY",
  "section": "factsheet",
  "topic": "project_overview",
  "source_doc": "All database - NEO CITY.docx",
  "source_title": "FACTSHEET DỰ ÁN",
  "status": "estimated",
  "legal_sensitivity": "medium",
  "version": "2026-05",
  "text": "..."
}
```

Required fields:

- `id`
- `project`
- `section`
- `topic`
- `source_doc`
- `source_title`
- `status`
- `legal_sensitivity`
- `version`
- `text`

## Allowed Sections

- `factsheet`
- `location_connectivity`
- `personas`
- `concept_positioning`
- `sales_strategy`
- `sales_policy`
- `legal`
- `pricing`
- `market`
- `price_sheet`

## Allowed Status Values

- `marketing_core`
- `strategy_data`
- `estimated`
- `hypothetical_policy`
- `legal_sensitive`
- `market_reference`
- `draft`

## Allowed Legal Sensitivity Values

- `low`
- `medium`
- `high`
- `critical`

## Guardrail Rules

### Pricing

For pricing questions, always use cautious wording:

- "theo tài liệu định hướng hiện tại"
- "dự kiến"
- "tùy tòa, tầng, view, thời điểm mở bán và chính sách từng đợt"
- "chưa phải giá bán chính thức nếu chưa có công bố chính thức"

Never say:

- "giá chính thức là"
- "chắc chắn giá là"
- "cam kết mức giá"

### Legal

For legal, opening for sale, deposit, fundraising, and official transaction questions:

- Only use chunks from `section = legal`
- Do not infer from marketing or sales content
- If context is not explicit, return fallback

Never say:

- "đã đủ điều kiện mở bán"
- "có thể đặt cọc ngay"
- "đã được phép huy động vốn"

unless legal context explicitly confirms it.

### Investment Return

Never claim guaranteed appreciation or guaranteed profit.

If asked, answer:

"Tài liệu hiện tại không đưa ra cam kết lợi nhuận. Các luận điểm về hạ tầng, thị trường và xu hướng giãn dân chỉ nên được hiểu là cơ sở tham khảo, không phải cam kết tăng giá hoặc cam kết sinh lời."

### Fallback

If retrieved context is insufficient, answer:

"Tôi chưa tìm thấy dữ liệu đủ rõ trong tài liệu NEO CITY hiện tại để trả lời chính xác câu hỏi này."

## Coding Rules

- Use Python.
- Use clear modules.
- Add tests for every important function.
- Keep functions small and typed.
- Do not hardcode secrets.
- Read API keys and Qdrant settings from `.env`.
- Do not commit `.env`.
- Prefer simple, maintainable code over complex abstractions.

## Expected Folder Structure

```text
neo-city-ai/
├── data/
│   ├── raw/
│   │   └── All database - NEO CITY.docx
│   ├── processed/
│   │   └── neo_city_chunks.jsonl
│   └── schema/
│       └── neo_city_schema.json
├── scripts/
│   ├── 01_parse_docx.py
│   ├── 02_create_chunks.py
│   ├── 03_embed_upsert_qdrant.py
│   └── 04_test_retrieval.py
├── app/
│   ├── config.py
│   ├── intent_classifier.py
│   ├── retriever.py
│   ├── guardrails.py
│   ├── answer.py
│   └── api.py
├── tests/
│   ├── test_chunk_schema.py
│   ├── test_intent_classifier.py
│   ├── test_guardrails.py
│   └── test_retriever.py
├── requirements.txt
├── .env.example
├── AGENTS.md
└── README.md
```

## Accuracy Target

The project must include an evaluation set of 100 questions.

A result is considered correct if:

- It retrieves the correct section
- It answers according to source context
- It does not hallucinate
- It applies pricing/legal/policy guardrails correctly

Target:

- At least 90 correct answers out of 100
- Zero severe errors for legal, pricing, deposit, opening for sale, or guaranteed profit questions

---



## Agent Workflow

Before making any changes, read this AGENTS.md file and follow it strictly.

For every task:
1. Implement only the requested scope.
2. Do not change unrelated files.
3. Add or update tests.
4. Run tests.
5. Report files changed, commands run, test results, and assumptions. 
## Scope Control

Do not implement future tasks unless explicitly requested.

For example:
- If the task is to create project skeleton, do not implement DOCX parsing.
- If the task is to parse DOCX, do not implement Qdrant upsert.
- If the task is to build retriever, do not rewrite the chunking pipeline.
## Intent to Section Mapping

Use this mapping for retrieval:

- project_overview → factsheet, concept_positioning
- amenities → factsheet
- product → factsheet, pricing
- location → location_connectivity, market
- persona → personas
- concept → concept_positioning
- sales_strategy → sales_strategy, personas
- sales_policy → sales_policy, price_sheet
- pricing → pricing, price_sheet
- legal → legal only
- market → market
- unknown → no retrieval or fallback
## Risk Level Mapping

- low: factsheet, personas, concept_positioning, sales_strategy
- medium: location_connectivity, market
- high: pricing, sales_policy, price_sheet
- critical: legal, deposit, opening for sale, fundraising, official transaction, guaranteed return
## Default Commands

Use these commands unless the project introduces a different runner:

```bash
python -m pytest
python scripts/01_parse_docx.py
python scripts/02_create_chunks.py
python scripts/03_embed_upsert_qdrant.py --dry-run

---

# Bản đánh giá nhanh

| Hạng mục | Đánh giá |
|---|---|
| Goal | Tốt |
| Architecture | Tốt |
| Schema | Tốt |
| Sections | Tốt |
| Guardrails | Rất tốt |
| Coding rules | Tốt |
| Folder structure | Tốt |
| Accuracy target | Tốt |
| Thiếu intent mapping | Nên bổ sung |
| Thiếu scope control | Nên bổ sung |
| Thiếu workflow cho Codex | Nên bổ sung |

## Kết luận

File hiện tại dùng được rồi, nhưng nếu bạn muốn Codex làm “ít lệch hướng” hơn, hãy bổ sung thêm 5 phần trên.

Sau khi sửa, bạn có thể giao ngay prompt đầu tiên cho Codex:

```text
Read AGENTS.md first.

Implement Task 1 only: create the initial project skeleton, schema file, config file, requirements.txt, .env.example, README.md, and basic schema tests.

Do not implement DOCX parsing yet.

Run pytest and report:
1. files changed
2. commands run
3. test results
4. assumptions