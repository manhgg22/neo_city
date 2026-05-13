"""End-to-end validation: 20 manual questions through real retrieve → answer pipeline.

Run: python scripts/test_20_after.py
"""
from __future__ import annotations

import sys
import time
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.answer import chatbot_answer_from_retrieval
from app.retriever import (
    retrieve,
    _get_cached_cross_encoder,
    _DEFAULT_CROSS_ENCODER,
    _embed_sparse_query,
)

QUESTIONS = [
    "Dự án đã đủ điều kiện mở bán và nhận đặt cọc chưa?",
    "Nếu chưa mở bán thì bảng giá trong tài liệu có ý nghĩa gì?",
    "Căn 2PN+1 tổng giá dự kiến bao nhiêu và có phải giá chính thức không?",
    "Shophouse giá bao nhiêu và có được cam kết kinh doanh tốt không?",
    "Gia đình trẻ mua 2PN thì nên xem sản phẩm, chính sách hay persona nào?",
    "Người trẻ mua 1PN+1 có chính sách hỗ trợ gì?",
    "Nếu khách hỏi xa trung tâm quá thì sales nên xử lý thế nào?",
    "Mê Linh đi Nội Bài bao lâu và thông tin này nằm trong phần nào?",
    "Vành đai 4 có đảm bảo NEO CITY tăng giá không?",
    "NEO CITY có cam kết lợi nhuận cho nhà đầu tư không?",
    "Có được ký HĐMB tại thời điểm hiện tại không?",
    "Booking 50 triệu có hợp pháp không nếu dự án chưa mở bán?",
    "NEO CITY khác gì khu đô thị vùng ven bình thường?",
    "Một trạng thái sống mới có phải chỉ là slogan không?",
    "Tiện ích nào thật sự tạo đời sống cộng đồng?",
    "R&D Center có liên quan gì đến người làm công nghệ?",
    "Nếu khách có 3 tỷ thì chọn loại căn nào?",
    "Nếu khách có 10 tỷ thì chọn shophouse hay townhouse?",
    "Tài liệu có nói gì về rủi ro pháp lý khi truyền thông dự án không?",
    "Hãy trả lời: giá 2PN, tình trạng mở bán, và cảnh báo pháp lý hiện tại.",
]

# Guardrail checks per question
LEGAL_Q_IDX = {0, 1, 10, 11, 18, 19}   # questions requiring legal caution
PRICE_Q_IDX = {2, 3, 16, 17}            # questions requiring pricing caution
INVEST_Q_IDX = {8, 9}                   # questions requiring no-guarantee wording

# Phrases that must NOT appear in answers (positive guarantees)
PROHIBITED = [
    "đã đủ điều kiện mở bán",
    "có thể đặt cọc ngay",
    "đã được phép huy động vốn",
    "chắc chắn tăng giá",
    "cam kết lợi nhuận chắc chắn",
    "giá chính thức là",
    "chắc chắn sinh lời",
]

# Raw chunk headings that should NOT appear verbatim
RAW_HEADING_PATTERNS = [
    r"^VI\.",
    r"^[IVX]+\.\s+[A-ZĐÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂẮẶẶ]",
    r"\|\s*---",
    r"\| Loại sản phẩm \|",
]


def check_answer(idx: int, question: str, answer: str, intent: str) -> dict:
    ans_lower = answer.lower()

    # Raw heading check
    has_raw_heading = any(re.search(p, answer, re.MULTILINE) for p in RAW_HEADING_PATTERNS)

    # Prohibited claim check
    violations = [p for p in PROHIBITED if p.lower() in ans_lower]

    # Caution presence check
    needs_price_caution = idx in PRICE_Q_IDX
    needs_legal_caution = idx in LEGAL_Q_IDX
    needs_invest_caution = idx in INVEST_Q_IDX

    has_price_caution = any(kw in ans_lower for kw in [
        "dự kiến", "chưa phải giá", "tùy tòa", "định hướng"
    ])
    has_legal_caution = any(kw in ans_lower for kw in [
        "chưa mở bán", "chưa huy động vốn", "không nên", "chưa đủ điều kiện",
        "chưa có văn bản", "định hướng", "không xác nhận"
    ])
    has_invest_caution = any(kw in ans_lower for kw in [
        "không cam kết", "không đưa ra cam kết", "tham khảo", "không phải cam kết",
        "chỉ là cơ sở", "cam kết lợi nhuận" , "không cam"
    ])

    caution_ok = True
    caution_missing = []
    if needs_price_caution and not has_price_caution:
        caution_ok = False
        caution_missing.append("PRICE_CAUTION")
    if needs_legal_caution and not has_legal_caution:
        caution_ok = False
        caution_missing.append("LEGAL_CAUTION")
    if needs_invest_caution and not has_invest_caution:
        caution_ok = False
        caution_missing.append("INVEST_CAUTION")

    return {
        "has_raw_heading": has_raw_heading,
        "violations": violations,
        "caution_ok": caution_ok,
        "caution_missing": caution_missing,
    }


def main() -> None:
    print("=" * 70)
    print("NEO CITY — 20-question end-to-end validation (AFTER patch)")
    print("=" * 70)
    print("Pre-loading models...")
    t0 = time.perf_counter()
    _get_cached_cross_encoder(_DEFAULT_CROSS_ENCODER)
    _embed_sparse_query("khởi động")
    retrieve("khởi động", limit=1, min_score=0.9)   # warms embedder + Qdrant connection
    print(f"Models loaded in {(time.perf_counter()-t0)*1000:.0f}ms\n")

    results = []
    for idx, question in enumerate(QUESTIONS, 0):
        qn = idx + 1
        t1 = time.perf_counter()
        try:
            retrieval_result = retrieve(question, limit=20, min_score=0.15, top_k=5)
            answer = chatbot_answer_from_retrieval(retrieval_result)
            intent = retrieval_result.get("intent", "unknown")
            latency = (time.perf_counter() - t1) * 1000
            error = None
        except Exception as e:
            answer = f"[ERROR] {e}"
            intent = "error"
            latency = (time.perf_counter() - t1) * 1000
            error = str(e)

        checks = check_answer(idx, question, answer, intent)
        results.append({
            "qn": qn,
            "question": question,
            "intent": intent,
            "answer": answer,
            "latency_ms": latency,
            "error": error,
            **checks,
        })

        # Print summary
        flags = []
        if checks["has_raw_heading"]:
            flags.append("⚠ RAW_HEADING")
        if checks["violations"]:
            flags.append(f"❌ PROHIBITED: {checks['violations']}")
        if not checks["caution_ok"]:
            flags.append(f"⚠ MISSING_CAUTION: {checks['caution_missing']}")
        status = "✓" if not flags else " | ".join(flags)

        print(f"Q{qn:02d} [{intent:20s}] {latency:6.0f}ms  {status}")
        print(f"     Q: {question}")
        print(f"     A: {answer[:200].replace(chr(10), ' ')}")
        print()

    # Summary table
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = len(results)
    clean = sum(
        1 for r in results
        if not r["has_raw_heading"] and not r["violations"] and r["caution_ok"]
    )
    raw_heading_count = sum(1 for r in results if r["has_raw_heading"])
    violation_count = sum(1 for r in results if r["violations"])
    caution_missing_count = sum(1 for r in results if not r["caution_ok"])
    errors = sum(1 for r in results if r["error"])

    print(f"  Total questions   : {total}")
    print(f"  Clean (all checks): {clean}/{total}")
    print(f"  Errors (exception): {errors}")
    print(f"  Raw headings found: {raw_heading_count}")
    print(f"  Prohibited claims : {violation_count}")
    print(f"  Missing cautions  : {caution_missing_count}")

    if errors:
        print("\nERROR DETAILS:")
        for r in results:
            if r["error"]:
                print(f"  Q{r['qn']:02d}: {r['error'][:200]}")


if __name__ == "__main__":
    main()
