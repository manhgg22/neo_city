"""Send the 20 validation questions to a running NEO CITY server via /ask API.

Usage
-----
1. Start the server first:
       python scripts/server.py

2. Then run this script:
       python scripts/test_20_via_api.py [--url http://localhost:8000]

The script POSTs each question to /ask, applies the same pass/fail checks as
test_20_after.py, and prints a live table + summary.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    print("httpx not installed. Run: pip install httpx")
    sys.exit(1)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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

LEGAL_Q_IDX  = {0, 1, 10, 11, 18, 19}
PRICE_Q_IDX  = {2, 3, 16, 17}
INVEST_Q_IDX = {8, 9}

PROHIBITED = [
    "đã đủ điều kiện mở bán",
    "có thể đặt cọc ngay",
    "đã được phép huy động vốn",
    "chắc chắn tăng giá",
    "cam kết lợi nhuận chắc chắn",
    "giá chính thức là",
    "chắc chắn sinh lời",
]

RAW_HEADING_PATTERNS = [
    r"^VI\.",
    r"^[IVX]+\.\s+[A-ZĐÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂẮẶẶ]",
    r"\|\s*---",
    r"\| Loại sản phẩm \|",
]


def check_answer(idx: int, answer: str) -> dict:
    ans_lower = answer.lower()
    has_raw_heading = any(re.search(p, answer, re.MULTILINE) for p in RAW_HEADING_PATTERNS)
    violations = [p for p in PROHIBITED if p.lower() in ans_lower]

    has_price_caution = any(kw in ans_lower for kw in [
        "dự kiến", "chưa phải giá", "tùy tòa", "định hướng"
    ])
    has_legal_caution = any(kw in ans_lower for kw in [
        "chưa mở bán", "chưa huy động vốn", "không nên", "chưa đủ điều kiện",
        "chưa có văn bản", "định hướng", "không xác nhận"
    ])
    has_invest_caution = any(kw in ans_lower for kw in [
        "không cam kết", "không đưa ra cam kết", "tham khảo", "không phải cam kết",
        "chỉ là cơ sở", "cam kết lợi nhuận", "không cam"
    ])

    caution_ok = True
    caution_missing: list[str] = []
    if idx in PRICE_Q_IDX and not has_price_caution:
        caution_ok = False
        caution_missing.append("PRICE_CAUTION")
    if idx in LEGAL_Q_IDX and not has_legal_caution:
        caution_ok = False
        caution_missing.append("LEGAL_CAUTION")
    if idx in INVEST_Q_IDX and not has_invest_caution:
        caution_ok = False
        caution_missing.append("INVEST_CAUTION")

    return {
        "has_raw_heading": has_raw_heading,
        "violations": violations,
        "caution_ok": caution_ok,
        "caution_missing": caution_missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000", help="Server base URL")
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout (s)")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    ask_url = f"{base}/ask"

    print("=" * 70)
    print(f"NEO CITY — 20-question validation via API  ({ask_url})")
    print("=" * 70)

    # Check server health first
    try:
        with httpx.Client(timeout=5) as client:
            r = client.get(f"{base}/health")
            r.raise_for_status()
            health = r.json()
            if not health.get("models_loaded"):
                print("WARNING: server reports models not loaded yet. Continuing anyway.")
            else:
                print(f"Server ready — startup took {health.get('startup_time_ms', '?')} ms\n")
    except Exception as e:
        print(f"Cannot reach server at {base}: {e}")
        print("Start the server first: python scripts/server.py")
        sys.exit(1)

    results = []
    with httpx.Client(timeout=args.timeout) as client:
        for idx, question in enumerate(QUESTIONS):
            qn = idx + 1
            t0 = time.perf_counter()
            try:
                resp = client.post(ask_url, json={"query": question})
                resp.raise_for_status()
                data = resp.json()
                answer = data.get("answer", "")
                intent = data.get("intent", "unknown")
                latency = data.get("latency_ms", (time.perf_counter() - t0) * 1000)
                error = None
            except Exception as e:
                answer = f"[ERROR] {e}"
                intent = "error"
                latency = (time.perf_counter() - t0) * 1000
                error = str(e)

            checks = check_answer(idx, answer)
            results.append({
                "qn": qn,
                "question": question,
                "intent": intent,
                "answer": answer,
                "latency_ms": latency,
                "error": error,
                **checks,
            })

            flags = []
            if checks["has_raw_heading"]:
                flags.append("RAW_HEADING")
            if checks["violations"]:
                flags.append(f"PROHIBITED: {checks['violations']}")
            if not checks["caution_ok"]:
                flags.append(f"MISSING: {checks['caution_missing']}")
            status = "OK" if not flags else " | ".join(flags)
            mark = "✓" if not flags else "✗"

            print(f"Q{qn:02d} {mark} [{intent:20s}] {latency:6.0f}ms  {status}")
            print(f"     Q: {question}")
            print(f"     A: {answer[:220].replace(chr(10), ' ')}")
            print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = len(results)
    clean = sum(
        1 for r in results
        if not r["has_raw_heading"] and not r["violations"] and r["caution_ok"]
    )
    errors = sum(1 for r in results if r["error"])
    raw_cnt = sum(1 for r in results if r["has_raw_heading"])
    vio_cnt = sum(1 for r in results if r["violations"])
    mis_cnt = sum(1 for r in results if not r["caution_ok"])
    avg_lat = sum(r["latency_ms"] for r in results) / total

    print(f"  Total questions    : {total}")
    print(f"  Clean (all checks) : {clean}/{total}")
    print(f"  Avg latency        : {avg_lat:.0f} ms")
    print(f"  Errors (HTTP)      : {errors}")
    print(f"  Raw headings       : {raw_cnt}")
    print(f"  Prohibited claims  : {vio_cnt}")
    print(f"  Missing cautions   : {mis_cnt}")

    if errors:
        print("\nERROR DETAILS:")
        for r in results:
            if r["error"]:
                print(f"  Q{r['qn']:02d}: {r['error'][:200]}")

    sys.exit(0 if clean == total else 1)


if __name__ == "__main__":
    main()
