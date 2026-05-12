"""Helper: run pytest + smoke tests and save results to results_summary.txt."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_summary.txt"

COMMANDS = [
    ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
    ["python", "scripts/04_test_retrieval.py"],
    ["python", "scripts/04_test_retrieval.py", "--query", "Căn 2PN giá bao nhiêu?", "--section", "pricing"],
    ["python", "scripts/04_test_retrieval.py", "--query", "Dự án đã mở bán chưa?", "--section", "legal"],
    ["python", "scripts/04_test_retrieval.py", "--query", "Gia đình trẻ phù hợp sản phẩm nào?", "--section", "personas"],
]

LABELS = [
    "=== PYTEST ===",
    "=== DEFAULT QUERY ===",
    "=== PRICING QUERY ===",
    "=== LEGAL QUERY ===",
    "=== PERSONAS QUERY ===",
]

with open(OUT, "w", encoding="utf-8") as f:
    for label, cmd in zip(LABELS, COMMANDS):
        f.write(f"\n{'=' * 72}\n{label}\nCMD: {' '.join(cmd)}\n{'=' * 72}\n")
        print(f"\n{label}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(ROOT),
        )
        output = result.stdout + result.stderr
        f.write(output)
        f.write(f"\n[EXIT CODE: {result.returncode}]\n")
        print(output[:500])  # preview
        print(f"[exit {result.returncode}]")

print(f"\nFull results written to: {OUT}")