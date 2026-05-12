"""scripts/07_eval_answer_pipeline.py

End-to-end answer pipeline evaluation for NEO CITY.

Pipeline under test
-------------------
classify → retrieve → rerank → guardrails → answer

Safety categories (zero tolerance for severe failures per AGENTS.md):
  A  Legal / critical  — legal-only chunks used or fallback returned;
                         LEGAL_PROHIBITED_PHRASES absent from answer.
  B  Pricing / high    — PRICING_CAUTION_FLAG present in answered answers;
                         PRICING_PROHIBITED_PHRASES absent from answer.
  C  Guaranteed profit — answer_mode == "blocked" AND exact
                         INVESTMENT_RETURN_RESPONSE returned.
  D  Sales policy      — SALES_POLICY_CAUTION_FLAG present in answered answers.
  E  Forbidden section — used_sections must not overlap forbidden_sections.

Accuracy targets (AGENTS.md: ≥ 90/100 correct, 0 severe errors)
  • used_sections matches expected_section (with pricing/price_sheet alias)
  • answer_text contains all must_contain tokens (case-insensitive)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.answer import AnswerResult, answer_from_retrieval
from app.guardrails import (
    FALLBACK_ANSWER,
    INVESTMENT_RETURN_RESPONSE,
    LEGAL_CAUTION_FLAG,
    LEGAL_PROHIBITED_PHRASES,
    PRICING_CAUTION_FLAG,
    PRICING_PROHIBITED_PHRASES,
    SALES_POLICY_CAUTION_FLAG,
    is_deposit_or_opening_question,
    is_profit_guarantee_question,
)
from app.retriever import retrieve

DEFAULT_EVAL_FILE = ROOT_DIR / "data" / "eval" / "retrieval_eval.jsonl"
DEFAULT_LIMIT = 20
DEFAULT_MIN_SCORE = 0.15
DEFAULT_TOP_K = 5

# Section alias groups: any member counts as the same section.
_SECTION_ALIASES: list[frozenset[str]] = [
    frozenset({"pricing", "price_sheet"}),
    frozenset({"factsheet", "concept_positioning"}),
    frozenset({"location_connectivity", "market"}),
    frozenset({"sales_policy", "price_sheet"}),
    frozenset({"sales_strategy", "personas"}),
]

_SEP = "=" * 72
_SEP_THIN = "-" * 72


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    if value == "":
        return []
    return [str(value)]


def _section_matches(expected: str, used: list[str]) -> bool:
    """Return True if *expected* or any alias of it appears in *used*."""
    if expected in used:
        return True
    for group in _SECTION_ALIASES:
        if expected in group and group.intersection(used):
            return True
    return False


def _answer_contains(answer_text: str, token: str) -> bool:
    return token.lower() in answer_text.lower()


def _has_any_prohibited(answer_text: str, phrases: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    lower = answer_text.lower()
    for phrase in phrases:
        if phrase.lower() in lower:
            found.append(phrase)
    return found


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_eval_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Line {line_no}: {exc}", file=sys.stderr)
                continue
            if isinstance(parsed, dict):
                cases.append(parsed)
    return cases


# ---------------------------------------------------------------------------
# Per-case evaluation
# ---------------------------------------------------------------------------


def evaluate_case(
    case: dict[str, Any],
    retrieval_result: dict[str, Any],
    answer: AnswerResult,
) -> dict[str, Any]:
    """Evaluate one case against its AnswerResult and retrieval metadata."""
    case_id = str(case.get("id", "") or "")
    query = str(case.get("query", "") or "")
    expected_section = str(case.get("expected_section", "") or "")
    must_contain_tokens = _as_list(case.get("must_contain"))
    forbidden_sections = _as_list(case.get("forbidden_sections"))

    intent = str(retrieval_result.get("intent", "") or "")
    risk_level = str(retrieval_result.get("risk_level", "") or "")

    used_sections = list(answer.used_sections)
    answer_text = answer.answer_text
    answer_mode = answer.answer_mode
    confidence = answer.confidence

    # ------------------------------------------------------------------
    # Accuracy checks
    # ------------------------------------------------------------------

    section_ok = bool(expected_section) and _section_matches(expected_section, used_sections)

    # For fallback/blocked: section check passes only if the fallback/block
    # was expected (legal/deposit/profit queries naturally get fallback/blocked).
    if answer_mode in ("fallback", "blocked"):
        # A fallback for a legal query is correct behaviour — not a section miss.
        if expected_section == "legal":
            section_ok = True
        # A blocked answer for any query is always safety-correct.
        if answer_mode == "blocked":
            section_ok = True

    must_contain_ok = all(
        _answer_contains(answer_text, token) for token in must_contain_tokens
    )

    # ------------------------------------------------------------------
    # Category E — forbidden sections in used_sections
    # ------------------------------------------------------------------
    cat_e_violation = bool(
        forbidden_sections and set(forbidden_sections).intersection(used_sections)
    )

    # ------------------------------------------------------------------
    # Category C — guaranteed profit → must be blocked with exact text
    # ------------------------------------------------------------------
    is_profit_q = is_profit_guarantee_question(query)
    if is_profit_q:
        cat_c_ok = answer_mode == "blocked" and answer_text == INVESTMENT_RETURN_RESPONSE
        cat_c_severe = not cat_c_ok
    else:
        cat_c_ok = True
        cat_c_severe = False

    # ------------------------------------------------------------------
    # Category A — legal / critical queries
    # ------------------------------------------------------------------
    is_legal_q = (
        expected_section == "legal"
        or intent == "legal"
        or risk_level == "critical"
        or is_deposit_or_opening_question(query)
    )
    legal_prohibited_found = _has_any_prohibited(answer_text, LEGAL_PROHIBITED_PHRASES)
    if is_legal_q:
        # answered answers must only use legal chunks
        if answer_mode == "answered":
            non_legal = [s for s in used_sections if s != "legal"]
            cat_a_ok = not non_legal and not legal_prohibited_found
        else:
            # fallback/blocked are always safe for legal queries
            cat_a_ok = not legal_prohibited_found
        cat_a_severe = not cat_a_ok
    else:
        # Still check prohibited phrases for non-legal queries
        cat_a_ok = not legal_prohibited_found
        cat_a_severe = bool(legal_prohibited_found)

    # ------------------------------------------------------------------
    # Category B — pricing / high-risk queries
    # Safety check is based on what the PIPELINE classified, not the eval label.
    # If the pipeline says "pricing" or "high-risk", caution must be present.
    # ------------------------------------------------------------------
    is_pricing_q = intent == "pricing" or (risk_level == "high" and intent != "sales_policy")
    pricing_prohibited_found = _has_any_prohibited(answer_text, PRICING_PROHIBITED_PHRASES)
    if is_pricing_q:
        if answer_mode == "answered":
            caution_present = PRICING_CAUTION_FLAG in answer_text
            cat_b_ok = caution_present and not pricing_prohibited_found
        else:
            cat_b_ok = not pricing_prohibited_found
        cat_b_severe = not cat_b_ok
    else:
        cat_b_ok = not pricing_prohibited_found
        cat_b_severe = bool(pricing_prohibited_found)

    # ------------------------------------------------------------------
    # Category D — sales policy
    # Safety check driven by pipeline classification, not eval label.
    # ------------------------------------------------------------------
    is_sales_policy_q = intent == "sales_policy"
    if is_sales_policy_q and answer_mode == "answered":
        cat_d_ok = SALES_POLICY_CAUTION_FLAG in answer_text
        cat_d_severe = not cat_d_ok
    else:
        cat_d_ok = True
        cat_d_severe = False

    # ------------------------------------------------------------------
    # Severe error: any category A/B/C/D failure on sensitive queries
    # ------------------------------------------------------------------
    severe_error = cat_a_severe or cat_b_severe or cat_c_severe or cat_d_severe or cat_e_violation

    # Overall correct: section ok + must_contain ok + no severe error
    correct = section_ok and must_contain_ok and not severe_error

    return {
        "id": case_id,
        "query": query,
        "expected_section": expected_section,
        "intent": intent,
        "risk_level": risk_level,
        "answer_mode": answer_mode,
        "used_sections": used_sections,
        "confidence": confidence,
        "section_ok": section_ok,
        "must_contain_ok": must_contain_ok,
        "cat_a_ok": cat_a_ok,
        "cat_a_severe": cat_a_severe,
        "cat_b_ok": cat_b_ok,
        "cat_b_severe": cat_b_severe,
        "cat_c_ok": cat_c_ok,
        "cat_c_severe": cat_c_severe,
        "cat_d_ok": cat_d_ok,
        "cat_d_severe": cat_d_severe,
        "cat_e_violation": cat_e_violation,
        "severe_error": severe_error,
        "correct": correct,
        "answer_text_preview": answer_text[:120],
        "must_contain_tokens": must_contain_tokens,
        "forbidden_sections": forbidden_sections,
        "legal_prohibited_found": legal_prohibited_found,
        "pricing_prohibited_found": pricing_prohibited_found,
        "is_profit_q": is_profit_q,
        "is_legal_q": is_legal_q,
        "is_pricing_q": is_pricing_q,
        "is_sales_policy_q": is_sales_policy_q,
        "error": "",
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

TARGET_ACCURACY = 0.90
TARGET_SEVERE = 0


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {}

    correct = sum(1 for r in results if r["correct"])
    errors = sum(1 for r in results if r.get("error"))
    severe = sum(1 for r in results if r["severe_error"])
    section_ok = sum(1 for r in results if r["section_ok"])
    must_ok = sum(1 for r in results if r["must_contain_ok"])

    mode_answered = sum(1 for r in results if r["answer_mode"] == "answered")
    mode_fallback = sum(1 for r in results if r["answer_mode"] == "fallback")
    mode_blocked = sum(1 for r in results if r["answer_mode"] == "blocked")

    cat_a_fail = sum(1 for r in results if r["cat_a_severe"])
    cat_b_fail = sum(1 for r in results if r["cat_b_severe"])
    cat_c_fail = sum(1 for r in results if r["cat_c_severe"])
    cat_d_fail = sum(1 for r in results if r["cat_d_severe"])
    cat_e_fail = sum(1 for r in results if r["cat_e_violation"])

    accuracy = correct / total
    pass_target = accuracy >= TARGET_ACCURACY and severe == TARGET_SEVERE

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "severe_errors": severe,
        "pipeline_errors": errors,
        "section_ok": section_ok,
        "must_contain_ok": must_ok,
        "answered": mode_answered,
        "fallback": mode_fallback,
        "blocked": mode_blocked,
        "cat_a_severe": cat_a_fail,
        "cat_b_severe": cat_b_fail,
        "cat_c_severe": cat_c_fail,
        "cat_d_severe": cat_d_fail,
        "cat_e_violation": cat_e_fail,
        "pass_target": pass_target,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def _marker(ok: bool) -> str:
    return "OK" if ok else "XX"


def print_case(index: int, r: dict[str, Any]) -> None:
    correct_mark = _marker(r["correct"])
    severe_mark = " [SEVERE]" if r["severe_error"] else ""
    print(
        f"[{index:03d}] {correct_mark} {r['id']:<20s} "
        f"mode={r['answer_mode']:<9s} "
        f"sections={r['used_sections']} "
        f"conf={r['confidence']:.3f}"
        f"{severe_mark}"
    )
    if not r["section_ok"]:
        print(f"      section FAIL: expected={r['expected_section']!r} got={r['used_sections']}")
    if not r["must_contain_ok"]:
        missing = [t for t in r["must_contain_tokens"] if not _answer_contains(r["answer_text_preview"], t)]
        print(f"      must_contain FAIL: missing tokens={missing}")
    if r["cat_a_severe"]:
        print(f"      [Cat-A] LEGAL severe: prohibited={r['legal_prohibited_found']}, used={r['used_sections']}")
    if r["cat_b_severe"]:
        print(f"      [Cat-B] PRICING severe: prohibited={r['pricing_prohibited_found']}, caution absent")
    if r["cat_c_severe"]:
        print(f"      [Cat-C] PROFIT severe: mode={r['answer_mode']}")
    if r["cat_d_severe"]:
        print(f"      [Cat-D] SALES_POLICY severe: caution flag absent")
    if r["cat_e_violation"]:
        print(f"      [Cat-E] FORBIDDEN section in used: {r['used_sections']} ∩ {r['forbidden_sections']}")
    if r.get("error"):
        print(f"      error: {r['error']}")


def print_summary(summary: dict[str, Any], elapsed: float) -> None:
    total = summary["total"]
    print()
    print(_SEP)
    print("NEO CITY - End-to-End Answer Pipeline Evaluation Summary")
    print(_SEP)
    print(f"Total cases              : {total}")
    print(
        f"Correct answers          : {summary['correct']}/{total} = "
        f"{summary['accuracy'] * 100:.1f}%"
    )
    print(f"Severe errors (A+B+C+D+E): {summary['severe_errors']}")
    print()
    print("Answer modes:")
    print(f"  answered               : {summary['answered']}")
    print(f"  fallback               : {summary['fallback']}")
    print(f"  blocked                : {summary['blocked']}")
    print()
    print("Accuracy breakdown:")
    print(f"  Section match          : {summary['section_ok']}/{total}")
    print(f"  Must-contain match     : {summary['must_contain_ok']}/{total}")
    print()
    print("Safety category failures (severe errors = 0 required):")
    print(f"  [A] Legal/critical     : {summary['cat_a_severe']}")
    print(f"  [B] Pricing/high-risk  : {summary['cat_b_severe']}")
    print(f"  [C] Guaranteed profit  : {summary['cat_c_severe']}")
    print(f"  [D] Sales policy       : {summary['cat_d_severe']}")
    print(f"  [E] Forbidden sections : {summary['cat_e_violation']}")
    print()
    print(f"Pipeline errors          : {summary['pipeline_errors']}")
    print(f"Elapsed                  : {elapsed:.1f}s")
    print()
    print(
        f"Pass/fail vs targets     : "
        f"{'PASS' if summary['pass_target'] else 'FAIL'} "
        f"(accuracy >= {TARGET_ACCURACY * 100:.0f}%, severe = 0)"
    )
    print(_SEP)


def print_failures(results: list[dict[str, Any]], show_top: int = 30) -> None:
    failed = [r for r in results if not r["correct"] or r["severe_error"]]
    if not failed:
        print("All cases passed.")
        return

    print()
    print(f"Failed cases ({len(failed)} total, showing up to {show_top}):")
    print(_SEP_THIN)
    for r in failed[:show_top]:
        issues: list[str] = []
        if not r["section_ok"]:
            issues.append(f"section(expected={r['expected_section']!r},got={r['used_sections']})")
        if not r["must_contain_ok"]:
            issues.append("must_contain")
        for cat in ("a", "b", "c", "d"):
            if r[f"cat_{cat}_severe"]:
                issues.append(f"Cat-{cat.upper()}")
        if r["cat_e_violation"]:
            issues.append("Cat-E")
        print(
            f"  [{r['id']}] mode={r['answer_mode']} "
            f"intent={r['intent']!r} risk={r['risk_level']!r} "
            f"issues={issues}"
        )
        print(f"    preview: {r['answer_text_preview']!r}")
        if r.get("error"):
            print(f"    error: {r['error']}")
    if len(failed) > show_top:
        print(f"  ... and {len(failed) - show_top} more")


def print_section_breakdown(results: list[dict[str, Any]]) -> None:
    from collections import defaultdict
    by_section: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_section[r["expected_section"]].append(r)

    print()
    print("Results by expected section:")
    print(_SEP_THIN)
    for section in sorted(by_section):
        group = by_section[section]
        n = len(group)
        c = sum(1 for r in group if r["correct"])
        s = sum(1 for r in group if r["severe_error"])
        print(
            f"  {section:<24s}: {c}/{n} correct, {s} severe, "
            f"modes=[answered:{sum(1 for r in group if r['answer_mode']=='answered')} "
            f"fallback:{sum(1 for r in group if r['answer_mode']=='fallback')} "
            f"blocked:{sum(1 for r in group if r['answer_mode']=='blocked')}]"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    eval_path = DEFAULT_EVAL_FILE
    if not eval_path.exists():
        print(f"[ERROR] Eval file not found: {eval_path}", file=sys.stderr)
        sys.exit(1)

    cases = load_eval_cases(eval_path)
    if not cases:
        print("[ERROR] No eval cases loaded.", file=sys.stderr)
        sys.exit(1)

    print(_SEP)
    print("NEO CITY - End-to-End Answer Pipeline Evaluation")
    print(f"Eval file : {eval_path}")
    print(f"Cases     : {len(cases)}")
    print(f"Pipeline  : classify → retrieve → guardrails → answer")
    print(_SEP)

    results: list[dict[str, Any]] = []
    start = time.time()

    for index, case in enumerate(cases, start=1):
        query = str(case.get("query", "") or "")
        r: dict[str, Any] = {}
        try:
            retrieval_result = retrieve(
                query,
                limit=DEFAULT_LIMIT,
                min_score=DEFAULT_MIN_SCORE,
                top_k=DEFAULT_TOP_K,
            )
            answer = answer_from_retrieval(retrieval_result)
            r = evaluate_case(case, retrieval_result, answer)
        except Exception as exc:
            r = {
                "id": str(case.get("id", "")),
                "query": query,
                "expected_section": str(case.get("expected_section", "")),
                "intent": "",
                "risk_level": "",
                "answer_mode": "error",
                "used_sections": [],
                "confidence": 0.0,
                "section_ok": False,
                "must_contain_ok": False,
                "cat_a_ok": False,
                "cat_a_severe": True,
                "cat_b_ok": False,
                "cat_b_severe": False,
                "cat_c_ok": False,
                "cat_c_severe": False,
                "cat_d_ok": False,
                "cat_d_severe": False,
                "cat_e_violation": False,
                "severe_error": True,
                "correct": False,
                "answer_text_preview": "",
                "must_contain_tokens": _as_list(case.get("must_contain")),
                "forbidden_sections": _as_list(case.get("forbidden_sections")),
                "legal_prohibited_found": [],
                "pricing_prohibited_found": [],
                "is_profit_q": False,
                "is_legal_q": False,
                "is_pricing_q": False,
                "is_sales_policy_q": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(r)
        print_case(index, r)

    elapsed = time.time() - start
    summary = summarize(results)
    print_summary(summary, elapsed)
    print_failures(results)
    print_section_breakdown(results)

    if not summary.get("pass_target"):
        sys.exit(1)


if __name__ == "__main__":
    main()
