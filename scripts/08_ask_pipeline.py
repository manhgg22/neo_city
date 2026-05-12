from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.answer import answer_from_retrieval, chatbot_answer_from_retrieval
from app.retriever import retrieve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NEO CITY terminal chatbot demo")
    parser.add_argument("--query", required=True, help="Customer question to ask")
    parser.add_argument("--limit", type=int, default=20, help="Retriever fetch limit")
    parser.add_argument("--min-score", type=float, default=0.15, help="Retriever min score")
    parser.add_argument("--top-k", type=int, default=5, help="Chunks kept after reranking")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show metadata together with the concise chatbot answer",
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="With --verbose, also print retrieved chunk ids/topics/scores",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    retrieval_result = retrieve(
        args.query,
        limit=args.limit,
        min_score=args.min_score,
        top_k=args.top_k,
    )

    if not args.verbose:
        print(chatbot_answer_from_retrieval(retrieval_result))
        return

    demo_answer = chatbot_answer_from_retrieval(retrieval_result)
    answer_result = answer_from_retrieval(retrieval_result)

    print(f"Q: {args.query}")
    print(f"intent: {retrieval_result.get('intent', '')}")
    print(f"risk: {retrieval_result.get('risk_level', '')}")
    print(f"target_sections: {retrieval_result.get('target_sections', [])}")
    print(f"mode: {answer_result.answer_mode}")
    print(f"used_sections: {answer_result.used_sections}")
    print(f"confidence: {answer_result.confidence}")
    print()
    print("A:")
    print(demo_answer)

    if args.show_chunks:
        print()
        print("chunks:")
        for index, chunk in enumerate(retrieval_result.get("chunks", []), start=1):
            print(
                f"  [{index}] id={chunk.get('id', '')} "
                f"section={chunk.get('section', '')} "
                f"topic={chunk.get('topic', '')} "
                f"score={chunk.get('score', 0.0):.4f} "
                f"rerank={chunk.get('rerank_score', 0.0):.4f}"
            )


if __name__ == "__main__":
    main()
