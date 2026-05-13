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
from app.retriever import retrieve, embed_query, _get_cached_cross_encoder, _DEFAULT_CROSS_ENCODER, _embed_sparse_query


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NEO CITY terminal chatbot demo")
    parser.add_argument("--query", help="Customer question to ask (omit to enter chat mode)")
    parser.add_argument("--chat", action="store_true", help="Interactive chat mode (load models once, ask many questions)")
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


def ask(query: str, args) -> None:
    retrieval_result = retrieve(
        query,
        limit=args.limit,
        min_score=args.min_score,
        top_k=args.top_k,
    )

    if not args.verbose:
        print(chatbot_answer_from_retrieval(retrieval_result))
        return

    demo_answer = chatbot_answer_from_retrieval(retrieval_result)
    answer_result = answer_from_retrieval(retrieval_result)

    print(f"intent: {retrieval_result.get('intent', '')}  |  risk: {retrieval_result.get('risk_level', '')}")
    print(f"sections: {retrieval_result.get('target_sections', [])}  |  mode: {answer_result.answer_mode}")
    print()
    print("A:")
    print(demo_answer)

    if args.show_chunks:
        print()
        for index, chunk in enumerate(retrieval_result.get("chunks", []), start=1):
            print(
                f"  [{index}] id={chunk.get('id', '')} "
                f"section={chunk.get('section', '')} "
                f"topic={chunk.get('topic', '')} "
                f"score={chunk.get('score', 0.0):.4f} "
                f"rerank={chunk.get('rerank_score', 0.0):.4f}"
            )


def chat_loop(args) -> None:
    from app.config import get_settings
    print("NEO CITY Chatbot — models loading, please wait...")
    settings = get_settings()
    embed_query("khởi động", settings.embedding_model)   # pre-load e5-large
    _get_cached_cross_encoder(_DEFAULT_CROSS_ENCODER)    # pre-load cross-encoder
    _embed_sparse_query("khởi động")                     # pre-load BM25
    print("Models loaded. Type your question (or 'exit' to quit).\n")
    while True:
        try:
            query = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", "thoat"}:
            print("Bye!")
            break
        print()
        ask(query, args)
        print()


def main() -> None:
    args = build_parser().parse_args()

    if args.chat or not args.query:
        chat_loop(args)
        return

    ask(args.query, args)


if __name__ == "__main__":
    main()
