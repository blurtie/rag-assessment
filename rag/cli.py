"""Command-line entry point.

    python -m rag.cli ingest                  PDFs -> data/index/chunks.jsonl
    python -m rag.cli index                   chunks -> embeddings
    python -m rag.cli search "question"       retrieval only, no LLM
    python -m rag.cli ask "question" --save   retrieval + answer, saved to outputs/
    python -m rag.cli ask "question" --closed-book   LLM only, no retrieval (baseline)

`search` exists on purpose: it lets you judge retrieval separately from generation.
"""
import argparse
import json
import re
import time

from rag.config import load_config, resolve
from rag.generate import answer, answer_closed_book, format_pages


def print_hits(hits: list[dict], preview_chars: int = 300) -> None:
    for h in hits:
        c = h["chunk"]
        ranks = " ".join(f"{k}={h[k]}" for k in ("dense_rank", "bm25_rank") if k in h)
        print(f"\n#{h['rank']} [{c['chunk_id']}] {c['doc_id']} {format_pages(c['pages'])} "
              f"| {c['heading'] or '-'} | score={h['score']:.4f} {ranks}")
        print("   " + c["text"][:preview_chars].replace("\n", " ") + " ...")


def save_run(cfg: dict, record: dict) -> None:
    out_dir = resolve(cfg["paths"]["outputs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"\W+", "_", record["question"].lower())[:40].strip("_")
    path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{slug}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest")
    sub.add_parser("index")
    for name in ("search", "ask"):
        p = sub.add_parser(name)
        p.add_argument("question")
        p.add_argument("--method", choices=["dense", "bm25", "hybrid"])
        p.add_argument("--k", type=int, help="number of chunks (default: config top_k)")
        if name == "ask":
            p.add_argument("--save", action="store_true")
            p.add_argument("--closed-book", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "ingest":
        from rag.ingest import ingest
        ingest(cfg)
        return
    if args.command == "index":
        from rag.index import build_index
        build_index(cfg)
        return

    record = {"question": args.question, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
              "llm": cfg["generation"]["model"], "embedding": cfg["embedding"]["model"]}

    if args.command == "ask" and args.closed_book:
        result = answer_closed_book(cfg, args.question)
        print(f"\n=== CLOSED-BOOK ANSWER ({result['seconds']}s) ===\n{result['content']}")
        record.update({"mode": "closed_book", **result})
        if args.save:
            save_run(cfg, record)
        return

    from rag.retrieve import Retriever
    retriever = Retriever(cfg)
    method = args.method or cfg["retrieval"]["method"]
    hits = retriever.search(args.question, method=method, top_k=args.k)
    print(f"=== RETRIEVED ({method}, top {len(hits)}) ===")
    print_hits(hits)
    record.update({"mode": "rag", "method": method, "retrieved": hits})

    if args.command == "ask":
        result = answer(cfg, args.question, hits)
        print(f"\n=== ANSWER ({result['seconds']}s) ===\n{result['content']}")
        print("\n=== SOURCES ===")
        for s in result["sources"]:
            print(f"[{s['label']}] {s['doc_id']} {format_pages(s['pages'])} | "
                  f"{s['section'] or '-'} | {s['chunk_id']}")
        if not result["sources"]:
            print("(no valid citations in the answer)")
        if result["invalid_citations"]:
            print(f"[warn] answer cites passages that were not provided: {result['invalid_citations']}")
        record.update(result)

    if getattr(args, "save", False):
        save_run(cfg, record)


if __name__ == "__main__":
    main()
