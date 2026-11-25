"""Command line interface: `python -m rag ingest` and `python -m rag ask "..."`."""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time

from rag.config import Settings
from rag.errors import friendly_error
from rag.loaders import pretty_source
from rag.pipeline import MODES, RAGPipeline


def _settings(args: argparse.Namespace) -> Settings:
    overrides = {}
    for name in ("chunk_size", "chunk_overlap", "top_k", "llm_model", "answer_mode"):
        value = getattr(args, name, None)
        if value is not None:
            overrides[name] = value
    return dataclasses.replace(Settings(), **overrides)


def cmd_ingest(args: argparse.Namespace) -> None:
    pipeline = RAGPipeline(_settings(args))
    start = time.perf_counter()
    report = pipeline.ingest(reset=True)
    print(f"Indexed {report.chunks} chunks from {report.files} file(s) into '{pipeline.store.name}' "
          f"in {time.perf_counter() - start:.1f}s")
    for source in report.skipped:
        print(f"  Skipped {source}: no extractable text (empty, unreadable, or a scanned PDF that needs OCR)")


def cmd_ask(args: argparse.Namespace) -> None:
    pipeline = RAGPipeline(_settings(args))
    if pipeline.store.count() == 0:
        sys.exit(f"Collection '{pipeline.store.name}' is empty. Run `python -m rag ingest` first.")
    hits = pipeline.retrieve(args.question)
    for token in pipeline.stream(args.question, hits):
        print(token, end="", flush=True)
    print("\n\nSources:")
    for i, hit in enumerate(hits, start=1):
        where = hit.section.split(" > ")[-1] if hit.section else f"p.{hit.page}"
        print(f"  [{i}] {pretty_source(hit.source)} · {where}  (similarity {hit.score:.2f})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rag", description="Local RAG assistant")
    parser.add_argument("--chunk-size", dest="chunk_size", type=int)
    parser.add_argument("--chunk-overlap", dest="chunk_overlap", type=int)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="index everything in data/docs").set_defaults(func=cmd_ingest)

    ask = sub.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("-k", "--top-k", dest="top_k", type=int)
    ask.add_argument("--model", dest="llm_model")
    ask.add_argument("--mode", dest="answer_mode", choices=sorted(MODES), help="answer length (default: detailed)")
    ask.set_defaults(func=cmd_ask)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:  # show an actionable message instead of a traceback
        sys.exit(f"Error: {friendly_error(exc)}")


if __name__ == "__main__":
    main()
