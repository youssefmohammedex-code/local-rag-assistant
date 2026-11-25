"""Benchmark retrieval (and optionally answer quality) across chunking configurations.

Usage:
    python -m eval.run_eval                       # retrieval metrics only (fast)
    python -m eval.run_eval --judge               # also LLM-judged faithfulness/correctness
    python -m eval.run_eval --sizes 300 600 1000 --questions eval/questions.jsonl
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import statistics
import time
from pathlib import Path

from rag.config import ROOT, Settings
from rag.pipeline import RAGPipeline
from rag.store import Hit

RESULTS_DIR = ROOT / "eval" / "results"

JUDGE_PROMPT = """You are grading a question-answering system. Reply with JSON only:
{{"faithful": true/false, "correct": true/false}}

"faithful" = every claim in the ANSWER is supported by the CONTEXT.
"correct" = the ANSWER agrees with the REFERENCE (if the reference is empty, use your best judgement).

CONTEXT:
{context}

QUESTION: {question}
REFERENCE: {reference}
ANSWER: {answer}
"""


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rank_of_expected(hits: list[Hit], expected_source: str, expected_page: int | None) -> int | None:
    """1-based rank of the first hit from the expected document (and page, if given)."""
    for rank, hit in enumerate(hits, start=1):
        if hit.source.endswith(expected_source) and (expected_page is None or hit.page == expected_page):
            return rank
    return None


def judge(pipeline: RAGPipeline, question: str, reference: str, answer: str, hits: list[Hit]) -> dict:
    context = "\n\n".join(h.text for h in hits)
    response = pipeline._llm.chat(
        model=pipeline.settings.llm_model,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            context=context, question=question, reference=reference, answer=answer)}],
        format="json",
        options={"temperature": 0},
    )
    try:
        verdict = json.loads(response["message"]["content"])
        return {"faithful": bool(verdict.get("faithful")), "correct": bool(verdict.get("correct"))}
    except (json.JSONDecodeError, AttributeError):
        return {"faithful": False, "correct": False}


def evaluate_config(settings: Settings, questions: list[dict], k: int, use_judge: bool, reingest: bool) -> dict:
    pipeline = RAGPipeline(settings)
    if reingest or pipeline.store.count() == 0:
        n_chunks = pipeline.ingest(reset=True)
    else:
        n_chunks = pipeline.store.count()

    ranks, latencies, faithful, correct = [], [], [], []
    for q in questions:
        start = time.perf_counter()
        hits = pipeline.retrieve(q["question"], k)
        latencies.append(time.perf_counter() - start)
        ranks.append(rank_of_expected(hits, q["expected_source"], q.get("expected_page")))
        if use_judge:
            answer = pipeline.generate(q["question"], hits)
            verdict = judge(pipeline, q["question"], q.get("reference_answer", ""), answer, hits)
            faithful.append(verdict["faithful"])
            correct.append(verdict["correct"])

    n = len(questions)
    row = {
        "chunk_size": settings.chunk_size,
        "overlap": settings.chunk_overlap,
        "chunks": n_chunks,
        "hit@1": sum(r == 1 for r in ranks) / n,
        f"hit@{k}": sum(r is not None for r in ranks) / n,
        "MRR": sum(1 / r for r in ranks if r) / n,
        "retrieval_ms": 1000 * statistics.mean(latencies),
    }
    if use_judge:
        row["faithfulness"] = sum(faithful) / n
        row["correctness"] = sum(correct) / n
    return row


def write_results(rows: list[dict], meta: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    header = "| " + " | ".join(rows[0].keys()) + " |"
    sep = "|" + "---|" * len(rows[0])
    body = ["| " + " | ".join(fmt(v) for v in r.values()) + " |" for r in rows]
    (RESULTS_DIR / "results.md").write_text("\n".join([meta, "", header, sep, *body, ""]), encoding="utf-8")
    print("\n".join([header, sep, *body]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.jsonl")
    parser.add_argument("--sizes", type=int, nargs="+", default=[300, 600, 1000])
    parser.add_argument("--overlap-ratio", type=float, default=0.15)
    parser.add_argument("-k", type=int, default=4)
    parser.add_argument("--judge", action="store_true", help="also grade generated answers with the LLM")
    parser.add_argument("--reingest", action="store_true", help="rebuild every index even if it exists")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    base = Settings()
    rows = []
    for size in args.sizes:
        settings = dataclasses.replace(base, chunk_size=size, chunk_overlap=int(size * args.overlap_ratio))
        print(f"Evaluating chunk_size={size} ...")
        rows.append(evaluate_config(settings, questions, args.k, args.judge, args.reingest))

    meta = (f"Questions: {len(questions)} · embed: `{base.embed_model}` · llm: `{base.llm_model}` · "
            f"k={args.k} · judge={'on' if args.judge else 'off'}")
    write_results(rows, meta)


if __name__ == "__main__":
    main()
