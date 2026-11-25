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


def judge(pipeline: RAGPipeline, question: str, reference: str, answer: str, hits: list[Hit],
          judge_model: str | None = None) -> dict:
    context = "\n\n".join(h.text for h in hits)
    response = pipeline._llm.chat(
        model=judge_model or pipeline.settings.llm_model,
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


def evaluate_config(settings: Settings, questions: list[dict], k: int, use_judge: bool, reingest: bool,
                    judge_model: str | None = None, mode: str = "concise") -> dict:
    pipeline = RAGPipeline(settings)
    if reingest or pipeline.store.count() == 0:
        n_chunks = pipeline.ingest(reset=True).chunks
    else:
        n_chunks = pipeline.store.count()

    ranks, latencies, faithful, correct, gen_seconds, log = [], [], [], [], [], []
    for q in questions:
        start = time.perf_counter()
        hits = pipeline.retrieve(q["question"], k)
        latencies.append(time.perf_counter() - start)
        ranks.append(rank_of_expected(hits, q["expected_source"], q.get("expected_page")))
        if use_judge:
            start = time.perf_counter()
            answer = pipeline.generate(q["question"], hits, mode=mode)
            gen_seconds.append(time.perf_counter() - start)
            verdict = judge(pipeline, q["question"], q.get("reference_answer", ""), answer, hits, judge_model)
            faithful.append(verdict["faithful"])
            correct.append(verdict["correct"])
            log.append({"llm": settings.llm_model, "chunk_size": settings.chunk_size,
                        "question": q["question"], "reference": q.get("reference_answer", ""),
                        "answer": answer, **verdict, "seconds": round(gen_seconds[-1], 1)})

    n = len(questions)
    row = {
        "llm": settings.llm_model,
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
        row["answer_s"] = statistics.mean(gen_seconds)
    row["_log"] = log
    return row


def write_results(rows: list[dict], meta: str, name: str = "results") -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Every generated answer with its verdict, so the judge can be checked by hand.
    answers = [entry for r in rows for entry in r.pop("_log", [])]
    if answers:
        with (RESULTS_DIR / f"{name}_answers.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(a, ensure_ascii=False) + "\n" for a in answers)
    with (RESULTS_DIR / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    header = "| " + " | ".join(rows[0].keys()) + " |"
    sep = "|" + "---|" * len(rows[0])
    body = ["| " + " | ".join(fmt(v) for v in r.values()) + " |" for r in rows]
    (RESULTS_DIR / f"{name}.md").write_text("\n".join([meta, "", header, sep, *body, ""]), encoding="utf-8")
    print("\n".join([header, sep, *body]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.jsonl")
    parser.add_argument("--sizes", type=int, nargs="+", default=[300, 600, 1000])
    parser.add_argument("--overlap-ratio", type=float, default=0.15)
    parser.add_argument("-k", type=int, default=4)
    parser.add_argument("--judge", action="store_true", help="also grade generated answers with the LLM")
    parser.add_argument("--reingest", action="store_true", help="rebuild every index even if it exists")
    parser.add_argument("--models", nargs="+", help="compare these LLMs (default: the configured one)")
    parser.add_argument("--judge-model", help="grade every model with this one LLM, for a fair comparison")
    parser.add_argument("--mode", default="concise", choices=["concise", "detailed"], help="answer mode to evaluate")
    parser.add_argument("--out", default="results", help="output file name (without extension)")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    base = Settings()
    rows = []
    for model in args.models or [base.llm_model]:
        for size in args.sizes:
            settings = dataclasses.replace(base, llm_model=model, chunk_size=size,
                                           chunk_overlap=int(size * args.overlap_ratio))
            print(f"Evaluating llm={model} chunk_size={size} ...", flush=True)
            rows.append(evaluate_config(settings, questions, args.k, args.judge, args.reingest,
                                        args.judge_model, args.mode))

    judge_desc = f"on ({args.judge_model or 'self-graded'})" if args.judge else "off"
    meta = (f"Questions: {len(questions)} · embed: `{base.embed_model}` · k={args.k} · "
            f"mode={args.mode} · judge={judge_desc}")
    write_results(rows, meta, args.out)


if __name__ == "__main__":
    main()
