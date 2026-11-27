"""End-to-end benchmark of list answers: retrieve, generate (with filler repair) and score.

For each list question: coverage (share of the expected items named in the answer), headings,
filler sections left, citation check, time. Defaults are reproducible: temperature 0 and a fixed
seed, so a rerun on the same machine and Ollama version gives the same answers (check it with
--runs 2). Use --temperature 0.1 --seed none --runs 3 to measure run-to-run spread at the app's
real setting instead.

Run once per document set, pointing at its docs and index (see eval/generalization/README.md):

    python -m eval.list_answers --docs ../gen/docs --db ../gen/db --out list_answers_gen \
        --questions eval/generalization/q_lists.jsonl eval/generalization/q_lists_fresh.jsonl \
                    eval/generalization/q_lists_heldout.jsonl
    python -m eval.list_answers --docs ../pdf/docs --db ../pdf/db --out list_answers_pdf \
        --questions eval/generalization/q_pdf_lists.jsonl eval/generalization/q_pdf_lists_heldout.jsonl
    python -m eval.list_answers --docs data/docs/sample --db data/chroma_eval/sample --out list_answers_sample \
        --questions eval/questions_broad.jsonl

Only the questions in SUITE are run unless --all is given. Results go to eval/results/<out>.md
(summary) and <out>_answers.jsonl (every answer).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import statistics
import time
from pathlib import Path

from rag import RAGPipeline, Settings
from rag.filler import filler_headings

ROOT = Path(__file__).resolve().parents[1]

# The list questions used to design and check list answers: 5 design + 7 held out (README,
# "Long list answers"). A sixth design question is about private documents and isn't in the repo.
SUITE = {
    "design": [
        "Summarize each section of the Rich README.",
        "Describe each constraint type that annotated-types provides.",
        "What are all the questions in orjson's FAQ?",
        "List all the other use cases python-dotenv describes.",
        "List every section of the student handbook.",
    ],
    "heldout": [
        "List all the types orjson can serialize natively.",
        "List all the option flags orjson supports.",
        "Walk me through each part of the Typer example upgrade.",
        "List all the sections of the Typer README.",
        "Describe each top-level section of the charset-normalizer documentation.",
        "List every 2.14 release of pydantic.",
        "What are all the sections of the httpcore guide?",
    ],
}


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def coverage(answer: str, items: list[str]) -> float:
    text = norm(answer)
    return sum(norm(item) in text for item in items) / len(items) if items else 0.0


def load_questions(files: list[Path], everything: bool) -> list[dict]:
    wanted = {q: split for split, qs in SUITE.items() for q in qs}
    rows = []
    for path in files:
        for line in path.read_text(encoding="utf-8-sig").splitlines():  # -sig: files saved with a BOM
            if not line.strip():
                continue
            row = json.loads(line)
            if everything or row["question"] in wanted:
                rows.append({**row, "split": wanted.get(row["question"], "other")})
    return rows


def answer_once(pipeline: RAGPipeline, question: str, items: list[str]) -> dict:
    start = time.perf_counter()
    hits = pipeline.retrieve(question)
    answer = pipeline.generate(question, hits)
    checks = pipeline.check_citations(answer, hits)
    return {"answer": answer, "coverage": coverage(answer, items), "seconds": round(time.perf_counter() - start),
            "headings": len(re.findall(r"(?m)^#{1,6} ", answer)), "filler_left": len(filler_headings(answer)),
            "claims": len(checks), "supported": sum(c.status == "supported" for c in checks)}


def summarize(results: list[dict], settings: Settings, runs: int) -> str:
    lines = [f"Model `{settings.llm_model}` · temperature {settings.temperature} · seed {settings.seed} · "
             f"{runs} run(s) per question", "",
             "| Split | Question | Coverage | Same answer every run | Headings | Filler left | "
             "Citations supported | Seconds |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        covs = [x["coverage"] for x in r["runs"]]
        cov = f"{statistics.mean(covs):.0%}" + (f" ({min(covs):.0%}–{max(covs):.0%})" if runs > 1 else "")
        same = "—" if runs == 1 else ("yes" if len({x["answer"] for x in r["runs"]}) == 1 else "**no**")
        first = r["runs"][0]
        lines.append(f"| {r['split']} | {r['question']} | {cov} | {same} | {first['headings']} | "
                     f"{first['filler_left']} | {first['supported']}/{first['claims']} | "
                     f"{round(statistics.mean(x['seconds'] for x in r['runs']))} |")
    all_cov = [x["coverage"] for r in results for x in r["runs"]]
    if all_cov:
        lines += ["", f"Mean coverage: {statistics.mean(all_cov):.1%} over {len(results)} question(s)."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, nargs="+", required=True)
    parser.add_argument("--docs", type=Path, help="documents folder (default: RAG_DOCS_DIR / data/docs)")
    parser.add_argument("--db", type=Path, help="index folder (default: RAG_DB_DIR / data/chroma)")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", default="42", help='an integer, or "none" for random sampling')
    parser.add_argument("--cold", action="store_true",
                        help="unload the model after every call (no cached prompts), for exact repeats")
    parser.add_argument("--all", action="store_true", help="run every question in the files, not just SUITE")
    parser.add_argument("--out", default="list_answers", help="output name in eval/results (no extension)")
    args = parser.parse_args(argv)

    settings = dataclasses.replace(Settings(), temperature=args.temperature, cold_calls=args.cold,
                                   seed=None if args.seed.lower() == "none" else int(args.seed))
    if args.docs:
        settings = dataclasses.replace(settings, docs_dir=args.docs)
    if args.db:
        settings = dataclasses.replace(settings, db_dir=args.db)
    pipeline = RAGPipeline(settings)
    if pipeline.store.count() == 0:
        print(f"Indexing {settings.docs_dir} into {settings.db_dir} ...", flush=True)
        pipeline.ingest()

    questions = load_questions(args.questions, args.all)
    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with (out_dir / f"{args.out}_answers.jsonl").open("w", encoding="utf-8") as answers:
        for q in questions:
            runs = []
            for run in range(args.runs):
                result = answer_once(pipeline, q["question"], q.get("expected_items", []))
                runs.append(result)
                answers.write(json.dumps({"question": q["question"], "run": run, **result}, ensure_ascii=False) + "\n")
                answers.flush()
                print(f"{q['split']:<8} run {run + 1}/{args.runs} coverage {result['coverage']:4.0%} "
                      f"filler {result['filler_left']} citations {result['supported']}/{result['claims']} "
                      f"{result['seconds']:4}s | {q['question'][:60]}", flush=True)
            results.append({"question": q["question"], "split": q["split"], "runs": runs})
    summary = summarize(results, settings, args.runs)
    (out_dir / f"{args.out}.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    main()
