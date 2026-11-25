"""Coverage of every list question set under: off, additive+full-path summaries, additive+heading-only.
Usage: python compare_variants.py <gen-dir>   (run from the project root)"""

import dataclasses
import json
import statistics
import sys
from pathlib import Path

import rag.pipeline as pl
from eval.run_eval import coverage
from rag import RAGPipeline, Settings
from rag.loaders import pretty_source

G = Path(sys.argv[1])
SETS = [  # (label, questions file, docs dir, db dir)
    ("your docs (6)", Path("eval/questions_broad_private.jsonl"), None, None),
    ("your held-out (4)", Path("eval/questions_broad_heldout_private.jsonl"), None, None),
    ("sample (2)", Path("eval/questions_broad.jsonl"), Path("data/docs/sample"), Path("data/chroma_eval")),
    ("new corpus (6)", G / "q_lists.jsonl", G / "docs", G / "db"),
    ("new held-out (5)", G / "q_lists_heldout.jsonl", G / "docs", G / "db"),
    ("new fresh (3)", G / "q_lists_fresh.jsonl", G / "docs", G / "db"),
]
full_path = pl.describe_list


def heading_only(source, parent, kids):
    heading = parent.split(" > ")[-1] if parent else pretty_source(source)
    return f"{heading}: " + "; ".join(name for name, _ in kids[:pl.MAX_NODE_CHILDREN])


variants = {"off": (False, full_path), "additive+full-path": (True, full_path),
            "additive+heading-only": (True, heading_only)}
all_scores = {v: [] for v in variants}
regressions = {v: 0 for v in variants}
print(f"{'set':<20}" + "".join(f"{v:>24}" for v in variants) + f"{'avg chars (full-path)':>24}")
for label, qfile, docs, db in SETS:
    if not qfile.exists():  # private question sets are not in the repository
        print(f"{label:<20} (skipped: {qfile} not found)")
        continue
    base = Settings()
    if docs:
        base = dataclasses.replace(base, docs_dir=docs, db_dir=db)
    questions = [json.loads(line) for line in qfile.open(encoding="utf-8")]
    row, chars = {}, []
    off_scores = None
    for name, (expand, describer) in variants.items():
        pl.describe_list = describer
        p = RAGPipeline(dataclasses.replace(base, expand_lists=expand))
        scores = []
        for q in questions:
            hits = p.retrieve(q["question"], 10)
            scores.append(coverage(hits, q["expected_items"]))
            if name == "additive+full-path":
                chars.append(sum(len(h.text) for h in hits))
        if name == "off":
            off_scores = scores
        else:
            regressions[name] += sum(s < o - 1e-9 for s, o in zip(scores, off_scores, strict=True))
        all_scores[name] += scores
        row[name] = statistics.mean(scores)
    print(f"{label:<20}" + "".join(f"{row[v]:>24.0%}" for v in variants) + f"{statistics.mean(chars):>24,.0f}")
pl.describe_list = full_path
print(f"{'ALL 26 questions':<20}" + "".join(f"{statistics.mean(all_scores[v]):>24.0%}" for v in variants))
print("questions worse than 'off':", {v: regressions[v] for v in variants if v != "off"})
