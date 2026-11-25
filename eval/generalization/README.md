# Generalization test

Checks whether the retrieval design overfits the documents it was developed on. It uses a second
corpus the system was never tuned on: the README/manual files that ship inside installed Python
packages (Markdown, plain text, and a generated multi-page PDF).

Run from the project root:

```bash
# 1. build the corpus from your installed packages (outside the repo)
python eval/generalization/build_corpus.py .venv/Lib/site-packages ../gen/docs

# 2. index it into its own Chroma folder
RAG_DOCS_DIR=../gen/docs RAG_DB_DIR=../gen/db python -m rag ingest

# 3. precise questions (hit@k / MRR) and list questions (coverage)
RAG_DOCS_DIR=../gen/docs RAG_DB_DIR=../gen/db python -m eval.run_eval \
    --questions eval/generalization/q_precise.jsonl --retrieval vector hybrid -k 4
RAG_DOCS_DIR=../gen/docs RAG_DB_DIR=../gen/db python -m eval.run_eval \
    --questions eval/generalization/q_lists.jsonl --expand off on -k 10

# 4. edge-case file audit (23 tricky files: encodings, setext headings, BOM, front matter, ...)
python eval/generalization/audit_files.py ../edge
```

| File | What it is |
|---|---|
| `q_precise.jsonl` | 30 questions written from the docs before any search was run |
| `q_lists.jsonl` | 6 list questions, used while redesigning list expansion |
| `q_lists_heldout.jsonl` | 5 list questions written before the redesign |
| `q_lists_fresh.jsonl` | 3 list questions written last and never used for design decisions |
| `q_ood.jsonl` | 3 questions the docs cannot answer (the model should refuse) |
| `make_pdfs.py` | realistic test PDFs (heading fonts, running header/footer) + one plain single-font PDF |
| `q_pdf_precise.jsonl`, `q_pdf_lists.jsonl` | 17 precise + 2 list questions about those PDFs, written before any search |
| `q_pdf_lists_heldout.jsonl` | 1 more PDF list question, written before trying any fix for PDF lists |
| `write_*.py` | how the question files were generated (expected items come from the real heading structure) |

Package versions differ between machines, so exact numbers may vary slightly. The questions only
use content that has been stable across releases.
