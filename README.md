# 📚 Local RAG Assistant

Ask questions about your own documents (PDF, Markdown, text) and get cited answers, **100% offline**.
Everything runs on a laptop CPU: local embeddings and a local LLM through [Ollama](https://ollama.com), a [Chroma](https://www.trychroma.com) vector store, and a Streamlit chat UI. No API keys, no cloud and no cost.

The project also includes an **evaluation harness** that measures how chunking choices affect retrieval quality and answer faithfulness, so design decisions are backed by numbers.

<!-- TODO: add a demo GIF here, e.g. docs/demo.gif -->

## Architecture

```
            ┌────────────── ingest ──────────────┐
 data/docs ─► load (pypdf) ─► chunk ─► embed ─► Chroma
                                     (nomic-embed-text)
                                                   │
 question ─► embed query ─► top-k similarity search┘
                                   │
                    numbered context + question
                                   ▼
                     LLM (llama3.2:3b via Ollama)
                                   ▼
                     answer with [n] citations
```

| Component | Choice | Why |
|---|---|---|
| Embeddings | `nomic-embed-text` (137M params) | Strong retrieval quality, fast on CPU, uses task prefixes |
| Vector store | Chroma (persistent, cosine) | Zero setup, stores metadata for citations |
| LLM | `llama3.2:3b` (swappable) | Runs comfortably in 16 GB RAM without a GPU |
| Chunking | Custom boundary-aware sliding window | Cuts at paragraph, sentence or word boundaries, with overlap |
| UI | Streamlit chat with streaming and a source viewer | |

## Quickstart

**1. Install Ollama** from <https://ollama.com/download> and pull the models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

**2. Set up Python (3.10+):**

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

**3. Add documents** to `data/docs/` (a sample handbook is included), then index them:

```bash
python -m rag ingest
```

**4. Ask questions:**

```bash
python -m rag ask "What GPA do I need for the Dean's Honor List?"
streamlit run app.py          # or use the chat UI
```

## Evaluation

`eval/questions.jsonl` holds hand-written questions, each with the document that contains the answer and a reference answer. The harness builds one index per chunk size and reports:

- **hit@1 / hit@k**: how often the correct document is the first result, or appears in the top k
- **MRR**: mean reciprocal rank of the correct document
- **retrieval latency**
- **faithfulness / correctness** (with `--judge`): an LLM-as-judge checks that the answer is supported by the retrieved context and agrees with the reference

```bash
python -m eval.run_eval              # retrieval metrics
python -m eval.run_eval --judge      # plus answer quality (slower)
```

Results are written to `eval/results/results.md` and `results.csv`.

### Results

Run on the included sample documents (12 questions, `llama3.2:3b` as both generator and judge, k=4, laptop CPU: Ryzen 7 4750U, 16 GB RAM, no GPU):

| chunk_size | overlap | chunks | hit@1 | hit@4 | MRR | retrieval_ms | faithfulness | correctness |
|---|---|---|---|---|---|---|---|---|
| 300 | 45 | 30 | 1.000 | 1.000 | 1.000 | 1912 | 0.750 | 0.917 |
| 600 | 90 | 14 | 1.000 | 1.000 | 1.000 | 1902 | 0.917 | 0.917 |
| 1000 | 150 | 8 | 1.000 | 1.000 | 1.000 | 1901 | 1.000 | 1.000 |

**Findings:**
- **Larger chunks gave more faithful answers.** Faithfulness rose from 75% at 300 characters to 100% at 1000. Small chunks often cut a rule away from its conditions, for example the GPA threshold separated from the "no grade below B" clause, so the model filled the gaps itself.
- **Document-level retrieval is saturated.** With only two sample documents, every config found the right document first, so hit@k and MRR don't separate the configs here. A harder benchmark needs more documents and passage-level labels.
- **Retrieval latency (about 1.9 s) is dominated by model swapping.** The eval alternates between the embedding model and the LLM, which Ollama reloads on CPU. In the interactive app, query embedding is much faster.
- **Caveats:** this is a small sample (12 questions, so one question is about 8 points), and the same 3B model generates and judges the answers, so treat these numbers as directional.

**Prompt engineering note:** the first prompt labelled passages as `[1] (source, p. 1)` and ended with `Answer:`. `llama3.2:3b` often replied with only the label `[1] (sample/student-handbook.md, p. 1)`. Wrapping passages in `<passage id=…>` tags and putting explicit answer instructions *after* the question fixed it.

## Configuration

Set these as environment variables, or pass the CLI flags `--chunk-size`, `--chunk-overlap`, `-k` and `--model`:

| Variable | Default |
|---|---|
| `RAG_LLM_MODEL` | `llama3.2:3b` |
| `RAG_EMBED_MODEL` | `nomic-embed-text` |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `1000` / `150` (best config in the eval) |
| `RAG_TOP_K` | `4` |
| `RAG_DOCS_DIR` | `data/docs` |

## Project structure

```
rag/
  config.py      settings (env-overridable)
  loaders.py     PDF / Markdown / text loading
  chunking.py    boundary-aware overlapping chunker
  store.py       Chroma + Ollama embeddings
  pipeline.py    retrieve → prompt → generate
  cli.py         `python -m rag ingest | ask`
app.py           Streamlit chat UI
eval/            question set + benchmarking harness
tests/           unit tests (pytest, no Ollama needed)
```

## Tests

```bash
pytest
```

## Roadmap

- [ ] Hybrid search (BM25 + vectors) and compare against pure vector search
- [ ] Cross-encoder re-ranking of the top-k passages
- [ ] Arabic document support (multilingual embedding model) and evaluation
- [ ] Answer caching and incremental re-indexing
