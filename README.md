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
                     LLM (qwen2.5:7b via Ollama)
                                   ▼
                     answer with [n] citations
```

| Component | Choice | Why |
|---|---|---|
| Embeddings | `nomic-embed-text` (137M params) | Strong retrieval quality, fast on CPU, uses task prefixes |
| Vector store | Chroma (persistent, cosine) | Zero setup, stores metadata for citations |
| Hybrid search | Embeddings + BM25 keyword search, merged with Reciprocal Rank Fusion | Embeddings blur exact IDs ("D13" vs "D21"); keywords catch them. BM25 is written from scratch with identifier boosting and heading boosting |
| LLM | `qwen2.5:7b` (swappable; `llama3.2:3b` for speed) | Most accurate in the model comparison below; fits in 16 GB RAM without a GPU |
| Chunking | Custom boundary-aware sliding window | Cuts at paragraph, sentence or word boundaries, with overlap |
| Markdown structure | Heading-aware sections + contextual headers | Each chunk is prefixed with its heading path (e.g. `Guide > 7. Modules > Module C`), so a passage keeps its meaning after it is cut out |
| Answer modes | **Detailed** (k=10, structured Markdown) / **Concise** (k=4, 1–3 sentences) | Broad questions ("describe every module…") need more context and room to answer |
| Context window | `num_ctx=8192` | Ollama's default is too small for 10 passages plus a long answer |
| UI | Streamlit "DocChat": streaming answers, citation badges, source cards with match scores, document upload, model picker | |

**Example of the impact:** for *"Describe every module, who owns it, and what it is responsible for"* over a 110 KB set of architecture docs, the first version gave a 79-word answer covering 3 of 9 modules. With heading-aware chunks, Detailed mode and the larger context window, it gave 559 words covering all 9 modules and their owners.

## Quickstart

**1. Install Ollama** from <https://ollama.com/download> and pull the models:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:7b        # default (4.7 GB)
ollama pull llama3.2:3b       # optional: ~2.5x faster, less accurate (2.0 GB)
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
python -m rag ask "What GPA do I need for the Dean's Honor List?"            # detailed answer
python -m rag ask --mode concise "How much does late registration cost?"
python -m rag ask --retrieval vector "..."   # compare against meaning-only search
streamlit run app.py          # or use the DocChat UI (upload files, pick model and answer length)
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

### Retrieval comparison: vector-only vs hybrid

This is scored at the **passage level**: a hit only counts if the retrieved passage contains the answer, not just the right document (document-level scores were 100% everywhere). Settings: k=4, 1000-character chunks.

| Question set | Retrieval | hit@1 | hit@4 | MRR | Latency |
|---|---|---|---|---|---|
| Sample docs, 12 questions | vector | 83% | 100% | 0.92 | ~0.2 s |
| Sample docs, 12 questions | hybrid | 83% | 100% | 0.92 | ~0.2 s |
| Private architecture docs, **20 held-out questions** | vector | 55% | 70% | 0.63 | ~0.15 s |
| Private architecture docs, **20 held-out questions** | hybrid | **70%** | **95%** | **0.83** | ~0.15 s |

Broken down on the held-out set:

| Question type | vector hit@4 | hybrid hit@4 | vector MRR | hybrid MRR |
|---|---|---|---|---|
| Asks about an ID (*"What is D19 about?"*), 10 questions | 4/10 | 9–10/10 | 0.30 | 0.78 |
| Plain language (*"How is an incident's status determined?"*), 10 questions | 10/10 | 10/10 | 0.95 | 0.90 |

**Findings:**
- **Embeddings can't tell identifiers apart:** "decision D13" and "decision D21" look almost the same to the embedding model. Vector search never found D13, D16 or D27 in its top 10.
- **Plain BM25 wasn't enough.** In "What did decision D13 *decide*?", the rare word "decide" outweighed "D13". The fix: boost identifier-like query terms (letters + digits), count the section heading twice so the section *named* D13 beats sections that merely mention it, and weight the keyword ranking higher when the question contains an identifier.
- **Honest trade-off:** on plain-language questions, hybrid pushed one answer from rank 1 to 2 (MRR 0.95 → 0.90). It still always lands in the top 4.
- **Avoiding self-deception:** the heuristics above were tuned on 5 queries, then measured on 20 *different* held-out questions. Tuning also exposed a real bug: the candidate pool depended on k, so asking for 4 passages could miss what 10 passages found. A fixed pool of 50 makes the ranking consistent: the top 4 is always the start of the top 10. The cost is one ID question at k=4 (10/10 → 9/10), which I kept rather than tuning on the held-out set.

### Model comparison: `llama3.2:3b` vs `qwen2.5:7b`

Both models were tested with the same retrieval (1000-character chunks, k=4, Concise mode). `qwen2.5:7b` judged every answer, and the private set was also **checked by hand** because a model judging its own answers can be biased. Every answer is saved to `eval/results/*_answers.jsonl` so the grading can be audited.

| Question set | Model | Correct (hand-checked) | Faithfulness (judge) | Avg. answer time |
|---|---|---|---|---|
| Sample docs, 12 questions | llama3.2:3b | 100% (judge) | 91.7% | 11.6 s |
| Sample docs, 12 questions | qwen2.5:7b | 100% (judge) | 100% | 29.3 s |
| Private 110 KB architecture docs, 8 questions | llama3.2:3b | **7 / 8** | 62.5% | 15.1 s |
| Private 110 KB architecture docs, 8 questions | qwen2.5:7b | **8 / 8** | 87.5% | 37.0 s |

**Findings:**
- **On the easy sample set, the models tie on correctness**, so harder, realistic documents were needed to tell them apart.
- **On the real docs, llama3.2:3b named the wrong owner for one module**, even though every retrieved passage named the right person. It also padded another answer with an unrelated name, and once cited a section heading instead of a passage number. qwen2.5:7b answered all 8 correctly, with clean citations.
- **The judge is imperfect:** it marked one of qwen's correct, well-cited answers as unfaithful. That's why the hand check matters.
- **Broad question in Detailed mode:** llama3.2:3b wrote 559 words but mixed data contracts into the module list. qwen2.5:7b wrote a cleaner 360-word answer with the correct owner and responsibilities for 8 of the 9 modules, but missed one module and repeated a generic "interactions" line.
- **Trade-off:** qwen2.5:7b is about 2.5x slower on CPU (about 4 tok/s vs 6–8 tok/s). Correctness matters more for a document assistant, so it is the default. The DocChat model picker switches to llama3.2:3b for speed.

## Configuration

Set these as environment variables, or pass the CLI flags `--chunk-size`, `--chunk-overlap`, `-k` and `--model`:

| Variable | Default |
|---|---|
| `RAG_LLM_MODEL` | `qwen2.5:7b` |
| `RAG_EMBED_MODEL` | `nomic-embed-text` |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `1000` / `150` (best config in the eval) |
| `RAG_ANSWER_MODE` | `detailed` (or `concise`) |
| `RAG_RETRIEVAL` | `hybrid` (or `vector`) |
| `RAG_TOP_K` | per mode: `10` detailed, `4` concise |
| `RAG_NUM_CTX` | `8192` |
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

## Reliability

- **Re-indexing is safe:** all embeddings are computed before the old index is replaced. If Ollama is down mid-way, or the docs folder is empty or misconfigured, the existing index is kept. This was a real bug found by testing failure cases: a failed re-index used to wipe the index.
- **Nothing fails silently:** files with no extractable text (empty files, corrupt PDFs, scanned PDFs that need OCR) are reported, and replacing an uploaded file is announced.
- **Actionable errors:** the CLI and the app show messages like *"Can't reach Ollama…"* or *"The model `x` isn't installed. Run `ollama pull x`"* instead of stack traces.

## Tests

```bash
pytest          # 50 tests, no Ollama needed (a fake embedder stands in for it)
```

## Roadmap

- [x] Hybrid search (BM25 + vectors) and compare against pure vector search
- [ ] Cross-encoder re-ranking of the top-k passages
- [ ] Arabic document support (multilingual embedding model) and evaluation
- [ ] Answer caching and incremental re-indexing

## License

[MIT](LICENSE)
