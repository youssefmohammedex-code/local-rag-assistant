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
| Hybrid search | Embeddings + BM25 keyword search (Snowball-stemmed), merged with Reciprocal Rank Fusion | Embeddings blur exact IDs ("D13" vs "D21"); keywords catch them. BM25 is written from scratch with identifier boosting and heading boosting |
| PDF structure | Headings detected from font size/weight, running headers/footers removed, sections that continue across pages | PDF text has no markup, so without this PDF passages had no headings and were cited only by page |
| Tiny sections | A section under 400 characters ("Requirements: Python 3.8+") borrows the following text for context but keeps its own heading | A 20-character passage is a weak match next to longer passages that mention the same words |
| LLM | `qwen2.5:7b` (swappable; `llama3.2:3b` for speed) | Most accurate in the model comparison below; fits in 16 GB RAM without a GPU |
| Chunking | Custom boundary-aware sliding window | Cuts at paragraph, sentence or word boundaries, with overlap |
| Markdown structure | Heading-aware sections + contextual headers | Each chunk is prefixed with its heading path (e.g. `Guide > 7. Modules > Module C`), so a passage keeps its meaning after it is cut out |
| List questions | Additive section expansion: *"describe every module"* keeps the normal top-k and adds one passage per sub-section, within a ~20k-character budget | Top-k similarity returns the k most similar passages, which cluster on a few items, so it can't answer "every X" |
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
| Private architecture docs, 20 held-out questions | hybrid + exact-code headings (see below) | **90%** | **100%** | **0.95** | ~0.2 s |
| Third-party package docs (Markdown, TXT, PDF), **30 blind questions** | vector | 57% | 80% | 0.67 | ~0.15 s |
| Third-party package docs (Markdown, TXT, PDF), **30 blind questions** | hybrid + exact-code headings | **70%** | **87%** | **0.76** | ~0.2 s |

Broken down on the held-out set:

| Question type | vector hit@4 | hybrid hit@4 | vector MRR | hybrid MRR |
|---|---|---|---|---|
| Asks about an ID (*"What is D19 about?"*), 10 questions | 4/10 | 9–10/10 | 0.30 | 0.78 |
| Plain language (*"How is an incident's status determined?"*), 10 questions | 10/10 | 10/10 | 0.95 | 0.90 |

**Findings:**
- **Embeddings can't tell identifiers apart:** "decision D13" and "decision D21" look almost the same to the embedding model. Vector search never found D13, D16 or D27 in its top 10.
- **Plain BM25 wasn't enough.** In "What did decision D13 *decide*?", the rare word "decide" outweighed "D13". The fix: boost identifier-like query terms (letters + digits), count the section heading twice so the section *named* D13 beats sections that merely mention it, and weight the keyword ranking higher when the question contains an identifier.
- **Honest trade-off:** on plain-language questions, hybrid pushed one answer from rank 1 to 2 (MRR 0.95 → 0.90). It still always lands in the top 4.
- **Exact codes in headings go first.** A later test on third-party docs showed that *"What changed in python-dotenv **0.10.0**?"* still failed. BM25 ranked the 0.10.0 section 1st, but fusion favours passages found by *both* rankings, so 0.10.1, 0.10.2 and other entries pushed it out of the top 10. The identifier rule also only knew letter+digit codes like "D13" and not versions like `0.10.0`. Now any code with digits (letters mixed in, or joined by `.` `-` `:`) counts as exact. A section whose own heading carries the exact code from the question goes first. On the 20 held-out questions this raised hit@1 from 70% to **90%** and hit@4 from 95% to **100%**. On 30 questions about third-party docs, hit@4 went from 83% to 87%.
- **Avoiding self-deception:** the heuristics above were tuned on 5 queries, then measured on 20 *different* held-out questions. Tuning also exposed a real bug: the candidate pool depended on k, so asking for 4 passages could miss what 10 passages found. A fixed pool of 50 makes the ranking consistent: the top 4 is always the start of the top 10. The cost is one ID question at k=4 (10/10 → 9/10), which I kept rather than tuning on the held-out set.

### List questions: section expansion

Questions like *"describe every module"* or *"list all the decisions"* need **every** item, but top-k retrieval returns the k most *similar* passages, which cluster on a few items. With k=10, the retrieved text contained only 22% of the modules and 23% of the decisions.

**How it works (current version):**
1. Detect list questions (*all / every / each / list*).
2. Retrieve 20 candidates. Each one's own section, every ancestor heading, and the file's top level (for documents without a title) count as a possible "list".
3. Summarize each possible list as its **full heading path plus its sub-headings** (*"orjson > Types: dataclass; datetime; enum; …"*). The summary most similar to the question, using the same embedding model as search, is the list. There are no hand-tuned weights.
4. **Additive by design:** keep exactly the normal top-k passages, then add the first passage of every sub-section in document order, trimmed to fit a ~20k-character total budget.
5. **Expand the two best-matching sections, not one** (added later; see "Choosing how many lists to expand" below).

**Coverage** is the share of required items that appear anywhere in the retrieved text (k=10), measured on 26 list questions across 3 document sets:

| Question set | Expansion off | Expansion on |
|---|---|---|
| Private architecture docs: 6 questions | 47% | 87% |
| Private architecture docs: 4 held-out questions | 73% | 92% |
| Sample docs: 2 questions | 94% | 100% |
| Third-party package docs: 6 questions | 59% | 78% |
| Third-party package docs: 5 held-out questions | 79% | 89% |
| Third-party package docs: 3 fresh questions (written last, never used for design) | 67% | 83% |
| **All 26 questions** | **66%** | **87%** |
| Questions worse than with expansion off | — | **0 of 26** |

**How it got here (an overfitting story):**
- **v1 used hand-tuned weights.** It scored 100% on the private docs, including held-out questions, but those questions all came from the same 3 documents. On new third-party documents it made some list questions **worse** than no expansion: held-out coverage fell from 79% to 67%, and the Typer README fell from 57% to 14%. Questions that name a product ("all the types *orjson* can serialize") matched the document title, so the whole document was "the list".
- **v2 chose the section by embedding similarity of "heading: sub-headings" summaries.** It was better on new documents but still picked the whole document when a question named the product, because only the document summary contained that name.
- **v3 adds the full heading path to every summary** (the same idea as the passage headers). It was better on unseen documents and slightly worse on the original private docs. No selection rule won everywhere: **deciding which section a question wants listed is genuinely ambiguous.**
- **The final change made expansion additive.** It never removes a normal result, only adds, so a wrong pick costs context space, not answers. Across all 26 questions, no question scored lower than with expansion off.

**Cost:** list questions send roughly twice the text to the model (about 8–15k characters instead of 5–8k), so on CPU the first word appears later, and complete answers to long lists are long. The earlier v1 answer to *"describe every module…"* covered all 9 modules with the correct owners but took about 10 minutes on CPU.

### Generalization test: documents the system was never tuned on

**Why:** every design choice above was tuned on one set of documents, so the results could be overfitting. To test that, I built a second corpus from documents I didn't write: the README/manual files of 9 installed Python packages.

**The corpus** (~400 KB, 777 passages, 3 file types):
- Markdown: pydantic, python-dotenv, orjson, typer, charset-normalizer, annotated-types
- Plain text: the RST-formatted tqdm manual, and numpy's bundled licenses
- A 14-page PDF made from httpcore's README

**The questions** were written by reading the documents *before* running any search:
- 30 precise questions
- 14 list questions in three batches (design, held-out, and fresh, written last)
- 3 questions the documents can't answer

**What it found:**
- **Overfitting in list expansion:** its hand-tuned v1 made some list questions *worse* on new documents (see above), which led to the additive redesign.
- **Overfitting in the identifier rule:** it only knew "D13"-style codes and missed versions like `0.10.0`.
- **6 file-handling bugs** your own documents never triggered: encoding, setext headings, BOM, front matter, lock files, and unsupported files being dropped silently.

**End-to-end answers** (qwen2.5:7b, Concise mode, hand-graded, 13 answerable and 3 unanswerable questions):

| Result | Count |
|---|---|
| Correct, with citation | **11 / 13** |
| Retrieval missed, model said "I don't know" (no hallucination) | 2 / 13 → 1 after the exact-code fix (the remaining one is a PDF) |
| **Made-up answers** | **0** |
| Unanswerable questions correctly refused | **3 / 3** |

**Known remaining weaknesses:**
- **Vocabulary mismatch can hurt hybrid search:** for "JSON Lines" vs "JSONL", vector-only search found it at rank 2 and hybrid didn't.
- **PDF list questions:** see the PDF section below.

### PDFs: recovering structure

**The problem:** PDF text extraction returns flat lines per page, so PDF passages had no headings and were cited only by page number. On top of that, every page's running header and "Page N" footer added noise. On 17 blind questions over 4 test PDFs, **hybrid search was worse than vector-only** (hit@4 88% vs 94%): the keyword boosts that help Markdown had nothing to work with.

**The test PDFs** are built like real documents: headings in larger bold fonts, body text, code in a monospace font, and a running header and page footer on every page. They're made from 3 package READMEs, plus one **plain single-font PDF** with no styling at all (see `eval/generalization/make_pdfs.py`).

**What changed:**
1. **Headings from fonts.** Each line's font size and weight is read. The body size is learned per document, and lines noticeably larger, or bold at body size, become headings, with bigger fonts at higher levels. Single-font PDFs fall back to numbered headings ("2.1 Setup") and otherwise stay page-based instead of inventing structure.
2. **Running headers and footers are removed:** short lines that repeat at the top or bottom of most pages.
3. **Sections continue across page breaks.** Each page of a section is its own piece, so citations read *"Dependencies, p.2"*.
4. **Tiny sections borrow context.** I tried three options; borrowing context was the only one that didn't hurt Markdown:

   | Option | PDF hit@4 | Third-party Markdown hit@4 | Third-party lists |
   |---|---|---|---|
   | Off | 88% | 87% | 78% |
   | Merge small sections | 94% | **80%** ⬇ | **61%** ⬇ (merged changelog entries lost their headings, breaking exact-version lookup and list expansion) |
   | **Borrow context from the following sections** | 88% | 87% | 81% |
5. **Stemmed keywords** (Snowball). The question "require" didn't match the heading "Requirements".

**Results** (hybrid, k=4; PDFs with all changes):

| | Before | After |
|---|---|---|
| PDF questions found first (hit@1) | 59% | **71%** |
| PDF questions in the top 4 (hit@4) | 88% | **100%** |
| PDF MRR | 0.73 | **0.83** |
| Third-party Markdown hit@4 | 87% | **90%** |
| Your-docs held-out hit@1 | 90% | 85% (one question dropped one rank) |
| PDF list coverage | 76% | 59% at first, then **87%** after expanding the two best lists (see below) |

**End-to-end with qwen2.5:7b** on the PDF questions that failed before:
- *"Which Python version does httpcore require?"* went from "I don't know" to **"HTTP Core requires Python 3.8+"**.
- *"What does pandas aim to become?"* went from "I don't know" to the correct answer, cited as *"What is it?, p.1"*.
- *"What changed in httpcore 1.0.5?"* is now correct, cited as *"1.0.5 (March 27th, 2024), p.3"*.

**The trade-off this introduced, and its fix:** structured PDFs first made PDF list questions *worse* (76% → 59%). A whole PDF page happened to pack in many changelog entries, while per-section passages depend on list expansion choosing the right section. For *"every httpcore release"*, it picked one release's "Added / Fixed" sub-list over the whole changelog, by 0.793 to 0.791 cosine similarity.

### Choosing how many lists to expand

**The near-tie above showed that the section selector is often unsure.** Rather than add a tuned tie-break rule (that's what overfit before), I compared four strategies with no tuned constants, on **29 list questions over 4 document sets** (private, sample, third-party Markdown/TXT and PDFs):

| Strategy | Coverage | Questions worse than top1 | Avg. characters sent |
|---|---|---|---|
| No expansion | 68% | 10 | 7.5k |
| top1: expand the best-matching section (previous default) | 86% | – | 12.2k |
| ancestor: prefer a parent section among the top 3 | 84% | 3 | 12.5k |
| nested-both: expand the best two only if one contains the other | 94% | 0 | 13.5k |
| **top2: expand the two best-matching sections** | **97%** | **0** | 15.5k |

- **PDF list questions:** 58% → **87%**, now *above* the old page-based 76%.
- **Every other list set** also improved: private 87% → 100%, third-party 77% → 100%.
- **Precise questions are unchanged,** because only list questions are routed through expansion.
- **Cost:** about 27% more text sent for list questions, still within the 20k-character budget.
- **End-to-end** (qwen2.5:7b, Detailed mode, hand-graded): for *"Summarize each section of the Rich README"* (a PDF), retrieval delivered all 10 sections. The answer covered 9 of them, each with a citation and page; it skipped "Toad" even though that passage was retrieved. It also attached the wrong code example to 2 sections and listed some sub-features as if they were sections. Retrieval is no longer the bottleneck for this question; the 7B model is.
- **Caveat:** the PDF list evidence is only 3 questions. One more held-out question was planned, but two of the three generated candidates had to be dropped (an empty list, and items that were common words).
- **Questions that don't name their subject are ambiguous:** in "Which command prints the variables in JSON format?", orjson is *about* JSON.

### Long list answers: helping the 7B model not skip items

**Once retrieval delivered every item, the remaining errors came from qwen2.5:7b itself.** Given 10+ sections in one prompt, it skipped some, merged others, and attached examples to the wrong section. I compared three generation strategies end-to-end (Detailed mode), all using the same retrieved passages:

- **single:** one call, as before.
- **checklist:** one call, with the outline of the expanded sections added to the prompt.
- **per-item:** one call per 3 sections, each seeing only its own sections' passages.

Coverage is the share of expected items found in the answer. **Every answer was also read by hand,** because the automatic score is fooled both ways: it gave 44% to a complete answer that dropped the "Module B —" prefix, and 100% to an answer with invented content.

| 6 list questions (used while designing) | Coverage | Total time | Notes |
|---|---|---|---|
| single | 94.7% | 37 min | Rich README: skipped 2 of 10 sections, put the Console example under "Rich Print" |
| checklist, strict ("exactly these headings") | 96.7% | 23 min | padded answers with the unrelated 2nd expanded list (13 headings for an 8-section handbook); orjson FAQ 100% → 80% |
| per-item | 96.7% | 37 min | same padding problem, slowest, choppy answers |
| **checklist, soft (default)** | **96.7%** | **14 min** | exactly the right headings on 5 of 6 questions |

**Held-out check:** the soft wording was revised once after seeing these 6 questions, so it was re-tested on 6 list questions from documents it was never tuned on:

| Held-out list question | single | checklist |
|---|---|---|
| orjson: types it serializes natively | 88% | 100% |
| orjson: option flags | 100% | 100% |
| Typer: sections of the README | 100% | 100% |
| charset-normalizer: top-level sections | 80% | 100% |
| Typer: walk through the example upgrade | partial (content right, sections unnamed) | 67%, padded with unrelated sections |
| httpcore guide (PDF) | wrong (listed changelog keywords) | right headings, **invented content** |
| Time (first 5 questions) | 20 min | **9.5 min** |

- **Wins:** checklist was never worse on coverage, never skipped a section that exists as a heading, and is about **2× faster**, because it doesn't pad.
- **What it doesn't fix:**
  - **Items that live inside a section's text, not as headings,** don't benefit. orjson's 5 FAQ questions scored 100%, 80% and 60% across runs; that's run-to-run variance at temperature 0.1.
  - **An outline can invite confident invention.** On httpcore the model wrote plausible facts with citations that don't support them. The single strategy was also wrong there, just more obviously.
- **Tried and reverted:**
  - A strict "exactly these headings" outline: it forced unrelated sections into the answer.
  - Outlining only the best-matching list: it would break annotated-types, whose constraints are in the second-ranked list.
  - Listing each section's passage ids in the outline: it lowered coverage on 3 of 13 questions and did not stop the httpcore invention.
- **Not counted as a model failure:** "List every 2.14 release of pydantic" scored 0% with both strategies, because only 38% of the releases were retrieved. That was a retrieval gap, fixed below.

### List questions that name a code ("every 2.14 release")

**Why it failed:** the right list was the changelog itself, but its summary was a wall of 100+ version numbers that didn't resemble the question, so three individual 2.14 releases' sub-lists ranked higher. Even when a whole changelog is picked, expansion fills the budget in document order and can run out before older releases.

**The fix reuses the exact-code detection from keyword search:**
- **Filtering:** when a list question contains a code (`2.14`, `v3`, `0.10.0`) and 2+ sub-sections of a section are named by it, that section's candidate list becomes just those sub-sections.
- **Matching at boundaries:** `2.14` matches `v2.14.4` and `v2.14.0b1` but not `v12.14` or `2.140`, and `v2.14.0` doesn't match `v2.14.0b1`.
- **Everything else is unchanged:** questions without a code, and codes that name only one sub-section ("the v2.14.0 release notes"), behave as before.

| Check | Before | After |
|---|---|---|
| The pydantic question | 38% | **100%** |
| The 29 existing list questions | 96.9% | 99.1% (only the pydantic question changed) |
| **20 generated questions** (every major.minor group of 2+ releases in every changelog in the test corpora, Markdown and PDF, not hand-picked) | 86.4% | **100%**, none worse |

End to end, qwen2.5:7b now lists all 8 releases in order, each with its own changes; spot-checked facts match the source.

Choose the strategy with `RAG_LIST_ANSWERS` or `python -m rag ask --list-answers single|checklist|per-item`. Concise mode and non-list questions always use one plain call.

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

Set these as environment variables, or pass the CLI flags `--chunk-size`, `--chunk-overlap`, `-k`, `--model`, `--mode`, `--retrieval` and `--list-answers`:

| Variable | Default |
|---|---|
| `RAG_LLM_MODEL` | `qwen2.5:7b` |
| `RAG_EMBED_MODEL` | `nomic-embed-text` |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `1000` / `150` (best config in the eval) |
| `RAG_ANSWER_MODE` | `detailed` (or `concise`) |
| `RAG_RETRIEVAL` | `hybrid` (or `vector`) |
| `RAG_EXPAND_LISTS` / `RAG_LIST_CHAR_BUDGET` | `1` / `20000` (set `0` to disable list expansion) |
| `RAG_LIST_ANSWERS` | `checklist` (or `single`, `per-item`; see "Long list answers") |
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
- **Nothing fails silently:** files with no extractable text (empty files, corrupt PDFs, scanned PDFs that need OCR) and unsupported files (.docx, .html, …) are reported, and replacing an uploaded file is announced.
- **Real-world file quirks:** an edge-case audit of 23 crafted files found and fixed:
  - Windows-1252 text that was silently losing every accented letter and smart quote
  - setext headings (`Title` + `===`) that weren't recognized
  - UTF-8 BOMs that hid the first heading
  - YAML front matter that was indexed as content
  - Office lock files (`~$x.md`) and hidden folders (`.obsidian/`) that were indexed

  The fixes were checked to leave every existing document's output unchanged.
- **Actionable errors:** the CLI and the app show messages like *"Can't reach Ollama…"* or *"The model `x` isn't installed. Run `ollama pull x`"* instead of stack traces.

## Tests

```bash
pytest          # 121 tests, no Ollama needed (a fake embedder stands in for it)
```

## Roadmap

- [x] Hybrid search (BM25 + vectors) and compare against pure vector search
- [ ] Cross-encoder re-ranking of the top-k passages
- [ ] Arabic document support (multilingual embedding model) and evaluation
- [ ] Answer caching and incremental re-indexing

## License

[MIT](LICENSE)
