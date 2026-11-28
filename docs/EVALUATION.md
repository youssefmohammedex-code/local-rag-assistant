# Evaluation and engineering notes

How each part of the [Local RAG Assistant](../README.md) was measured, what failed, and what was changed. Sections are in the order the work was done, so later sections sometimes correct earlier ones; corrections say so.

## Contents

- [Retrieval evaluation harness](#retrieval-evaluation-harness)
  - [Results](#results)
  - [Retrieval comparison: vector-only vs hybrid](#retrieval-comparison-vector-only-vs-hybrid)
  - [List questions: section expansion](#list-questions-section-expansion)
  - [Generalization test: documents the system was never tuned on](#generalization-test-documents-the-system-was-never-tuned-on)
  - [PDFs: recovering structure](#pdfs-recovering-structure)
  - [Choosing how many lists to expand](#choosing-how-many-lists-to-expand)
  - [Long list answers: helping the 7B model not skip items](#long-list-answers-helping-the-7b-model-not-skip-items)
  - [List questions that name a code ("every 2.14 release")](#list-questions-that-name-a-code-every-214-release)
  - [Choosing between the two expanded lists (the httpcore invention)](#choosing-between-the-two-expanded-lists-the-httpcore-invention)
  - [Filler under each item of a whole-changelog list (fixed by repairing only those sections)](#filler-under-each-item-of-a-whole-changelog-list-fixed-by-repairing-only-those-sections)
  - [Checking citations](#checking-citations)
  - [Reproducible list-answer benchmark](#reproducible-list-answer-benchmark)
  - [The list chooser answered by position: a heading-name fallback](#the-list-chooser-answered-by-position-a-heading-name-fallback)
  - [Model comparison: `llama3.2:3b` vs `qwen2.5:7b`](#model-comparison-llama323b-vs-qwen257b)
  - [Stopping repetition loops](#stopping-repetition-loops)
  - [Very long lists: whole changelogs](#very-long-lists-whole-changelogs)
  - [How-to answers with code left most sentences uncited](#how-to-answers-with-code-left-most-sentences-uncited)

## Retrieval evaluation harness

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
  - **An outline can invite confident invention.** On httpcore the model wrote plausible facts with citations that don't support them. The single strategy was also wrong there, just more obviously. *(Fixed below, under "Choosing between the two expanded lists".)*
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

### Choosing between the two expanded lists (the httpcore invention)

**Why the model invented content:** for *"What are all the sections of the httpcore guide?"*, retrieval expanded two lists, the guide's 7 sections and the changelog's 39 releases, because embedding similarity ranked the changelog higher. The answer had the right headings, but filled them with plausible facts taken from the 40 unrelated passages.

**The fix: ask the model which list the question means, before answering.**
- **The question:** a one-letter multiple choice ("A or B?"), which a 7B model handles far more reliably than ignoring 40 irrelevant passages.
- **Asked in both orders:** the model leans towards one letter when unsure, so a pick counts only if both orders agree. Otherwise both lists are kept, as before.
- **What the answer then sees:** the normal top-k passages (always kept) plus the chosen list's passages only, shown in document order.
- **Cost:** two extra replies of about one token each.

**Chooser picks on all 13 list questions:**
- **Correct:** 11 picks.
- **Abstained:** 2 questions, python-dotenv's use cases and Typer's example upgrade, where the orders disagreed and both lists were kept. These turned out to be position bias, and are now resolved by a fallback question (see "The list chooser answered by position" below).
- **Nothing to choose:** the Rich README expands a single list.
- **Wrong picks:** none.

| Check | Before | After |
|---|---|---|
| httpcore guide, 3 runs, read by hand | right headings, invented content | **all 7 sections in every run, facts match the passages** (Python 3.8+, h11/certifi, socksio, SEMVER, `httpcore==1.*`) |
| 6 list questions used while designing | 96.7% | **100%** (orjson FAQ was 60–100% across runs) |
| 7 held-out list questions | 5 at 100%, Typer upgrade 67%, httpcore invented | **all 7 at 100%** |

One caveat on the held-out row: Typer's upgrade question is one where the chooser abstained. Its 67% → 100% change came with document order and may partly be run-to-run variance.

### Filler under each item of a whole-changelog list (fixed by repairing only those sections)

**What happens:**
- **The questions:** ones that ask only for the names of a changelog's releases, e.g. *"Which releases does the python-dotenv changelog list?"*
- **What works:** the answer usually names every release, in order.
- **What fails:** under each release, qwen2.5:7b writes a filler bullet such as "No specific versions mentioned" or "not relevant to the question". The passages it was shown do contain that release's changes. It's not invented content, but the answer doesn't say what changed.

**How it was measured:**
- **The questions:** every changelog in the test corpora with 5+ releases (httpcore, python-dotenv, pydantic), each asked in 3 phrasings. They were generated, not hand-picked.
- **The score:** the share of a section's bullets that only say there is nothing there.
- **A check on the score itself:** it reads 0 on all 13 regular list questions.

| Wording of the per-section instruction | python-dotenv (held out) | httpcore | pydantic |
|---|---|---|---|
| current: "1-3 bullet points using only the passages from that section" | 84 of 84 sections filler (2 phrasings) | 35 of 39 | none |
| + "summarize ... even when the question only asks for the names" | still filler; one answer looped | 102 of 102 | **real changes** under each release |
| + "never write that a section has no information" (every outlined section has passages) | still filler, all 3 phrasings | not finished (stopped once python-dotenv failed) | not run |

**Decision:** none of the wordings fixed the documents they weren't tuned on, so the original wording stays.

**Also tried and reverted:** numbering passages 1..n in document order, with citations mapped back to their real ids. It didn't help, and the pydantic 2.14 question dropped to 88%.

**Why it's probably hard to fix with wording:** the model seems to check each section's text for the thing the question asks about, like "versions". An instruction that's needed elsewhere invites exactly that check: *"if the items the question asks for are listed inside a section's text, use those items as headings"*. orjson's FAQ depends on it.

**The fix: leave the answer as written and regenerate only its filler sections** (`RAGPipeline.repair_filler`).
- **Finding filler:** after the answer is written, code finds the sections whose body only says there is nothing there, or has no text.
- **Regenerating:** those sections are regenerated a few at a time, without the user's question, which is what triggers the filler. The task is neutral ("what does each of these sections say?"). Each call sees only those sections' passages, under their real ids.
- **Splicing:** the new text replaces just those sections. A section whose new text is filler again is left as it was.
- **Cost:** answers without filler aren't touched and cost nothing. Across the 13 regular list questions, 0 sections are filler, so their results can't change.

**Three more problems showed up on python-dotenv, each fixed in code:**
- **Bracketed headings:** names like `[0.17.1] - 2021-04-29` made the model write passage ids as headings (`### [26]`). Repair prompts now list names without brackets, and an id heading is still matched through its passage.
- **Neighbouring releases mixed in:** passages run on into the next release, and the model mixed its changes in, so one `dotenv_path` change appeared under four releases. Passages in repair prompts are now cut to their own section, using the same splitter as the citation checker.
- **Pull-request numbers copied as citations** (`[#700]` → `[700]`): these are replaced by the section's passage.
  - **Uncited bullets:** a section regenerated from a single passage can only restate that passage, so its uncited bullets get that citation. The model had left 213 of 378 bullets uncited.

| Filler sections repaired | Before the three fixes | After |
|---|---|---|
| httpcore "list every release" (used to design it) | 35 of 35 | — |
| python-dotenv, 2 answers (exposed the three problems) | 65 of 84 | **82 of 84** |
| **httpcore, 3 answers never used for design or tuning** | — | **102 of 102** |

**Checking the repaired sections:**
- **Citations:** all 367 cited bullets in them pass the citation check. 8 bullets in sections built from several passages are left uncited.
- **By hand:** 16 random repaired sections were compared word for word with their source passages. All are correct, with no invented changes and none moved from a neighbouring release.

**Cost:** about one extra model call per 8 filler sections. A 45-release answer takes about 15 more minutes on a laptop CPU.

### Checking citations

**Why:** qwen2.5:7b often cites by position rather than content. It puts [1], [2], [3] under the 1st, 2nd and 3rd headings, whatever those passages say. It also moves a fact to a neighbouring heading. A citation is a claim to verify, not a fact.

**How:** after the answer is written, every cited sentence is compared with every passage it was given. This uses plain word overlap, with no extra model call; it takes milliseconds.
- **The score:** the share of the sentence's content words that the passage contains.
- **Supported:** a cited passage contains at least half of them.
- **Miscited:** no cited passage does, but another passage clearly does, and that one is named as the likely right source.
- **Unsupported:** no passage does. The sentence may be invented, or paraphrased too loosely to check.

The app shows the result under each answer. It's a single ✓ line when everything checks out, or a "Citation check" list of the sentences to verify. The CLI prints the same.

**Three rules, each added for a failure found in real answers:**
- **Numbers and codes must match:** "Python 3.6" is not supported by a passage that says 3.8, however many other words they share. The same goes for `#478`, `v2.11.8`, "Module C" and "2 weeks".
- **Only a passage's own section counts:**
  - **The problem:** a passage can run on into the next sections. The one for release 0.14.6 ends with "0.14.5 (January 18th, 2022)" and that release's changes, so a 0.14.5 change cited for 0.14.6 looked supported.
  - **The rule:** a passage is cut where another section of the same document begins.
  - **Exception 1:** its own sub-sections (a release's "Fixed") don't end it.
  - **Exception 2:** text it runs on into still counts for a sentence written under that section's heading.
- **A sentence that names a version may use that version's text:** "v2.11.8: Fix mypy plugin" may use the v2.11.8 text that a neighbouring passage runs on into.

**How it was evaluated:**
- **Answers that already existed:** saved benchmark answers, re-checked against the passages they were generated from. No answer was generated for this.
- **Hand-labeling:** every flag was read by hand, and so was a random sample of what it called supported.
- **Freezing:** the rules were fixed before the unseen sets. Set 2 was never used for any change.

| Set | Claims | Flags that are real errors | Wrong claims missed |
|---|---|---|---|
| Known-bad: two old httpcore answers with invented content (used to design the rules) | 66 | — | caught 35 of 36, 1 false alarm in 30 correct claims |
| Unseen set 1: 14 list answers | 172 | 12 of 14 | 0 in a sample of 40 |
| Unseen set 2: 11 list answers (no change made after seeing it) | 112 | **10 of 10** | 1 in a sample of 30 |

**What it found in answers that scored 100% on coverage:**
- **The student handbook answer:** probation, graduation and exam rules cite the lab-safety or grading passages: 5 sentences in one run and 6 in another.
- **orjson:** two option flags cite each other's passages.
- **Typer:** "FastAPI's little sibling" cites the Installation passage.

**Limits:**
- **No judgment of meaning:** a sentence that reuses a passage's words but reverses what it says would pass.
- **Generic sentences can pass on shared words alone.** In the known-bad set, "the major version is incremented for backwards incompatible changes" shared "major", "version" and "changes" with the versioning passage.
- **The 2 false alarms on set 1:**
  - **Rich:** a summary of Rich's examples that the cited passage introduces under other words.
  - **pydantic:** a release change filed under a sub-heading whose release line carries extra link text.
- **Stricter thresholds:** at 0.6 it caught one more real error on the regular answers but added three false alarms on fair paraphrases ("brute force approach" for "I'm brute forcing"), so 0.5 stays.

The checker also flags filler. "No relevant information provided [1]" comes out unsupported in 88 of 88 cases.

### Reproducible list-answer benchmark

`eval/list_answers.py` runs the list questions end to end (retrieve, answer, repair filler) and scores:
- coverage of the expected items
- headings
- filler sections left
- citation check
- time

It runs with temperature 0, a fixed seed and `--cold` by default, so a rerun on the same machine and Ollama version gives the same answers.

**Why `--cold` is needed:**
- **A seed alone wasn't enough:** two seeded runs of the Typer README question differed in two sentences' wording.
- **Why:** the second run reused Ollama's cached copy of the prompt (80 s instead of 157 s), and computing it from cache gives slightly different numbers.
- **The fix:** `--cold` unloads the model after every call. Two cold runs were then identical character for character, and identical to the first, uncached, run.
- **The evidence:** `eval/results/determinism_*.md`.

```bash
# build the test documents first (eval/generalization/README.md), then one command per document set:
python -m eval.list_answers --cold --docs ../gen/docs --db ../gen/db --out list_answers_gen \
    --questions eval/generalization/q_lists.jsonl eval/generalization/q_lists_fresh.jsonl eval/generalization/q_lists_heldout.jsonl
python -m eval.list_answers --cold --docs ../pdf/docs --db ../pdf/db --out list_answers_pdf \
    --questions eval/generalization/q_pdf_lists.jsonl eval/generalization/q_pdf_lists_heldout.jsonl
python -m eval.list_answers --cold --docs data/docs/sample --db data/chroma_eval/sample --out list_answers_sample \
    --questions eval/questions_broad.jsonl
```

**Baseline (seed 42, temperature 0):** 12 public list questions, 5 used in design and 7 held out.
- **First run:** 11 were at 100%. Typer's "walk me through each part of the example upgrade" was at 67%, so the mean was 97.2%. This corrected the list-chooser results above, where that question showed 100% in one run, flagged there as possibly run-to-run variance.
- **After fixing the chooser's position bias (next section):** **all 12 at 100%**.
- **Citations:** 124 of 135 cited sentences pass the citation check.
- **Filler:** no answer has any left.
- **Time:** about 45 minutes.
- **Per-question tables:** `eval/results/list_answers_{gen,pdf,sample}.md`.

### The list chooser answered by position: a heading-name fallback

**The Typer 67%, investigated:**
- **Retrieval found both lists:** Typer's 7 top-level README sections, and the 3 parts of "Example upgrade" the question asks about.
- **The chooser answered "B" in both orders,** so it was picking a position, not a list. The two-order check correctly called that a disagreement and kept both lists.
- **The answer outline then mixed both lists.** The model filed "An example with two subcommands" under the heading "Example upgrade", so the scorer, which looks for the item's name, missed it.
- **The same thing happened to python-dotenv's use cases.**

**The fix:**
- **When the letters disagree,** the model is asked again in both orders, this time to answer with the list's heading (`choose_heading_prompt`). A heading can't be chosen by position.
- **The reply must match one heading exactly,** in full or by its last part ("Other Use Cases"), and both orders must agree.
- **Questions the letters already decide never reach the heading question,** so their picks can't change.

**Why not use the heading question alone:** on the 13 design questions it was worse than letters wherever letters already worked. It named "Serialize" for orjson's option flags, which identifies neither list, and gave a different heading in each order for the private modules question. As a fallback it only gets the cases letters can't decide.

| Chooser check | Result |
|---|---|
| 13 design questions | both stuck cases resolved correctly; the other 11 unchanged |
| **16 list questions not used for this** (7 public, 9 on private docs) | letters decided 11, all correct. Of the 5 that needed the fallback: **3 correct picks, 2 still undecided (both lists kept, as before), 0 wrong** |
| Reproducible rerun of the 9 package-doc list questions | Typer upgrade 67% → **100%**; python-dotenv use cases still 100%, now with exactly its 4 use cases. The **7 questions the letters decide reproduced the earlier answers character for character**, in a separate run on a later day |

To measure the spread at the app's real setting instead, use `--temperature 0.1 --seed none --runs 3`. That takes about 3 hours.

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

### Stopping repetition loops

Answers are capped at 3,000 tokens (see "Reliability" in the README), but at ~3 tokens/s on a laptop CPU the cap alone still means ~15 minutes of repeated text. So the answer is also watched as it streams.

- **The rule:** once citation numbers are ignored, the answer stops when its last lines are one block of 1–30 lines repeated 3 times in a row. A note is added saying why, and the stream is closed, which makes Ollama cancel the generation (checked in its log).
- **Checked on every saved benchmark answer:** all 124 were replayed in small chunks, the way tokens stream. It stopped exactly the 3 runaways and none of the other 121, including filler answers that repeat one bullet under 39 different headings. The 8.8-hour answer now stops after 4,700 characters instead of 345,000.
- **The margin, honestly:**
  - **Why 3 copies:** with 2 copies in a row, it would also cut two normal answers whose example code has a similar block twice.
  - **What gets lost:** one stopped answer (a single line repeated about 48 times) had later recovered and listed the releases. Stopping loses that part, a trade-off for never showing a long loop.

### Very long lists: whole changelogs

**The problem:** "List every release" over a changelog with 35–48 releases. The context budget for passages (20,000 characters) holds about 30 releases, and two things went wrong:
- **Retrieval picked sub-lists:** for pydantic, two releases' "What's Changed" lists (New Features, Fixes, …) ranked above the changelog itself, so only 17–33% of the releases reached the model.
- **The model skipped what it was given:** charset-normalizer had 35 releases with passages; the answer wrote 11 and skipped all 24 of the 3.x releases at the top (57% coverage).

**The changes:**
- **A list the hits spread across:** when one candidate list has hits in at least 4 of its sub-sections and one of the two picked lists sits inside it, it replaces the second pick. Unrelated lists are never swapped in (the first version of this rule did that in 4 questions, including python-dotenv's use cases, and was narrowed).
- **A closing note:** items of the answered list that had no passages in the context are named from the document's headings under "Also in this list, without details", so the answer says what it left out instead of looking complete.
- **Skipped sections are written afterwards,** from their own passages, and inserted in document order. This only runs when the answer clearly follows the list: at least 3 headings, and most of them, must be exact item names. A looser check matched orjson's FAQ answer (one heading, "Questions") and Typer's example steps.
- **Copied sections are repaired:** in pydantic's answer the model copied one release's three bullets, citation and all, under 13 other releases that had passages of their own. A section whose body is word for word another section's body is now regenerated like filler. The citation checker also missed those bullets, because they were written "- [22] Text" with the citation first; it now reads that form.
- **Filler detection** also catches placeholder bullets like "- [Not applicable for this version]".

None of these rules changes any of the 12 suite answers: the spread rule leaves their retrieval unchanged, and the repair rules don't fire on any saved answer except the long-list ones.

| Check | Before | After |
|---|---|---|
| Retrieved coverage, 41 list questions over 4 document sets | 87.2% | **95.7%** (6 better, 0 worse; thresholds 3–5 give the same result) |
| charset-normalizer, "List every release" (35 of 46 releases fit) | 57% | **100%** of headings; the 24 skipped 3.x releases were written and each checked against the document; the 11 that didn't fit are named in the note; 49 of 57 citations check out (the 8 that don't are in sections the model wrote in its first pass) |
| httpcore (PDF), "What are all the versions" | 85% retrieved | **100%**, 0 filler, **88 of 88** citations check out |
| pydantic, "What are all the versions" | 17% retrieved | **100%**, 0 filler, **47 of 55** citations check out; the 13 copied sections now each have their own release's changes; the 17 releases that didn't fit are named in the note |

All answers are cold and seeded (`--cold`, seed 42, temperature 0); the before numbers come from the same questions on the earlier code. Results are in `eval/results/long_lists_{gen,pdf,pyd}.md`. The pydantic row in `long_lists_gen` is the answer with the copied sections, before that repair; `long_lists_pyd` is the rerun after it.

**Still imperfect in the pydantic answer:** the chooser picked the README's top-level list, which in that file holds both the README sections and the releases. So the answer also has 8 headings such as "Installation" and "Help", and the note names a "New Contributors" sub-heading as if it were a release. Nothing is made up, but a person would have left those out.

**Tried and reverted:**
- **A yes/no question per list,** for when the chooser can't decide: in every undecided case the model said yes to both lists or no to both, so it never decided anything.
- **Completing the outer list** when the two candidates are nested (the whole changelog and one release's "Added / Fixed"): right for "every release in the changelog", wrong for questions like Typer's "each part of the example upgrade", where the inner list is the answer. The document's structure can't tell these apart; only the question's wording can.

**The limit that remains:** across 12 phrasings of whole-changelog questions, the chooser decided 7 (all correctly) and stayed undecided on 5, all of them nested lists. When it's undecided, both lists are kept, and filler repair still runs, but the note and the skipped-section repair don't, because it isn't clear which list the answer should complete. The three questions above are phrasings the chooser decides, one per changelog.

### How-to answers with code left most sentences uncited

**Found while recording the README demo:** "How do I load variables from a .env file with python-dotenv?" got a correct, well-structured answer with code, and 0 of 10 passages cited. The benchmarks had no how-to questions, so this was never measured.

**Measured** on 12 how-to questions over the demo library (python-dotenv, Typer, orjson, httpcore): 6 to find the cause, 6 held out to test the fix, plus 3 factual questions as controls. One seeded answer each, counting the sentences outside code that carry a citation.

**Why it happened:**
- **Citations moved off the text:** the model put a step's citation on its own line after the code block ("[6]"), or as a footer ("Passage id: [9]"). The step itself was left uncited, so the citation checker had nothing to check.
- **No citations at all** in some answers. The citation rule's only example was a short fact ("a GPA of 3.0 [2]"), not a step or a code example.

**The fix:**
- **The citation rule now says** steps and code examples are facts too, and that citations go inside the sentences, not in a list at the end.
- **That made the model cite inside the code** ("pip install python-dotenv [1]", "load_dotenv()  # ... [6]"), which breaks code a user copies. So after the answer is written, citations at the end of a code line, and citation lines right after a block, are moved to the last sentence before the block (`move_code_citations`). Only real passage ids move, and only after a word or comment, never after "=", "(" or ",", so code such as `x = [1]` is untouched. Replayed on every saved answer, it changed only the how-to answers.

| How-to answers | Sentences with a citation |
|---|---|
| 6 questions used to find the cause | 11% (8 of 73) → **41%** (23 of 56) |
| **6 held-out questions** | 15% (8 of 53) → **37%** (17 of 46); citations that check out 3 → 8 |

**Honest limits:**
- **Most how-to sentences are still uncited.** Adding citations automatically by word overlap was not done: it would attribute sentences the model never attributed.
- **One control answer got worse:** for "Which Python versions does orjson support?" the new answer invented "Python 3.95" (it mixed in Rust 1.95). The citation checker flagged all three of its claims. With one sample per question, it can't be told apart from run-to-run variation.
- **The rule is in every prompt,** so the list benchmark answers saved before this change no longer match the current code exactly.
