"""Streamlit chat UI. Run with: streamlit run app.py"""

from __future__ import annotations

import dataclasses
import html
import re
import time
from pathlib import Path

import streamlit as st

from rag import citations
from rag.config import Settings
from rag.errors import friendly_error
from rag.filler import filler_headings
from rag.loaders import SUPPORTED_SUFFIXES, pretty_source
from rag.pipeline import MODES, RAGPipeline
from rag.store import Hit

st.set_page_config(page_title="DocChat · Local RAG", page_icon="📚", layout="centered")

CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2.2rem; max-width: 820px;}

.hero h1 {font-size: 2.1rem; font-weight: 750; margin: 0 0 .2rem 0; letter-spacing: -.02em;}
.hero h1 span {background: linear-gradient(90deg, #4f46e5, #0ea5e9); -webkit-background-clip: text;
               background-clip: text; color: transparent;}
.hero p {opacity: .7; margin: 0 0 .9rem 0;}
.pills {display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: 1.2rem;}
.pill {font-size: .78rem; padding: .18rem .65rem; border-radius: 999px;
       border: 1px solid rgba(128,128,128,.28); background: rgba(128,128,128,.07);}
.pill b {font-weight: 650;}
.pill.ok::before {content: "●"; color: #10b981; margin-right: .35rem;}
.pill.warn::before {content: "●"; color: #f59e0b; margin-right: .35rem;}

.welcome {border: 1px solid rgba(128,128,128,.22); border-radius: 14px; padding: 1.1rem 1.25rem;
          background: linear-gradient(135deg, rgba(79,70,229,.07), rgba(14,165,233,.05)); margin-bottom: .8rem;}
.welcome h3 {margin: 0 0 .3rem 0; font-size: 1.1rem;}
.welcome p {margin: 0; opacity: .75; font-size: .92rem;}

.cite {display: inline-block; font-size: .68rem; font-weight: 700; line-height: 1; padding: .18rem .38rem;
       margin: 0 .1rem; border-radius: 6px; background: rgba(79,70,229,.14); color: #4f46e5;
       vertical-align: .12rem;}
@media (prefers-color-scheme: dark) {.cite {color: #a5b4fc; background: rgba(129,140,248,.2);}}

.meta {font-size: .76rem; opacity: .6; margin-top: .4rem;}

.src {border: 1px solid rgba(128,128,128,.22); border-radius: 10px; padding: .6rem .8rem; margin-bottom: .55rem;}
.src.cited {border-left: 3px solid #4f46e5;}
.src-head {display: flex; justify-content: space-between; gap: .6rem; align-items: baseline;}
.src-title {font-weight: 650; font-size: .88rem;}
.src-score {font-size: .72rem; opacity: .65; white-space: nowrap;}
.src-path {font-size: .75rem; opacity: .65; margin: .1rem 0 .35rem 0;}
.src {overflow: hidden;}
.src-text {font-size: .8rem; opacity: .85; overflow-wrap: anywhere;}
.src-title {overflow-wrap: anywhere;}
.bar {height: 3px; border-radius: 3px; background: rgba(128,128,128,.18); margin-top: .45rem;}
.bar > div {height: 100%; border-radius: 3px; background: linear-gradient(90deg, #4f46e5, #0ea5e9);}

section[data-testid="stSidebar"] .doc {font-size: .84rem; padding: .3rem .1rem;
       border-bottom: 1px solid rgba(128,128,128,.15); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

VIA_LABEL = {"keyword": "keyword match · ", "both": "keyword + meaning · ", "meaning": "",
             "section": "list item · "}

EXAMPLES = [
    "Give me an overview of what these documents cover.",
    "What are the key decisions, and why were they made?",
    "List the main components and how they interact.",
]


# ---------- helpers ----------

@st.cache_resource(show_spinner=False)
def get_pipeline(model: str) -> RAGPipeline:
    return RAGPipeline(dataclasses.replace(Settings(), llm_model=model))


CITATION = re.compile(r"\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]")  # [3] or [1, 3]


def with_citation_badges(text: str) -> str:
    return CITATION.sub(lambda m: "".join(f'<span class="cite">{n}</span>' for n in re.findall(r"\d+", m.group(1))),
                        text)


def cited_ids(text: str) -> set[int]:
    return {int(n) for m in CITATION.finditer(text) for n in re.findall(r"\d+", m.group(1))}


def source_cards(hits: list[Hit], answer: str) -> str:
    used = cited_ids(answer)
    order = sorted(range(len(hits)), key=lambda i: (i + 1 not in used, i))
    cards = []
    for i in order:
        h = hits[i]
        body = h.text.split("\n\n", 1)[1] if h.section and "\n\n" in h.text else h.text
        body = re.sub(r"\*\*|__|`|\|", " ", body)  # drop Markdown emphasis/table syntax for a clean snippet
        body = re.sub(r"\s+", " ", body).strip()
        path = h.section.split(" > ") if h.section else []
        # Markdown docs: use the real "# Title" (export file names are often truncated).
        title = path[0] if path else pretty_source(h.source)
        where = " › ".join(path[1:]) or (f"Page {h.page}" if not path else "Introduction")
        if path and h.source.lower().endswith(".pdf"):
            where += f" · page {h.page}"
        snippet = body[:360] + ("…" if len(body) > 360 else "")
        cards.append(
            f'<div class="src {"cited" if i + 1 in used else ""}">'
            f'<div class="src-head"><span class="src-title">[{i + 1}] {html.escape(title)}</span>'
            f'<span class="src-score">{"cited · " if i + 1 in used else ""}{VIA_LABEL.get(h.via, "")}'
            f'{h.score:.0%} match</span></div>'
            f'<div class="src-path">{html.escape(where)}</div>'
            f'<div class="src-text">{html.escape(snippet)}</div>'
            f'<div class="bar"><div style="width:{max(h.score, 0) * 100:.0f}%"></div></div></div>'
        )
    return "".join(cards)


def render_assistant(message: dict) -> None:
    st.markdown(with_citation_badges(message["content"]), unsafe_allow_html=True)
    if message.get("meta"):
        st.markdown(f'<div class="meta">{message["meta"]}</div>', unsafe_allow_html=True)
    checks = message.get("checks") or []
    problems = citations.issues(checks)
    if problems:
        with st.expander(f"⚠️ Citation check · {citations.summary(checks)}"):
            st.caption("Checked by word overlap with each passage, so a flag means \"verify this\", not \"wrong\".")
            st.markdown("\n".join(f"- {html.escape(line)}" for line in problems))
    elif checks:
        st.markdown(f'<div class="meta">✓ Citation check · {citations.summary(checks)}</div>',
                    unsafe_allow_html=True)
    if message.get("hits"):
        n_cited = len(cited_ids(message["content"]) & set(range(1, len(message["hits"]) + 1)))
        n_list = sum(h.via == "section" for h in message["hits"])
        list_note = f" · {n_list} added as list items" if n_list else ""
        with st.expander(f"Sources · {n_cited} cited of {len(message['hits'])} retrieved{list_note}"):
            st.markdown(source_cards(message["hits"], message["content"]), unsafe_allow_html=True)


def reindex(pipeline: RAGPipeline, label: str, notes: list[str] | None = None) -> None:
    """Rebuild the index and leave a message for the next run. On failure the old index is kept."""
    start = time.perf_counter()
    try:
        with st.spinner(label):
            report = pipeline.ingest(reset=True)
    except Exception as exc:
        st.session_state.flash = [
            ("error", f"Indexing failed: {friendly_error(exc)} Your previous index is unchanged.")]
    else:
        flash = [("success", f"Indexed {report.chunks} passages from {report.files} file(s) "
                             f"in {time.perf_counter() - start:.0f}s.")]
        flash += [("info", note) for note in notes or []]
        flash += [("warning", f"Skipped **{s}**: no extractable text (empty, unreadable, or a scanned PDF "
                              "that needs OCR).") for s in report.skipped]
        if report.unsupported:
            flash.append(("warning", f"Ignored {len(report.unsupported)} unsupported file(s): "
                                     f"{', '.join(report.unsupported)}. Supported: .pdf, .md, .txt."))
        st.session_state.flash = flash
        st.session_state.skipped = set(report.skipped)  # known-unreadable files aren't "pending"
    st.rerun()


def docs_on_disk(settings: Settings) -> set[str]:
    if not settings.docs_dir.exists():
        return set()
    return {p.relative_to(settings.docs_dir).as_posix() for p in settings.docs_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES}


# ---------- sidebar ----------

defaults = Settings()
bootstrap = get_pipeline(defaults.llm_model)
models = bootstrap.installed_models() or [defaults.llm_model]

with st.sidebar:
    st.markdown("### ⚙️ Answer settings")
    mode_label = st.segmented_control("Answer length", ["Detailed", "Concise"], default="Detailed",
                                      help="Detailed retrieves more passages and writes a full, structured answer.")
    mode = (mode_label or "Detailed").lower()
    model = st.selectbox("Model", models, index=models.index(defaults.llm_model) if defaults.llm_model in models else 0,
                         help="Any chat model installed in Ollama. Bigger models are smarter but slower on CPU.")
    with st.expander("Advanced"):
        top_k = st.slider("Passages to retrieve", 1, 20, MODES[mode].k,
                          help="More passages give the model more to work with, but answers take longer.")
        search_label = st.radio("Search", ["Hybrid", "Meaning only"], horizontal=True,
                                help="Hybrid combines meaning-based search with keyword search, so exact "
                                     "IDs and names (e.g. D13, BR-05) are found too.")
        retrieval = "hybrid" if search_label == "Hybrid" else "vector"

    pipeline = get_pipeline(model)
    indexed = set(pipeline.store.sources())
    on_disk = docs_on_disk(pipeline.settings)

    st.markdown("### 📚 Library")
    uploads = st.file_uploader("Add documents", type=[s.lstrip(".") for s in SUPPORTED_SUFFIXES],
                               accept_multiple_files=True, label_visibility="collapsed")
    new_uploads = [u for u in uploads or [] if u.file_id not in st.session_state.setdefault("saved_uploads", set())]
    if new_uploads:
        pipeline.settings.docs_dir.mkdir(parents=True, exist_ok=True)
        notes = []
        for u in new_uploads:
            target = pipeline.settings.docs_dir / Path(u.name).name  # basename only: no path tricks
            if target.exists():
                notes.append(f"Replaced the existing **{target.name}** with the uploaded version.")
            target.write_bytes(u.getbuffer())
            st.session_state.saved_uploads.add(u.file_id)
        reindex(pipeline, f"Indexing {len(new_uploads)} new file(s)…", notes)

    for kind, message in st.session_state.pop("flash", []):
        getattr(st, kind)(message)

    pending = on_disk - indexed - st.session_state.get("skipped", set())
    missing = indexed - on_disk
    if pending:
        st.warning(f"{len(pending)} file(s) not indexed yet.")
    if missing:
        st.warning(f"{len(missing)} indexed file(s) were removed from disk. Re-index to drop them.")
    if st.button("Re-index all documents", use_container_width=True,
                 type="primary" if pending or missing else "secondary"):
        reindex(pipeline, "Embedding documents… (about 1 s per 5 passages)")

    for source in sorted(indexed):
        st.markdown(f'<div class="doc" title="{html.escape(source)}">📄 {html.escape(pretty_source(source))}</div>',
                    unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- header ----------

chunk_count = pipeline.store.count()
status = "ok" if chunk_count else "warn"
st.markdown(
    f"""<div class="hero"><h1>📚 <span>DocChat</span></h1>
    <p>Ask questions about your documents. Everything runs locally on this machine.</p></div>
    <div class="pills">
      <span class="pill {status}"><b>{len(indexed)}</b>&nbsp;documents · <b>{chunk_count}</b>&nbsp;passages</span>
      <span class="pill">Model&nbsp;<b>{html.escape(model)}</b></span>
      <span class="pill">Mode&nbsp;<b>{mode}</b> · k={top_k} · {search_label.lower()} search</span>
    </div>""",
    unsafe_allow_html=True,
)

# ---------- conversation ----------

messages: list[dict] = st.session_state.setdefault("messages", [])
question = st.chat_input("Ask something about your documents…") or st.session_state.pop("pending_question", None)

if not messages and not question:
    st.markdown(
        '<div class="welcome"><h3>👋 What would you like to know?</h3>'
        "<p>Answers come only from your documents, with numbered citations you can check in "
        "<b>Sources</b>. Try one of these:</p></div>",
        unsafe_allow_html=True,
    )
    for example in EXAMPLES:
        if st.button(f"💬 {example}", use_container_width=True):
            st.session_state.pending_question = example
            st.rerun()

for message in messages:
    with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "📚"):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_assistant(message)

if question:
    messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="📚"):
        if chunk_count == 0:
            reply = {"role": "assistant",
                     "content": "Your library is empty. Add files in the sidebar, "
                                "then click **Re-index all documents**."}
            render_assistant(reply)
        else:
            try:
                start = time.perf_counter()
                with st.spinner("Searching your documents…"):
                    hits = pipeline.retrieve(question, top_k, retrieval)
                placeholder, text, last_paint = st.empty(), "", 0.0
                placeholder.markdown('<span class="meta">Thinking…</span>', unsafe_allow_html=True)
                for token in pipeline.stream(question, hits, mode=mode):
                    text += token
                    if time.perf_counter() - last_paint > 0.08:
                        placeholder.markdown(with_citation_badges(text) + " ▍", unsafe_allow_html=True)
                        last_paint = time.perf_counter()
                n_filler, n_repaired = len(filler_headings(text)), 0
                if n_filler:
                    placeholder.markdown(with_citation_badges(text), unsafe_allow_html=True)
                    with st.spinner(f"Filling in {n_filler} section(s) the model left empty…"):
                        text, n_repaired = pipeline.repair_filler(question, hits, text)
                text = citations.move_code_citations(text, len(hits))  # keep copied code runnable
                placeholder.empty()
                stats = pipeline.last_stats
                meta = (f"{html.escape(model)} · {len(hits)} passages · {stats.tokens} tokens · "
                        f"{stats.tokens_per_second:.1f} tok/s · {time.perf_counter() - start:.0f}s")
                if n_repaired:
                    meta += f" · {n_repaired} empty section(s) filled in"
                checks = pipeline.check_citations(text, hits)
                reply = {"role": "assistant", "content": text.strip(), "hits": hits, "meta": meta,
                         "checks": checks}
                render_assistant(reply)
            except Exception as exc:  # most often: Ollama is not running or the model isn't pulled
                reply = {"role": "assistant", "content": f"⚠️ {friendly_error(exc)}"}
                st.markdown(reply["content"])
    messages.append(reply)
