"""Streamlit chat UI. Run with: streamlit run app.py"""

from __future__ import annotations

import dataclasses
import html
import re
import time

import streamlit as st

from rag.config import Settings
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

EXAMPLES = [
    "Give me an overview of what these documents cover.",
    "What are the key decisions, and why were they made?",
    "List the main components and how they interact.",
]


# ---------- helpers ----------

@st.cache_resource(show_spinner=False)
def get_pipeline(model: str) -> RAGPipeline:
    return RAGPipeline(dataclasses.replace(Settings(), llm_model=model))


def with_citation_badges(text: str) -> str:
    return re.sub(r"\[(\d{1,2})\]", r'<span class="cite">\1</span>', text)


def cited_ids(text: str) -> set[int]:
    return {int(n) for n in re.findall(r"\[(\d{1,2})\]", text)}


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
        snippet = body[:360] + ("…" if len(body) > 360 else "")
        cards.append(
            f'<div class="src {"cited" if i + 1 in used else ""}">'
            f'<div class="src-head"><span class="src-title">[{i + 1}] {html.escape(title)}</span>'
            f'<span class="src-score">{"cited · " if i + 1 in used else ""}{h.score:.0%} match</span></div>'
            f'<div class="src-path">{html.escape(where)}</div>'
            f'<div class="src-text">{html.escape(snippet)}</div>'
            f'<div class="bar"><div style="width:{max(h.score, 0) * 100:.0f}%"></div></div></div>'
        )
    return "".join(cards)


def render_assistant(message: dict) -> None:
    st.markdown(with_citation_badges(message["content"]), unsafe_allow_html=True)
    if message.get("meta"):
        st.markdown(f'<div class="meta">{message["meta"]}</div>', unsafe_allow_html=True)
    if message.get("hits"):
        n_cited = len(cited_ids(message["content"]) & set(range(1, len(message["hits"]) + 1)))
        with st.expander(f"Sources · {n_cited} cited of {len(message['hits'])} retrieved"):
            st.markdown(source_cards(message["hits"], message["content"]), unsafe_allow_html=True)


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

    pipeline = get_pipeline(model)
    indexed = set(pipeline.store.sources())
    on_disk = docs_on_disk(pipeline.settings)

    st.markdown("### 📚 Library")
    uploads = st.file_uploader("Add documents", type=[s.lstrip(".") for s in SUPPORTED_SUFFIXES],
                               accept_multiple_files=True, label_visibility="collapsed")
    new_uploads = [u for u in uploads or [] if u.file_id not in st.session_state.setdefault("saved_uploads", set())]
    if new_uploads:
        pipeline.settings.docs_dir.mkdir(parents=True, exist_ok=True)
        for u in new_uploads:
            (pipeline.settings.docs_dir / u.name).write_bytes(u.getbuffer())
            st.session_state.saved_uploads.add(u.file_id)
        with st.spinner(f"Indexing {len(new_uploads)} new file(s)…"):
            n = pipeline.ingest(reset=True)
        st.toast(f"Indexed {n} passages", icon="✅")
        st.rerun()

    pending = on_disk - indexed
    if pending:
        st.warning(f"{len(pending)} file(s) not indexed yet.")
    if st.button("Re-index all documents", use_container_width=True, type="primary" if pending else "secondary"):
        with st.spinner("Embedding documents… (about 1 s per 5 passages)"):
            start = time.perf_counter()
            n = pipeline.ingest(reset=True)
        st.toast(f"Indexed {n} passages in {time.perf_counter() - start:.0f}s", icon="✅")
        st.rerun()

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
      <span class="pill">Mode&nbsp;<b>{mode}</b> · k={top_k}</span>
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
                     "content": "Your library is empty. Add files in the sidebar, then click **Re-index all documents**."}
            render_assistant(reply)
        else:
            try:
                start = time.perf_counter()
                with st.spinner("Searching your documents…"):
                    hits = pipeline.retrieve(question, top_k)
                placeholder, text, last_paint = st.empty(), "", 0.0
                placeholder.markdown('<span class="meta">Thinking…</span>', unsafe_allow_html=True)
                for token in pipeline.stream(question, hits, mode=mode):
                    text += token
                    if time.perf_counter() - last_paint > 0.08:
                        placeholder.markdown(with_citation_badges(text) + " ▍", unsafe_allow_html=True)
                        last_paint = time.perf_counter()
                placeholder.empty()
                stats = pipeline.last_stats
                meta = (f"{html.escape(model)} · {len(hits)} passages · {stats.tokens} tokens · "
                        f"{stats.tokens_per_second:.1f} tok/s · {time.perf_counter() - start:.0f}s")
                reply = {"role": "assistant", "content": text.strip(), "hits": hits, "meta": meta}
                render_assistant(reply)
            except Exception as exc:  # most often: Ollama is not running
                reply = {"role": "assistant",
                         "content": f"⚠️ Couldn't reach the local model: `{exc}`. "
                                    "Make sure Ollama is running (open it from the Start menu) and try again."}
                st.markdown(reply["content"])
    messages.append(reply)
