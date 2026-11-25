"""Blind question sets for the generalization corpus, written from reading the docs, before any retrieval run."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from rag.loaders import read_text_file, split_markdown  # noqa: E402

OUT = Path(sys.argv[1])
DOCS = OUT / "docs"

precise = [
    # python-dotenv (markdown)
    ("How can I stop load_dotenv from reading .env files in production without changing the code that calls it?",
     "PYTHON_DOTENV_DISABLED", "semantic"),
    ("If a key appears in my .env file with no equals sign, what value does dotenv_values give it?",
     "associating that variable name with the value", "semantic"),
    ("With override turned off, which wins: the variable already in the environment or the one in the .env file?",
     "override=False", "semantic"),
    ("Can a quoted value in a .env file span several lines?", "span multiple lines", "semantic"),
    ("Which command prints the variables in JSON format?", "dotenv list --format=json", "semantic"),
    ("What changed in python-dotenv 0.10.0?", "Add support for trailing comments", "id"),
    # orjson (markdown)
    ("What range of integers can orjson handle by default?", "18446744073709551615", "semantic"),
    ("How do I make orjson raise an error for integers that web browsers cannot represent exactly?",
     "OPT_STRICT_INTEGER", "semantic"),
    ("What format does orjson use when writing datetime objects?", "RFC 3339", "semantic"),
    ("Does orjson output a str or bytes?", "is the correct type for a serialized blob", "semantic"),
    ("Is JSON Lines supported by orjson?", "orjsonl", "semantic"),
    ("What option is needed to serialize numpy arrays with orjson?", "OPT_SERIALIZE_NUMPY", "semantic"),
    ("What version constraint should I put in requirements.txt for orjson?", "orjson >= 3.10,<4", "semantic"),
    # annotated-types (markdown)
    ("What are the two ways MultipleOf can be interpreted?", "JSONschema semantics", "semantic"),
    ("How do I say a datetime must be naive?", "Timezone(None)", "semantic"),
    ("How do I express both a lower and an upper bound in one constraint?", "Interval(gt, ge, lt, le)", "semantic"),
    # typer (markdown)
    ("What happened to typer-slim?", "now simply installs", "semantic"),
    ("How can I turn off Rich formatting in Typer globally?", "TYPER_USE_RICH", "semantic"),
    ("Which package does Typer use to detect the shell when installing completion?", "shellingham", "semantic"),
    # tqdm (plain text, RST markup, no sections)
    ("How do I reduce tqdm log spam in CI jobs?", "TQDM_MININTERVAL=5", "semantic"),
    ("What is the default maximum interval between tqdm display refreshes?", "maxinterval=10.0", "semantic"),
    # numpy licenses (plain text, legal)
    ("Under which license is the bundled OpenBLAS distributed?", "Name: OpenBLAS", "semantic"),
    ("What license covers the GCC runtime library shipped with numpy?", "GCC-exception-3.1", "semantic"),
    # httpcore (PDF, no headings)
    ("Which Python version does httpcore require?", "Python 3.8+", "semantic"),
    ("Which async backends does httpcore support?", "Async backend support for", "semantic"),
    ("How do I install httpcore with HTTP/2 and SOCKS support?", "asyncio,trio,http2,socks", "semantic"),
    # charset-normalizer (markdown, emoji headings, no title)
    ("When is charset-normalizer's language detection unreliable?",
     "two or more languages sharing identical letters", "semantic"),
    ("What detection accuracy does charset-normalizer report in its benchmark?", "98 %", "semantic"),
    # pydantic (large markdown changelog, version IDs)
    ("Which Python versions did pydantic v2.14.0 drop and add?", "drops support for Python 3.9", "id"),
    ("What is in the pydantic v2.13.4 release?", "v2.13.4", "id"),
]
with (OUT / "q_precise.jsonl").open("w", encoding="utf-8") as f:
    for q, text, kind in precise:
        f.write(json.dumps({"question": q, "expected_text": text, "type": kind}) + "\n")


def child_headings(file: str, parent_last: str) -> list[str]:
    kids = []
    for path, _ in split_markdown(read_text_file(DOCS / file)):
        parts = path.split(" > ")
        if parent_last in parts[:-1] and parts[parts.index(parent_last) + 1] not in kids:
            kids.append(parts[parts.index(parent_last) + 1])
    return kids


changelog = child_headings("python-dotenv.md", "Changelog")
lists = [
    ("Summarize every feature of the .env file format.",
     ["Multiline values", "Variable without a value", "Variable expansion"]),
    ("Describe each constraint type that annotated-types provides.",
     ["Gt, Ge, Lt, Le", "Interval", "MultipleOf", "MinLen, MaxLen, Len", "Timezone", "Unit", "Predicate", "> Doc\n"]),
    ("List all the types orjson can serialize natively.",
     ["> dataclass\n", "> datetime\n", "> enum\n", "> float\n", "> int\n", "> numpy\n", "> str\n", "> uuid\n"]),
    ("What are all the questions in orjson's FAQ?",
     ["Will it deserialize to dataclasses", "Will it serialize to `str`?", "Will it support NDJSON or JSONL?",
      "Will it support JSON5 or RJSON?", "How do I depend on orjson in a Rust project?"]),
    ("List every python-dotenv release in the changelog.", [f"Changelog > {v}" for v in changelog]),
    ("List all of Typer's dependencies.", ["`rich`", "`pydantic`", "`shellingham`", "`annotated-doc`", "`colorama`"]),
]
with (OUT / "q_lists.jsonl").open("w", encoding="utf-8") as f:
    for q, items in lists:
        f.write(json.dumps({"question": q, "expected_items": items}) + "\n")

ood = ["What is the capital of Australia?", "How do I configure Kubernetes autoscaling?",
       "Does orjson support YAML?"]
with (OUT / "q_ood.jsonl").open("w", encoding="utf-8") as f:
    for q in ood:
        f.write(json.dumps({"question": q}) + "\n")

print(f"{len(precise)} precise, {len(lists)} list ({len(changelog)} changelog releases), {len(ood)} out-of-scope")
