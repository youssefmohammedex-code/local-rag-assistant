"""Finding filler sections in list answers and regenerating only those."""

import dataclasses
import re

from rag.filler import filler_headings, is_filler, sections, splice


def test_filler_bodies():
    for body in ["- No specific versions mentioned [1].", "- Not relevant to the question [3]",
                 "- No specific release notes provided [1.2.4]", "- [38]", "", "- No relevant information provided",
                 "- [Not applicable for this version]"]:
        assert is_filler(body), body
    for body in ["- Fix SOCKS support for http URLs (#492) [12].", "- Python 3.8+ is required [10].",
                 "- Adds X. " + "More real content here. " * 10 + "Details are not mentioned elsewhere."]:
        assert not is_filler(body), body


def test_sections_and_splice_change_only_the_named_bodies():
    answer = "Intro line.\n\n### 1.0\n- No changes listed [1].\n\n### 0.9\n- Real change [2].\n"
    assert [h for h, _, _ in sections(answer)] == ["1.0", "0.9"]
    assert filler_headings(answer) == ["1.0"]
    fixed = splice(answer, {"1.0": "- Added a parser [1]."})
    assert fixed == "Intro line.\n\n### 1.0\n- Added a parser [1].\n\n### 0.9\n- Real change [2].\n"


def _list_setup(pipeline):
    from test_lists import GUIDE, overlap_scorer
    pipeline.settings = dataclasses.replace(pipeline.settings, list_answers="checklist")
    pipeline.list_scorer = overlap_scorer
    (pipeline.settings.docs_dir / "guide.md").write_text(GUIDE, encoding="utf-8")
    pipeline.ingest()
    question = "Describe every module"
    return question, pipeline.retrieve(question, k=4)


class SummarizingLLM:
    """Writes a real bullet under each "### name" it is asked for, citing that section's first passage."""

    def __init__(self, filler_again=()):
        self.prompts = []
        self.filler_again = set(filler_again)

    def chat(self, model, messages, options, stream=False):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        out = []
        for name in re.findall(r"^### (.+)$", prompt, re.MULTILINE):
            first = re.search(rf'<passage id="(\d+)"[^>]*section="[^"]*{re.escape(name)}"', prompt)
            bullet = "- Nothing is mentioned here." if name in self.filler_again else \
                f"- {name} owns its component [{first.group(1) if first else 1}]."
            out.append(f"### {name}\n{bullet}\n")
        yield {"message": {"content": "\n".join(out)}, "done": True, "eval_count": 1, "eval_duration": 1e8}


def test_repair_regenerates_only_filler_sections_without_the_question(pipeline):
    question, hits = _list_setup(pipeline)
    names = [i.name for i in pipeline.list_items(question, hits) if i.name.startswith("Module")]
    answer = "\n\n".join(f"### {n}\n" + ("- No specific information provided [1]." if n.endswith(("A", "C"))
                                           else f"- {n} is described here [2].") for n in names)
    pipeline._llm = SummarizingLLM()
    fixed, repaired = pipeline.repair_filler(question, hits, answer)
    assert repaired == 2 and filler_headings(fixed) == []
    assert "### Module A — Part A\n- Module A — Part A owns its component" in fixed
    assert "- Module B — Part B is described here [2]." in fixed  # untouched
    prompt = pipeline._llm.prompts[0]
    assert question not in prompt and "Question: What does each of these sections" in prompt
    items = {i.name: i for i in pipeline.list_items(question, hits)}
    shown = {int(n) for n in re.findall(r'<passage id="(\d+)"', prompt)}
    assert shown == set(items[names[0]].passages) | set(items[names[2]].passages)  # real ids, only those sections


def test_repair_keeps_a_section_whose_new_text_is_filler_too(pipeline):
    question, hits = _list_setup(pipeline)
    name = next(i.name for i in pipeline.list_items(question, hits) if i.name.startswith("Module"))
    answer = f"### {name}\n- Not relevant to the question [1].\n"
    pipeline._llm = SummarizingLLM(filler_again={name})
    assert pipeline.repair_filler(question, hits, answer) == (answer, 0)


def test_bracketed_names_and_citation_fixes():
    from rag.pipeline import fix_item_citations, plain_name
    assert plain_name("[0.17.1] - 2021-04-29") == "0.17.1 - 2021-04-29"
    # A pull-request number copied as a citation becomes the section's only passage, or is dropped.
    assert fix_item_citations("- Fix FIFO reads [586] [26].", [26]) == "- Fix FIFO reads [26] [26]."
    assert fix_item_citations("- Fix FIFO reads [586].", [26, 27]) == "- Fix FIFO reads ."
    assert fix_item_citations("- Fix FIFO reads [27].", [26, 27]) == "- Fix FIFO reads [27]."
    # One passage: every bullet can only come from it, so uncited bullets get its citation.
    assert fix_item_citations("- Fix A.\n - Detail B.\n[34]\n", [34]) == "- Fix A. [34]\n - Detail B. [34]\n"
    assert fix_item_citations("- Fix A.\n- Fix B [27].", [26, 27]) == "- Fix A.\n- Fix B [27]."  # can't tell which


class IdHeadingLLM:
    """Writes the section's passage id as its heading ("### [3]"), as qwen did for bracketed names."""

    def chat(self, model, messages, options, stream=False):
        prompt = messages[-1]["content"]
        ids = re.findall(r'<passage id="(\d+)"', prompt)
        text = "\n".join(f"### [{i}]\n- Real content from this passage [{i}].\n" for i in ids)
        yield {"message": {"content": text}, "done": True, "eval_count": 1, "eval_duration": 1e8}


def test_a_passage_id_written_as_heading_still_maps_to_its_section(pipeline):
    question, hits = _list_setup(pipeline)
    item = next(i for i in pipeline.list_items(question, hits) if i.name.startswith("Module") and len(i.passages) == 1)
    answer = f"### {item.name}\n- Nothing is mentioned here.\n"
    pipeline._llm = IdHeadingLLM()
    fixed, repaired = pipeline.repair_filler(question, hits, answer)
    assert repaired == 1 and f"- Real content from this passage [{item.passages[0]}]." in fixed


def test_answers_without_filler_cost_no_call(pipeline):
    question, hits = _list_setup(pipeline)
    pipeline._llm = SummarizingLLM()
    answer = "### Module A — Part A\n- Module A owns component A [1].\n"
    assert pipeline.repair_filler(question, hits, answer) == (answer, 0)
    assert pipeline._llm.prompts == []
