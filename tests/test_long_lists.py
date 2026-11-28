"""Very long lists: finding a changelog whose summary ranks low, and naming the items that don't fit."""

import dataclasses

from rag.pipeline import NOT_COVERED_HEADING, hits_per_child, items_without_passages, not_covered_note
from rag.store import Hit


def _hit(section, source="log.md"):
    return Hit(id=section, text=f"{section}\n\ntext", source=source, page=1, section=section, score=0.5)


RELEASES = ("log.md", "Log", [(f"v1.{i}", f"c{i}") for i in range(10)])


def test_hits_per_child_counts_sub_sections_not_passages():
    hits = [_hit("Log > v1.1"), _hit("Log > v1.1 > Fixed"), _hit("Log > v1.2 > Added"), _hit("Log > v1.3"),
            _hit("Other > v1.4"), _hit("Log > v1.5", source="other.md")]
    assert hits_per_child(RELEASES, hits) == 3  # v1.1 (twice), v1.2, v1.3; wrong path and wrong file don't count


def test_items_without_passages_and_the_note():
    hits = [_hit(f"Log > v1.{i}") for i in range(7)]
    assert items_without_passages(RELEASES, hits) == ["v1.7", "v1.8", "v1.9"]
    note = not_covered_note(["v1.7", "v1.8", "v1.9"])
    assert f"### {NOT_COVERED_HEADING}" in note and "v1.7; v1.8; v1.9." in note and "3 more item(s)" in note


def test_a_list_the_hits_spread_across_is_expanded_when_sub_lists_rank_first(pipeline):
    """The pydantic case: two releases' 'What's Changed' sub-lists ranked above the changelog itself."""
    whole = ("log.md", "Log", [(f"v1.{i}", f"c{i}") for i in range(6)])
    sub_a = ("log.md", "Log > v1.0", [("Fixed", "a1"), ("Added", "a2")])
    sub_b = ("log.md", "Log > v1.1", [("Fixed", "b1"), ("Added", "b2")])
    pipeline.rank_lists = lambda question, candidates: [sub_a, sub_b, whole]
    hits = [_hit(f"Log > v1.{i}") for i in range(5)]
    assert pipeline.pick_lists("List every release", hits) == [sub_a, whole]
    # Unrelated lists that merely have many hits are not swapped in.
    other = ("readme.md", "Readme", [("Install", "r1"), ("Usage", "r2")])
    other_b = ("readme.md", "Readme > Usage", [("Load", "u1"), ("Parse", "u2")])
    pipeline.rank_lists = lambda question, candidates: [other, other_b, whole]
    assert pipeline.pick_lists("List every use case", hits) == [other, other_b]


def test_the_note_names_what_the_answered_list_could_not_fit(pipeline):
    from test_lists import GUIDE, overlap_scorer
    pipeline.settings = dataclasses.replace(pipeline.settings, list_answers="checklist", list_char_budget=3000)
    pipeline.list_scorer = overlap_scorer
    (pipeline.settings.docs_dir / "guide.md").write_text(GUIDE, encoding="utf-8")
    pipeline.ingest()
    question = "Describe every module"
    hits = pipeline.retrieve(question, k=4)
    plan = pipeline._list_plans[question]
    modules = next(i for i, lst in enumerate(plan) if lst[1].endswith("Modules"))
    missing = items_without_passages(plan[modules], hits)
    assert missing  # a budget this small can't fit all 8 modules

    class Picks:
        def chat(self, model, messages, options, stream=False):
            prompt = messages[-1]["content"]
            if "single letter" in prompt:
                lines = [line for line in prompt.splitlines() if line[:3] in ("A: ", "B: ")]
                reply = next(line[0] for line in lines if "Modules:" in line)
            else:
                reply = "### Module A — Part A\n- Module A owns component A [1]."
            yield {"message": {"content": reply}, "done": True}

    pipeline._llm = Picks()
    answer = pipeline.generate(question, hits)
    assert answer.endswith("; ".join(missing) + ".")


def test_insert_sections_keeps_document_order_and_stays_before_the_note():
    from rag.citations import same_heading
    from rag.filler import insert_sections
    answer = f"### v3\n- c [3].\n\n### v1\n- a [1].\n\n### {NOT_COVERED_HEADING}\nv0."
    fixed = insert_sections(answer, [("v4", "- d [4]."), ("v2", "- b [2].")], ["v4", "v3", "v2", "v1", "v0"],
                            same_heading, before=NOT_COVERED_HEADING)
    assert [line for line in fixed.splitlines() if line.startswith("### ")] == \
        ["### v4", "### v3", "### v2", "### v1", f"### {NOT_COVERED_HEADING}"]


def test_follows_list_needs_most_headings_to_be_exact_item_names():
    from rag.pipeline import ListItem, follows_list
    items = [ListItem(n, n, [1]) for n in ["Usage", "Types", "Questions", "Example", "License"]]
    assert follows_list(["Usage", "Types", "License", "Extra"], items)
    assert not follows_list(["Questions"], items)  # one heading: the FAQ answer, not this list
    assert not follows_list(["An example with two subcommands", "Run the upgraded example", "Recap"], items)


def test_skipped_sections_are_written_and_inserted(pipeline):
    """charset-normalizer: given 35 releases with passages, the model wrote 11 and skipped 24."""
    from test_filler import SummarizingLLM, _list_setup
    question, hits = _list_setup(pipeline)
    plan = pipeline._list_plans[question]
    modules = next(lst for lst in plan if lst[1].endswith("Modules"))
    pipeline._answered_lists[question] = modules
    items = pipeline.list_items(question, hits, lists=[modules])
    written = [i.name for i in items if not i.name.endswith(("B — Part B", "C — Part C"))]
    answer = "\n\n".join(f"### {n}\n- {n} is described here [1]." for n in written)
    pipeline._llm = SummarizingLLM()
    fixed, repaired = pipeline.repair_filler(question, hits, answer)
    headings = [line[4:] for line in fixed.splitlines() if line.startswith("### ")]
    assert repaired == 2 and headings == [i.name for i in items]  # the two skipped ones, in document order


def test_no_note_when_the_chooser_keeps_both_lists(pipeline):
    from test_list_answers import FakeLLM, _setup
    question, hits = _setup(pipeline, "checklist")
    pipeline.settings = dataclasses.replace(pipeline.settings, list_char_budget=3000)
    hits = pipeline.retrieve(question, k=4)
    pipeline._llm = FakeLLM()  # unusable chooser replies: both lists kept, so it's unclear which list to complete
    assert NOT_COVERED_HEADING not in pipeline.generate(question, hits)
