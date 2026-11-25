"""List questions ("describe every module") get one passage per sub-section."""

import dataclasses

import pytest

from rag.pipeline import is_list_question, pick_list_parent
from rag.store import Hit


@pytest.mark.parametrize("question,expected", [
    ("Describe every module and its owner.", True),
    ("List the decisions.", True),
    ("What are all the end-to-end flows?", True),
    ("Summarize each section.", True),
    ("Who owns the Backend API module?", False),
    ("What did decision D13 decide?", False),
    ("Is the overall plan realistic?", False),  # "overall" is not "all"
])
def test_list_question_detection(question, expected):
    assert is_list_question(question) is expected


def hit(section, source="guide.md"):
    return Hit("text", source, 1, 0.5, section=section)


CARDS = "Guide > 7. Module cards"


def test_parent_with_matching_sub_headings_wins():
    hits = [hit(f"{CARDS} > Module A — Engine"), hit("Guide > 2. Overview"), hit(f"{CARDS} > Module B — World"),
            hit("Guide > 3. Testing"), hit(f"{CARDS} > Module C — API")]
    assert pick_list_parent("Describe every module", hits) == ("guide.md", CARDS)


def test_heading_named_in_question_beats_bigger_section():
    decisions = [hit(f"Guide > 15. Working decisions > D{i} — x") for i in range(1, 8)]
    flows = [hit("Guide > 8. End-to-end flows > 8.1 Manual"), hit("Guide > 8. End-to-end flows > 8.2 AI")]
    assert pick_list_parent("What are all the end-to-end flows?", decisions + flows)[1] == "Guide > 8. End-to-end flows"


def test_whole_document_can_be_the_list():
    hits = [hit("Guide > 1. Scope"), hit("Guide > 2. Architecture"), hit("Guide > 3. Rules")]
    assert pick_list_parent("Summarize every section of the guide", hits) == ("guide.md", "Guide")


def test_needs_two_distinct_sub_sections():
    assert pick_list_parent("List everything", [hit("Guide > 1. Scope"), hit("Guide > 1. Scope")]) is None


def _write(pipeline, text):
    (pipeline.settings.docs_dir / "guide.md").write_text(text, encoding="utf-8")
    pipeline.ingest()


GUIDE = "# Guide\n\n## Intro\nOverview text.\n\n## Modules\n" + "".join(
    f"\n### Module {c} — Part {c}\nModule {c} owns component {c}. " + "Detail. " * 40 + "\n" for c in "ABCDEFGH")


def test_list_question_returns_every_sub_section_in_document_order(pipeline):
    _write(pipeline, GUIDE)
    hits = pipeline.retrieve("Describe every module", k=4)
    sections = [h.section for h in hits]
    expected = [f"Guide > Modules > Module {c} — Part {c}" for c in "ABCDEFGH"]
    assert set(expected) <= set(sections)
    expanded = [s for s, h in zip(sections, hits, strict=True) if h.via == "section"]
    assert expanded == [s for s in expected if s in expanded]  # document order


def test_expansion_respects_the_character_budget(pipeline):
    pipeline.settings = dataclasses.replace(pipeline.settings, list_char_budget=3000)
    _write(pipeline, GUIDE)
    hits = pipeline.retrieve("Describe every module", k=4)
    assert sum(len(h.text) for h in hits) <= 3000 + 1000 * 3  # top hits + trimmed items
    assert any(h.text.endswith(" …") for h in hits if h.via == "section")


def test_precise_questions_are_unchanged(pipeline):
    _write(pipeline, GUIDE)
    assert all(h.via != "section" for h in pipeline.retrieve("Who owns component C?", k=4))


def test_expansion_can_be_switched_off(pipeline):
    pipeline.settings = dataclasses.replace(pipeline.settings, expand_lists=False)
    _write(pipeline, GUIDE)
    assert len(pipeline.retrieve("Describe every module", k=4)) == 4
