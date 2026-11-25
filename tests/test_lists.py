"""List questions ("describe every module") get one extra passage per sub-section."""

import dataclasses

import pytest

from rag.bm25 import tokenize
from rag.pipeline import candidate_lists, describe_list, is_list_question
from rag.store import Hit, children_in_outline


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


OUTLINE = [  # (chunk id, section) in document order
    ("g::s1::c0", "Guide"),
    ("g::s2::c0", "Guide > Modules"),
    ("g::s3::c0", "Guide > Modules > Module A"), ("g::s3::c1", "Guide > Modules > Module A"),
    ("g::s4::c0", "Guide > Modules > Module B > Details"),
    ("g::s5::c0", "Guide > Testing"),
]


def test_children_in_outline_are_direct_and_ordered():
    assert children_in_outline(OUTLINE, "Guide > Modules") == [("Module A", "g::s3::c0"),
                                                              ("Module B", "g::s4::c0")]
    assert [n for n, _ in children_in_outline(OUTLINE, "Guide")] == ["Modules", "Testing"]


def test_top_level_of_a_document_without_a_title():
    outline = [("r::s1::c0", "Installation"), ("r::s2::c0", "Usage > Basics"), ("r::s3::c0", "License")]
    assert [n for n, _ in children_in_outline(outline, "")] == ["Installation", "Usage", "License"]


def test_candidate_lists_include_own_section_ancestors_and_top_level():
    hits = [Hit("t", "g.md", 1, 0.5, section="Guide > Modules > Module A")]
    keys = set(candidate_lists(hits, {"g.md": OUTLINE}))
    # "Guide > Modules" (2 kids) and "Guide" (2 kids) qualify; "Module A" has none; top level has 1 ("Guide").
    assert keys == {("g.md", "Guide > Modules"), ("g.md", "Guide")}


def test_summary_uses_full_path_so_the_document_name_does_not_decide():
    kids = [("dataclass", "x"), ("datetime", "y")]
    assert describe_list("orjson.md", "orjson > Types", kids) == "orjson > Types: dataclass; datetime"
    assert describe_list("notes/README 0123456789abcdef0123456789abcdef.md", "", kids).startswith("README: ")


def overlap_scorer(question, texts):
    """Stand-in for embedding similarity in tests: shared words."""
    q = set(tokenize(question))
    return [len(q & set(tokenize(t))) for t in texts]


GUIDE = "# Guide\n\n## Intro\nOverview text.\n\n## Modules\n" + "".join(
    f"\n### Module {c} — Part {c}\nModule {c} owns component {c}. " + "Detail. " * 40 + "\n" for c in "ABCDEFGH")


def _setup(pipeline, **settings):
    pipeline.settings = dataclasses.replace(pipeline.settings, **settings)
    pipeline.list_scorer = overlap_scorer
    (pipeline.settings.docs_dir / "guide.md").write_text(GUIDE, encoding="utf-8")
    pipeline.ingest()


def test_list_question_gets_every_sub_section_in_document_order(pipeline):
    _setup(pipeline)
    hits = pipeline.retrieve("Describe every module", k=4)
    sections = {h.section for h in hits}
    assert {f"Guide > Modules > Module {c} — Part {c}" for c in "ABCDEFGH"} <= sections
    expanded = [h.section for h in hits if h.via == "section"]
    assert expanded == sorted(expanded)  # document order (A..H)


def test_expansion_is_additive_never_drops_normal_results(pipeline):
    _setup(pipeline)
    normal = pipeline.store.search("Describe every module", 4, "hybrid")
    expanded = pipeline.retrieve("Describe every module", k=4)
    assert [h.id for h in expanded[:4]] == [h.id for h in normal]


def test_expansion_respects_the_character_budget(pipeline):
    _setup(pipeline, list_char_budget=2000)
    hits = pipeline.retrieve("Describe every module", k=2)
    assert sum(len(h.text) for h in hits) <= 2000
    items = [h for h in hits if h.via == "section"]
    assert items and all(h.text.endswith(" …") for h in items)  # trimmed to fit


def test_precise_questions_are_unchanged(pipeline):
    _setup(pipeline)
    assert all(h.via != "section" for h in pipeline.retrieve("Who owns component C?", k=4))


def test_expansion_can_be_switched_off(pipeline):
    _setup(pipeline, expand_lists=False)
    assert len(pipeline.retrieve("Describe every module", k=4)) == 4
