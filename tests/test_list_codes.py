"""List questions that name a code ("every 2.14 release") ask for a subset of a long list."""

import dataclasses

import pytest
from test_lists import overlap_scorer

from rag.pipeline import candidate_lists, filter_by_codes, names_code
from rag.store import Hit


@pytest.mark.parametrize("name,code,expected", [
    ("v2.14.4 (2026-11-19)", "2.14", True),
    ("v2.14.0b1 (2026-08-06)", "2.14", True),
    ("2.14", "2.14", True),
    ("v12.14.0", "2.14", False),   # different version
    ("v2.140", "2.14", False),     # longer number
    ("v2.14.0b1", "v2.14.0", False),  # a pre-release is not the release
    ("v2.14.0 (2026-10-08)", "v2.14.0", True),
    ("Release notes", "2.14", False),
])
def test_names_code_matches_whole_codes_and_prefixes_only(name, code, expected):
    assert names_code(name, code) is expected


KIDS = [("v2.14.1", "a"), ("v2.14.0", "b"), ("v2.13.1", "c"), ("v2.13.0", "d")]


def test_filter_keeps_only_named_items_when_two_or_more_match():
    assert filter_by_codes(KIDS, {"2.14"}) == KIDS[:2]


def test_filter_leaves_the_list_alone_for_a_single_match_or_no_codes():
    assert filter_by_codes(KIDS, {"v2.14.0"}) == KIDS  # one release: the question is about it, not a subset
    assert filter_by_codes(KIDS, set()) == KIDS


def test_candidate_lists_narrow_a_changelog():
    outline = [(f"c::s{i}::c0", f"Changelog > {name}") for i, (name, _) in enumerate(KIDS)]
    hits = [Hit("t", "c.md", 1, 0.5, section="Changelog > v2.14.0")]
    nodes = candidate_lists(hits, {"c.md": outline}, {"2.14"})
    assert [name for name, _ in nodes[("c.md", "Changelog")]] == ["v2.14.1", "v2.14.0"]


CHANGELOG = "# Changelog\n\n" + "\n\n".join(  # newest first, like most changelogs
    f"## v3.{minor}.{patch}\n\nThis release fixed parser bug {minor}{patch} in the tokenizer module."
    for minor in range(9, -1, -1) for patch in range(5, -1, -1)
)


def test_old_releases_in_a_long_changelog_are_retrieved(pipeline):
    """60 releases, more than the context budget holds; the question wants the 6 old 3.1 ones near
    the end. Expanding the whole changelog fills the budget in document order and never reaches them."""
    pipeline.settings = dataclasses.replace(pipeline.settings, list_char_budget=6000)
    pipeline.list_scorer = overlap_scorer
    (pipeline.settings.docs_dir / "changelog.md").write_text(CHANGELOG, encoding="utf-8")
    pipeline.ingest()
    sections = {h.section for h in pipeline.retrieve("List every 3.1 release.", k=2)}
    assert {f"Changelog > v3.1.{p}" for p in range(6)} <= sections
