import pytest

from rag.bm25 import BM25, identifiers, is_identifier, reciprocal_rank_fusion, tokenize


def test_tokenizer_keeps_identifiers_whole_and_split():
    tokens = tokenize("Decision D13 replaces BR-05 in v2.1; see PYTHONHASHSEED.")
    assert {"d13", "br-05", "br", "05", "v2.1", "pythonhashseed", "decision"} <= set(tokens)
    assert "in" not in tokens  # stopword


@pytest.mark.parametrize("token,expected", [
    ("d13", True), ("br-05", True), ("v2.1", True), ("0.10.0", True), ("2026-10-08", True), ("3.9", True),
    ("500", False), ("module", False), ("end-to-end", False),
])
def test_identifier_detection(token, expected):
    assert is_identifier(token) is expected


def test_bm25_finds_exact_version_among_similar_ones():
    docs = [f"Changelog 0.10.{i}: notes for release 0.10.{i}" for i in range(6)]
    assert BM25(docs).top("What changed in 0.10.0?", 1) == [0]


def test_tokenizer_handles_arabic():
    assert "التسجيل" in tokenize("رسوم التسجيل المتأخر 500 جنيه")


def test_bm25_ranks_exact_identifier_first():
    docs = ["D21 releases assignments", "D13 sets ten-second ticks", "Ticks and assignments overview"]
    assert BM25(docs).top("What did D13 decide?", 3) == [1]


def test_bm25_prefers_rarer_terms_and_ignores_non_matches():
    docs = ["module module module owner", "module owner zelda", "unrelated text"]
    top = BM25(docs).top("zelda module", 3)
    assert top[0] == 1 and 2 not in top


def test_bm25_empty_corpus():
    assert BM25([]).top("anything", 5) == []


def test_rrf_rewards_agreement_between_rankings():
    fused = reciprocal_rank_fusion([["a", "b"], ["x", "b"]])
    assert fused[0] == "b"  # 2nd in both lists beats 1st in only one list
    assert set(fused) == {"a", "b", "x"}


def test_rrf_is_stable_for_ties():
    assert reciprocal_rank_fusion([["a", "b"], ["b", "a"]]) == ["a", "b"]


def _write(pipeline, **files):
    for name, text in files.items():
        (pipeline.settings.docs_dir / name).write_text(text, encoding="utf-8")
    pipeline.ingest()


def test_hybrid_finds_exact_id_that_vector_search_ranks_randomly(pipeline):
    # The fake embedder gives unrelated vectors, so only keyword search can find the right chunk.
    _write(pipeline, **{f"d{i}.md": f"# Decisions\n\n## D{i}\nDecision number {i} text." for i in range(1, 16)})
    hits = pipeline.retrieve("What did D13 decide?", k=3, mode="hybrid")
    assert hits[0].section == "Decisions > D13"
    assert hits[0].via in {"keyword", "both"}


def test_identifiers_are_whole_codes_only():
    assert identifiers("What's in pydantic v2.14.0 and D13?") == {"v2.14.0", "d13"}


def test_heading_with_exact_version_beats_many_similar_versions(pipeline):
    # Many near-identical changelog entries appear in both rankings; only one is named 0.10.0.
    changelog = "# Tool\n\n## Changelog\n" + "".join(
        f"\n### 0.10.{i}\n- Changed things in release 0.10.{i} of the tool.\n" for i in range(1, 8))
    changelog += "\n### 0.10.0\n- Add support for trailing comments.\n"
    _write(pipeline, **{"tool.md": changelog})
    hits = pipeline.retrieve("What changed in tool 0.10.0?", k=3, mode="hybrid")
    assert hits[0].section == "Tool > Changelog > 0.10.0"


def test_vector_mode_still_available_and_labelled(pipeline):
    _write(pipeline, **{"a.md": "# A\n\nAlpha", "b.md": "# B\n\nBravo"})
    hits = pipeline.retrieve("Alpha", k=2, mode="vector")
    assert len(hits) == 2 and all(h.via == "meaning" for h in hits)


def test_hybrid_scores_are_cosine_similarities(pipeline):
    _write(pipeline, **{"a.md": "# A\n\nAlpha", "b.md": "# B\n\nBravo"})
    assert all(-1.0 <= h.score <= 1.0 for h in pipeline.retrieve("Alpha", k=2))


def test_keyword_index_refreshes_after_reingest(pipeline):
    _write(pipeline, **{"a.md": "# A\n\nAlpha"})
    assert pipeline.retrieve("zebra", k=1, mode="hybrid")[0].via == "meaning"
    _write(pipeline, **{"z.md": "# Z\n\nzebra crossing"})
    assert pipeline.retrieve("zebra", k=1, mode="hybrid")[0].source == "z.md"


def test_unknown_search_mode_rejected(pipeline):
    _write(pipeline, **{"a.md": "# A\n\nAlpha"})
    with pytest.raises(ValueError):
        pipeline.retrieve("Alpha", k=1, mode="fuzzy")
