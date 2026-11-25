from eval.run_eval import rank_of_expected
from rag.loaders import load_documents
from rag.pipeline import build_prompt
from rag.store import Hit

HITS = [
    Hit("Grades use a 4.0 scale.", "sample/student-handbook.md", 1, 0.81),
    Hit("Goggles are mandatory.", "sample/lab-safety-rules.md", 1, 0.62),
]


def test_prompt_numbers_passages_and_cites_sources():
    prompt = build_prompt("What scale?", HITS)
    assert '<passage id="1" source="student-handbook" page="1">' in prompt
    assert '<passage id="2" source="lab-safety-rules" page="1">' in prompt
    assert "Question: What scale?" in prompt
    assert prompt.index("</passage>") < prompt.index("Question:")


def test_prompt_uses_section_when_available():
    hit = Hit("Owns the engine.", "Guide 421ffbd20ecf819b81b9f89e8cdb422a.md", 1, 0.7, section="Guide > Module A")
    assert '<passage id="1" source="Guide" section="Guide > Module A">' in build_prompt("q", [hit])


def test_prompt_modes_change_length_instructions():
    assert "thorough" in build_prompt("q", HITS, mode="detailed")
    assert "1-3 sentences" in build_prompt("q", HITS, mode="concise")


def test_prompt_without_hits():
    assert "(no context found)" in build_prompt("anything?", [])


def test_rank_of_expected():
    assert rank_of_expected(HITS, "student-handbook.md", None) == 1
    assert rank_of_expected(HITS, "lab-safety-rules.md", None) == 2
    assert rank_of_expected(HITS, "lab-safety-rules.md", 5) is None
    assert rank_of_expected(HITS, "missing.md", None) is None


def test_loader_reads_text_files(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "notes.md").write_text("# Title\nSome content", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    pages = load_documents(tmp_path)
    assert [(p.source, p.page) for p in pages] == [("sub/notes.md", 1)]
