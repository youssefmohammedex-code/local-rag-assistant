"""PDF structure recovery: headings from fonts, running headers/footers removed, sections across pages."""

from conftest import make_pdf, make_styled_pdf

from rag.loaders import load_documents
from rag.pdf import read_pdf

BODY = "This is ordinary body text that explains the section in a full sentence."


def guide_pages():
    return [
        [("bold", 18, "User Guide"), ("regular", 10, BODY),
         ("bold", 14, "Installation"), ("regular", 10, "Run pip install guide."),
         ("bold", 12, "From source"), ("regular", 10, BODY)],
        [("regular", 10, "The source build continues on page two."),
         ("bold", 14, "Usage"), ("regular", 10, BODY), ("mono", 9, "# a comment in code, not a heading")],
    ]


def test_headings_by_font_size_with_levels_and_title(tmp_path):
    make_styled_pdf(tmp_path / "g.pdf", guide_pages())
    pieces = read_pdf(tmp_path / "g.pdf")
    paths = [section for _, section, _ in pieces]
    assert paths == ["User Guide", "User Guide > Installation", "User Guide > Installation > From source",
                     "User Guide > Installation > From source", "User Guide > Usage"]


def test_section_continues_across_page_break_with_page_numbers(tmp_path):
    make_styled_pdf(tmp_path / "g.pdf", guide_pages())
    source_build = [(page, body) for page, section, body in read_pdf(tmp_path / "g.pdf")
                    if section.endswith("From source")]
    assert [p for p, _ in source_build] == [1, 2]
    assert "continues on page two" in source_build[1][1]


def test_code_in_a_smaller_font_is_not_a_heading(tmp_path):
    make_styled_pdf(tmp_path / "g.pdf", guide_pages())
    usage = next(body for _, section, body in read_pdf(tmp_path / "g.pdf") if section.endswith("Usage"))
    assert "# a comment in code" in usage


def test_running_header_and_page_footer_are_removed(tmp_path):
    pages = [[("bold", 14, f"Chapter {i}"), ("regular", 10, BODY)] for i in range(1, 5)]
    make_styled_pdf(tmp_path / "r.pdf", pages, header="ACME Manual - Confidential")
    text = "\n".join(body for _, _, body in read_pdf(tmp_path / "r.pdf"))
    assert "ACME Manual" not in text and "Page 3" not in text
    assert text.count(BODY) == 4


def test_bold_body_size_lines_are_lowest_level_headings(tmp_path):
    make_styled_pdf(tmp_path / "b.pdf", [[("bold", 16, "Report"), ("regular", 10, BODY),
                                          ("bold", 10, "Key findings"), ("regular", 10, BODY),
                                          ("regular", 10, BODY), ("regular", 10, BODY)]])
    assert [s for _, s, _ in read_pdf(tmp_path / "b.pdf")] == ["Report", "Report > Key findings"]


def test_single_font_pdf_falls_back_to_numbered_headings(tmp_path):
    lines = [("regular", 10, t) for t in ["1 Introduction", BODY, BODY, "2 Method", BODY,
                                          "2.1 Data", BODY, BODY, "3 Results", BODY]]
    make_styled_pdf(tmp_path / "n.pdf", [lines])
    paths = [s for _, s, _ in read_pdf(tmp_path / "n.pdf")]
    assert paths == ["1 Introduction", "2 Method", "2 Method > 2.1 Data", "3 Results"]


def test_single_font_pdf_without_numbering_stays_page_based(tmp_path):
    make_pdf(tmp_path / "plain.pdf", ["Install", "pip install it.", "Usage", "Run it."])
    pages = load_documents(tmp_path)
    assert [(p.section, p.page) for p in pages] == [("", 1)]


def test_prose_heavy_bold_document_is_not_all_headings(tmp_path):
    # If most lines look like headings, the body-size guess is wrong: don't invent structure.
    make_styled_pdf(tmp_path / "x.pdf", [[("bold", 12, f"Short line {i}") for i in range(10)]
                                         + [("regular", 10, BODY)]])
    assert all(section == "" for _, section, _ in read_pdf(tmp_path / "x.pdf"))


def test_pdf_pages_have_unique_parts_for_chunk_ids(tmp_path):
    make_styled_pdf(tmp_path / "g.pdf", guide_pages())
    pages = load_documents(tmp_path)
    assert len({p.part for p in pages}) == len(pages)
