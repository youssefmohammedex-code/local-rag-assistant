"""Citation checking by word overlap, with the cases found in real benchmark answers."""

from rag.citations import check_citations, claims, issues, summary

PASSAGES = [
    "Guide > Requirements\n\nPython 3.8+ is required. The package depends on h11 and certifi.",
    "Guide > Installation\n\nInstall with pip install httpcore. Optional extras add HTTP/2 and SOCKS support.",
]


def _status(answer, passages=PASSAGES, **kw):
    return [(c.status, c.best) for c in check_citations(answer, passages, **kw)]


def test_claims_split_sentences_and_keep_every_cited_id():
    answer = "### Setup\n- Python 3.8+ is required [1]. Install it with pip [2][1].\n- See both [1, 2]."
    assert claims(answer) == [("Python 3.8+ is required", [1], "Setup"),
                              ("Install it with pip", [2, 1], "Setup"),
                              ("See both", [1, 2], "Setup")]


def test_a_citation_written_before_the_bullet_text_covers_it():
    # pydantic long list: "- [22] Latest V1.10.26 release" under 13 releases went unchecked.
    assert claims("### v2.14.4\n- [22] Latest V1.10.26 release.\n- [5] Supports Python 3.14.") == \
        [("Latest V1.10.26 release.", [22], "v2.14.4"), ("Supports Python 3.14.", [5], "v2.14.4")]


def test_citations_inside_code_move_to_the_sentence_before_the_block():
    from rag.citations import move_code_citations
    answer = ("1. **Install**: install it with pip.\n   ```shell\n   pip install python-dotenv [1]\n   ```\n"
              "2. **Load**: call `load_dotenv`:\n   ```python\n   load_dotenv()  # sets os.environ [6]\n"
              "   print(os.environ['A'])  # example [6]\n   ```")
    assert move_code_citations(answer, 10) == (
        "1. **Install**: install it with pip. [1]\n   ```shell\n   pip install python-dotenv\n   ```\n"
        "2. **Load**: call `load_dotenv` [6]:\n   ```python\n   load_dotenv()  # sets os.environ\n"
        "   print(os.environ['A'])  # example\n   ```")


def test_a_citation_line_after_a_block_moves_up_and_real_code_is_untouched():
    from rag.citations import move_code_citations
    after = "Read the values:\n```python\nconfig = dotenv_values()\n```\n\n   [5]\n\nNext step."
    assert move_code_citations(after, 10) == "Read the values [5]:\n```python\nconfig = dotenv_values()\n```\n\n\nNext step."
    several = "Use OPT_SORT_KEYS to sort the keys:\n```python\norjson.dumps(d)\n```\n\n[1] [4] [6]"
    assert move_code_citations(several, 10) == "Use OPT_SORT_KEYS to sort the keys [1, 4, 6]:\n```python\norjson.dumps(d)\n```\n"
    # A bare "Example:" line is too short to check, so the step it belongs to gets the citation.
    step = "- `load_dotenv()` adds each key to os.environ.\n\n   Example:\n   ```python\n   load_dotenv()\n   ```\n   [6]"
    assert move_code_citations(step, 10).startswith("- `load_dotenv()` adds each key to os.environ. [6]\n\n   Example:\n")
    for code in ["Make a list:\n```python\nx = [1]\nf(a, [2])\n```",  # code, not citations
                 "```python\nprint(x)  # [3]\n```",  # no sentence before the block: left as written
                 "Run it:\n```shell\ncmd [42]\n```"]:  # not a passage id
        assert move_code_citations(code, 10) == code


def test_supported_miscited_and_unsupported():
    assert _status("The package depends on h11 and certifi [1].") == [("supported", None)]
    assert _status("Optional extras add HTTP/2 and SOCKS support [1].") == [("miscited", 2)]
    assert _status("It supports GET, POST, PUT and DELETE requests [2].") == [("unsupported", None)]


def test_numbers_must_match():
    # The benchmark answer that said 3.6 shared every other word with the passage that says 3.8.
    assert _status("Python 3.6+ is required by the package [1].") == [("unsupported", None)]


def test_single_letter_and_digit_codes_must_match():
    passages = ["Guide > Modules\n\nModule D owns component D and ships within 3 weeks."]
    assert _status("Module C owns component C [1].", passages) == [("unsupported", None)]
    assert _status("Module D ships within 2 weeks [1].", passages) == [("unsupported", None)]
    assert _status("Module D owns component D [1].", passages) == [("supported", None)]
    assert _status("A module owns a component within weeks [1].", passages) == [("supported", None)]


def test_digits_inside_a_longer_number_are_not_extra_codes():
    from rag.citations import content_words
    assert content_words("a GPA below 2.0 and 3,000 credits in 2 years.") == {"gpa", "2.0", "3,000", "credit",
                                                                            "year", "2"}


def test_short_claims_are_not_judged():
    assert check_citations("Yes [1].", PASSAGES) == []


RELEASES = [
    "Changelog > 0.14.6\n\n- Fix SOCKS support for http URLs. (#492)\n\n0.14.5\n- SOCKS proxy support. (#478)",
    "Changelog > 0.14.5\n\n- SOCKS proxy support. (#478)\n- Add proxy_auth argument. (#481)",
]
SECTIONS = ["Changelog > 0.14.6", "Changelog > 0.14.5"]
PATHS = {"Changelog", "Changelog > 0.14.6", "Changelog > 0.14.5"}


def test_a_change_moved_to_the_neighbouring_release_is_caught():
    # Passage 1 runs on into 0.14.5's changes; citing them for 0.14.6 is the model's typical mistake.
    answer = "### 0.14.6\n- SOCKS proxy support (#478) [1]."
    assert _status(answer, RELEASES, sections=SECTIONS, paths=PATHS) == [("miscited", 2)]
    assert _status(answer, RELEASES) == [("supported", None)]  # without section info it can't tell


def test_run_on_text_counts_for_the_section_it_belongs_to():
    under_its_heading = "### 0.14.5\n- SOCKS proxy support (#478) [1]."
    naming_its_version = "### Changes\n- 0.14.5: SOCKS proxy support (#478) [1]."
    for answer in (under_its_heading, naming_its_version):
        assert _status(answer, RELEASES, sections=SECTIONS, paths=PATHS) == [("supported", None)]


def test_a_passage_is_not_cut_at_its_own_sub_sections():
    passage = ["Changelog > 2.0\n\nFixed\n- Repair the parser for quoted values. (#12)"]
    paths = {"Changelog > 2.0", "Changelog > 2.0 > Fixed", "Changelog > 1.0 > Fixed"}
    answer = "### 2.0\n- Repair the parser for quoted values (#12) [1]."
    assert _status(answer, passage, sections=["Changelog > 2.0"], paths=paths) == [("supported", None)]


def test_summary_and_issue_lines():
    checks = check_citations("Python 3.8+ is required [1]. Optional extras add SOCKS support [1].", PASSAGES)
    assert summary(checks) == "1 of 2 cited sentences match their passage · 1 to check"
    assert issues(checks) == ["“Optional extras add SOCKS support” cites [1], but passage [2] matches it"]
    assert summary([]) == "" and issues([]) == []


def test_pipeline_checks_against_the_hits_it_answered_from(pipeline):
    from test_lists import GUIDE

    (pipeline.settings.docs_dir / "guide.md").write_text(GUIDE, encoding="utf-8")
    pipeline.ingest()
    hits = pipeline.store.search("Which part does Module C own?", 4, "hybrid")
    right = next(i for i, h in enumerate(hits, 1) if "owns component C" in h.text)
    wrong = next(i for i, h in enumerate(hits, 1) if "owns component C" not in h.text)
    checks = pipeline.check_citations(f"Module C owns component C [{right}]. Module C owns component C [{wrong}].", hits)
    assert [(c.status, c.best) for c in checks] == [("supported", None), ("miscited", right)]
