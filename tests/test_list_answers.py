"""Generation strategies for list answers, tested with a fake model that records its prompts."""

import dataclasses
import re

from test_lists import GUIDE, overlap_scorer


class FakeLLM:
    """Records answer prompts in .prompts and list-choice prompts in .choices. `chooser` answers the
    choice prompts (default: an unusable reply, so the pipeline keeps both lists)."""

    def __init__(self, chooser=lambda prompt: "?"):
        self.prompts: list[str] = []
        self.choices: list[str] = []
        self.chooser = chooser

    def chat(self, model, messages, options, stream=False):
        prompt = messages[-1]["content"]
        if "single letter A or B" in prompt or "exactly as written before the colon" in prompt:
            self.choices.append(prompt)
            yield {"message": {"content": self.chooser(prompt)}, "done": True, "eval_count": 1, "eval_duration": 1e8}
            return
        self.prompts.append(prompt)
        sections = re.findall(r"### (.+)", prompt) or ["answer"]
        yield {"message": {"content": f"call {len(self.prompts)}: {', '.join(sections)}"}, "done": False}
        yield {"message": {"content": ""}, "done": True, "eval_count": 5, "eval_duration": 1e9}


def _setup(pipeline, strategy):
    pipeline.settings = dataclasses.replace(pipeline.settings, list_answers=strategy)
    pipeline.list_scorer = overlap_scorer
    (pipeline.settings.docs_dir / "guide.md").write_text(GUIDE, encoding="utf-8")
    pipeline.ingest()
    pipeline._llm = FakeLLM()
    question = "Describe every module"
    return question, pipeline.retrieve(question, k=4)


def test_list_items_follow_the_expanded_list(pipeline):
    question, hits = _setup(pipeline, "per-item")
    items = pipeline.list_items(question, hits)
    module_items = [i.name for i in items if i.path.startswith("Guide > Modules > ")]
    assert module_items == [f"Module {c} — Part {c}" for c in "ABCDEFGH"]
    for item in items:
        assert all(hits[i - 1].section.startswith(item.path) for i in item.passages)


def test_per_item_calls_see_only_their_items_passages(pipeline):
    question, hits = _setup(pipeline, "per-item")
    items = pipeline.list_items(question, hits)
    answer = pipeline.generate(question, hits)
    prompts = pipeline._llm.prompts
    assert len(prompts) == -(-len(items) // 3)  # ceil(items / 3) calls
    first_batch = {i for item in items[:3] for i in item.passages}
    shown = {int(n) for n in re.findall(r'<passage id="(\d+)"', prompts[0])}
    assert shown == first_batch  # global passage ids kept, nothing from other items
    assert answer.count("call ") == len(prompts)
    assert pipeline.last_stats.tokens == 5 * len(prompts)


def test_checklist_is_one_call_listing_every_section(pipeline):
    question, hits = _setup(pipeline, "checklist")
    items = pipeline.list_items(question, hits)
    pipeline.generate(question, hits)
    prompts = pipeline._llm.prompts
    assert len(prompts) == 1
    assert all(f"### {item.name}" in prompts[0] for item in items)
    assert all(f"### Module {c} — Part {c}" in prompts[0] for c in "ABCDEFGH")


def test_single_strategy_and_concise_mode_use_one_plain_call(pipeline):
    question, hits = _setup(pipeline, "single")
    pipeline.generate(question, hits)
    assert len(pipeline._llm.prompts) == 1 and "### " not in pipeline._llm.prompts[0]
    pipeline.settings = dataclasses.replace(pipeline.settings, list_answers="per-item")
    pipeline._llm = FakeLLM()
    pipeline.generate(question, hits, mode="concise")
    assert len(pipeline._llm.prompts) == 1


def _letter_of(prompt, words):
    """Fake model that picks, by content, the option line containing `words` (order-independent)."""
    for line in prompt.splitlines():
        if line[:3] in ("A: ", "B: ") and words in line:
            return line[0]
    return "?"


def test_checklist_answers_from_the_list_both_orders_agree_on(pipeline):
    question, hits = _setup(pipeline, "checklist")
    lists = pipeline._list_plans[question]
    assert len(lists) == 2
    wanted = next(lst for lst in lists if lst[1].endswith("Modules"))
    pipeline._llm = FakeLLM(chooser=lambda p: _letter_of(p, "Modules:"))
    pipeline.generate(question, hits)
    llm = pipeline._llm
    assert len(llm.choices) == 2  # asked in both orders
    assert llm.choices[0].index("A: ") < llm.choices[0].index(wanted[1]) and \
        llm.choices[1].index("B: ") < llm.choices[1].index(wanted[1])  # the order really was swapped
    prompt = llm.prompts[0]
    wanted_ids = {chunk_id for _, chunk_id in wanted[2]}
    shown = {int(n) for n in re.findall(r'<passage id="(\d+)"', prompt)}
    expected = {i for i, h in enumerate(hits, start=1) if h.via != "section" or h.id in wanted_ids}
    assert shown == expected and len(shown) < len(hits)  # the other list's expanded passages are gone
    for name, _ in wanted[2]:
        assert f"### {name}" in prompt


def test_checklist_shows_passages_in_document_order_with_their_real_ids(pipeline):
    question, hits = _setup(pipeline, "checklist")
    pipeline.generate(question, hits)
    shown = [int(n) for n in re.findall(r'<passage id="(\d+)"', pipeline._llm.prompts[0])]
    assert sorted(shown) == list(range(1, len(hits) + 1))  # every passage, ids unchanged
    positions = [int(re.search(r"::s(\d+)::", hits[i - 1].id).group(1)) for i in shown]
    assert positions == sorted(positions)
    assert shown != sorted(shown)  # retrieval order really was different


def test_checklist_keeps_both_lists_when_the_orders_disagree(pipeline):
    question, hits = _setup(pipeline, "checklist")
    pipeline._llm = FakeLLM(chooser=lambda p: "A")  # position bias: always the first option
    pipeline.generate(question, hits)
    shown = re.findall(r'<passage id="(\d+)"', pipeline._llm.prompts[0])
    assert len(shown) == len(hits)


def test_answers_have_a_length_cap(pipeline):
    """A model stuck repeating itself once generated for 55 minutes: every answer call is capped."""
    request = pipeline._request("q", [], "detailed")
    assert request["options"]["num_predict"] == pipeline.settings.max_answer_tokens > 0


def test_seed_is_sent_only_when_set(pipeline):
    assert "seed" not in pipeline._request("q", [], "detailed")["options"]
    pipeline.settings = dataclasses.replace(pipeline.settings, seed=7, temperature=0)
    options = pipeline._request("q", [], "detailed")["options"]
    assert options["seed"] == 7 and options["temperature"] == 0 and options["num_ctx"] == pipeline.settings.num_ctx


def test_heading_question_decides_when_the_letters_follow_position(pipeline):
    """Typer's example upgrade: the model answered "B" in both orders, i.e. by position."""
    question, hits = _setup(pipeline, "checklist")
    lists = pipeline._list_plans[question]
    wanted = next(lst for lst in lists if lst[1].endswith("Modules"))

    def chooser(prompt):
        return "B" if "single letter" in prompt else "Modules"  # the last part of the wanted heading

    pipeline._llm = FakeLLM(chooser=chooser)
    pipeline.generate(question, hits)
    assert len(pipeline._llm.choices) == 4  # two letter questions, then two heading questions
    assert pipeline.choose_list(question) == lists.index(wanted)


def test_letters_that_agree_are_never_second_guessed(pipeline):
    question, hits = _setup(pipeline, "checklist")
    pipeline._llm = FakeLLM(chooser=lambda p: _letter_of(p, "Modules:") if "single letter" in p else "Guide")
    pipeline.generate(question, hits)
    assert len(pipeline._llm.choices) == 2


def test_heading_matches():
    from rag.pipeline import heading_matches
    headings = ["python-dotenv > Other Use Cases", "python-dotenv"]
    assert heading_matches("Other Use Cases", headings) == 0
    assert heading_matches("'python-dotenv'.", headings) == 1
    assert heading_matches("python-dotenv > Other Use Cases", headings) == 0
    assert heading_matches("Serialize", ["orjson > Usage > Serialize > option", "orjson > Usage"]) is None
    assert heading_matches("A", headings) is None


def test_cold_calls_unload_the_model_after_every_call(pipeline):
    """A reused cached prompt changed a few words of an otherwise identical seeded answer."""
    seen = []

    class Recording:
        def chat(self, stream=False, **request):
            seen.append(request.get("keep_alive", "default"))
            yield {"message": {"content": "ok"}, "done": True}

    pipeline._llm = Recording()
    pipeline.generate("Who owns component C?", [])
    pipeline.settings = dataclasses.replace(pipeline.settings, cold_calls=True)
    pipeline.generate("Who owns component C?", [])
    assert seen == ["default", 0]


def test_unknown_strategy_is_rejected(pipeline):
    import pytest

    from rag import RAGPipeline
    with pytest.raises(ValueError, match="list_answers"):
        RAGPipeline(dataclasses.replace(pipeline.settings, list_answers="bogus"))


def test_checklist_lets_the_model_skip_unrelated_sections(pipeline):
    """Retrieval expands two candidate lists and one is often unrelated: the outline must not be
    a hard "exactly these headings" order, or the unrelated list gets forced into the answer."""
    question, hits = _setup(pipeline, "checklist")
    pipeline.generate(question, hits)
    prompt = pipeline._llm.prompts[0]
    assert "no other headings" not in prompt and "unrelated" in prompt


def test_precise_question_is_never_split(pipeline):
    _setup(pipeline, "per-item")
    hits = pipeline.retrieve("Who owns component C?", k=4)
    pipeline.generate("Who owns component C?", hits)
    assert len(pipeline._llm.prompts) == 1
