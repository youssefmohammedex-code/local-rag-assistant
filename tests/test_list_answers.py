"""Generation strategies for list answers, tested with a fake model that records its prompts."""

import dataclasses
import re

from test_lists import GUIDE, overlap_scorer


class FakeLLM:
    def __init__(self):
        self.prompts: list[str] = []

    def chat(self, model, messages, options, stream=False):
        prompt = messages[-1]["content"]
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
