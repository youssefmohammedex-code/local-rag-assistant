"""Stopping an answer once the model starts repeating itself."""

import itertools

from rag.loops import LoopDetector
from rag.pipeline import LOOP_NOTE


def _stream(text, detector=None, chunk=5):
    """Feed `text` in small chunks, as tokens arrive; return the chunk index it stopped at."""
    detector = detector or LoopDetector()
    for i in range(0, len(text), chunk):
        if detector.feed(text[i:i + chunk]):
            return i
    return None


SECTIONS = ["Installation", "Requirements", "Dependencies"]


def test_whole_sections_cycling_with_counting_citations_is_a_loop():
    # The 8.8-hour benchmark answer: the same sections over and over, citations counting up.
    n = itertools.count(1)
    text = "".join(f"### {s}\n- The library can be installed and used in many ways [{next(n)}].\n\n"
                   for _ in range(10) for s in SECTIONS)
    stop = _stream(text)
    assert stop is not None and stop < len(text) // 2


def test_one_line_repeating_is_a_loop():
    text = "".join(f"- Add `find_dotenv` method that will try to find a `.env` file. [{i}]\n" for i in range(50, 0, -1))
    assert _stream(text) is not None


def test_the_same_bullet_under_different_headings_is_not_a_loop():
    # Filler answers repeat one bullet under every release; the headings keep them apart.
    text = "".join(f"### [1.{i}.0] - 2024-01-{i + 1:02}\n- No specific changes listed [{i}].\n\n" for i in range(30))
    assert _stream(text) is None


def test_short_repeated_lines_are_not_a_loop():
    text = "| a | b |\n|---|---|\n" + "}\n" * 10 + "---\n" * 10
    assert _stream(text) is None


def test_two_copies_are_not_enough():
    block = "@app.command()\ndef hello(name: str):\n    print(f'Hello {name}')\n"
    assert _stream(block * 2) is None
    assert _stream(block * 3) is not None


class EndlessLLM:
    """A model stuck in a loop: it never sends `done`. Records whether the stream was closed."""

    def __init__(self):
        self.closed = False

    def chat(self, model, messages, options, stream=False):
        try:
            for i in itertools.count():
                yield {"message": {"content": f"- The same sentence, over and over again [{i}].\n"}, "done": False}
        finally:
            self.closed = True


def test_a_looping_answer_is_stopped_and_the_stream_closed(pipeline):
    pipeline._llm = EndlessLLM()
    answer = pipeline.generate("Who owns component C?", [])
    assert answer.endswith(LOOP_NOTE.strip())
    assert answer.count("over and over") == 3
    assert pipeline._llm.closed  # so Ollama cancels the generation
