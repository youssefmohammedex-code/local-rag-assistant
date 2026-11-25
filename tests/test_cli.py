import pytest

import rag.cli as cli


class OllamaDownPipeline:
    def __init__(self, settings):
        raise ConnectionError("Failed to connect to Ollama.")


def test_cli_shows_friendly_error_instead_of_traceback(monkeypatch):
    monkeypatch.setattr(cli, "RAGPipeline", OllamaDownPipeline)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["ingest"])
    assert str(exit_info.value).startswith("Error: Can't reach Ollama.")


def test_cli_rejects_unknown_mode(capsys):
    with pytest.raises(SystemExit):
        cli.main(["ask", "--mode", "verbose", "hi"])
    assert "invalid choice" in capsys.readouterr().err
