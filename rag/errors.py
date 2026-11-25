"""Project errors and user-facing error messages shared by the CLI and the app."""

from __future__ import annotations

import re

import ollama


class NoDocumentsError(ValueError):
    """Raised when indexing finds nothing readable; the existing index is left untouched."""


def friendly_error(exc: BaseException) -> str:
    """Turn common failures into a short, actionable message instead of a traceback."""
    text = str(exc)
    if isinstance(exc, ConnectionError) or "Failed to connect to Ollama" in text:
        return "Can't reach Ollama. Make sure it is running (open Ollama from the Start menu) and try again."
    if isinstance(exc, ollama.ResponseError) and "not found" in text.lower():
        match = re.search(r"""model ["']?([^"'\s]+)["']? not found""", text)
        if match:
            return f"The model `{match.group(1)}` isn't installed. Run `ollama pull {match.group(1)}` and try again."
        return f"Ollama couldn't find the requested model ({exc.error}). Pull it with `ollama pull <model>`."
    if isinstance(exc, (NoDocumentsError, FileNotFoundError, ValueError)):
        return text
    return f"Unexpected error ({type(exc).__name__}): {text}"
