"""Local RAG assistant: offline question answering over your own documents."""

from rag.config import Settings
from rag.errors import NoDocumentsError, friendly_error
from rag.pipeline import Answer, IngestReport, RAGPipeline

__all__ = ["Settings", "RAGPipeline", "Answer", "IngestReport", "NoDocumentsError", "friendly_error"]
