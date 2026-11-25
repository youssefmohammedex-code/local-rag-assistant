"""Local RAG assistant: offline question answering over your own documents."""

from rag.config import Settings
from rag.pipeline import Answer, RAGPipeline

__all__ = ["Settings", "RAGPipeline", "Answer"]
