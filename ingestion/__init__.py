"""
Ingestion package for CodeChat vector database

This package contains all tools for ingesting code into the vector database:
- vectordb_sync.py: Main sync script (CI/CD automated)
- llama_chunker.py: LlamaIndex-based code chunking
"""

from .llama_chunker import LlamaChunker

__all__ = ['LlamaChunker']
