"""
Shared library utilities for CodeChat

This package contains shared utilities used across ingestion and scripts:
- pine.py: Pinecone client wrapper
"""

from .pine import PineconeClient

__all__ = ['PineconeClient']
