"""
Simplified Unified Chunker using LlamaIndex

Replaces:
- SmartChunker (2327 lines)
- chunking/ directory (51 files)
- Custom chunking functions in vectordb_sync.py

Single source of truth for all chunking operations.
"""

import os
from pathlib import Path
from typing import List, Dict, Any

from llama_index.core.node_parser import (
    CodeSplitter,
    MarkdownNodeParser,
    JSONNodeParser,
    SentenceSplitter,
)
from llama_index.core import Document
from llama_index.core.schema import BaseNode
import json
import yaml


class LlamaChunker:
    """
    Simplified chunker using LlamaIndex node parsers

    Provides unified chunking for both:
    - CI/CD incremental sync (vectordb_sync.py)
    - Bulk ingestion (vectordb_sync.py with empty tree commit)

    No file caching needed - git handles change detection
    """

    def __init__(self):
        """Initialize LlamaIndex parsers"""

        # Code parser (tree-sitter based, AST-aware)
        self.code_splitter = CodeSplitter(
            language="python",  # Updated per file
            chunk_lines=50,
            chunk_lines_overlap=15,
            max_chars=2000
        )

        # Markdown parser (header-aware)
        self.markdown_parser = MarkdownNodeParser()

        # JSON/YAML parser (structure-aware)
        self.json_parser = JSONNodeParser()

        # Generic text parser (sentence-aware)
        self.sentence_splitter = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=200
        )

        # File extension to language mapping (for code splitting)
        self.language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.cjs': 'javascript',
            '.mjs': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'c_sharp',
            '.go': 'go',
            '.rb': 'ruby',
            '.php': 'php',
            '.rs': 'rust',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.kts': 'kotlin',
            '.scala': 'scala',
            '.sh': 'bash',
        }

        # Parser selection by extension
        self.code_exts = set(self.language_map.keys())
        self.markdown_exts = {'.md', '.mdx', '.txt', '.rst', '.adoc'}
        self.json_exts = {'.json', '.jsonl'}
        self.yaml_exts = {'.yaml', '.yml'}

    def chunk_file(self, file_path: str) -> List[str]:
        """
        Chunk a file into semantic segments

        Args:
            file_path: Path to file to chunk

        Returns:
            List of chunk strings (raw text content)
        """
        # Read file
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

        if not content.strip():
            return []

        # Get file extension
        ext = Path(file_path).suffix.lower()
        language = self.language_map.get(ext, '')

        # Create LlamaIndex Document
        document = Document(
            text=content,
            metadata={
                'file_path': str(file_path),
                'file_name': Path(file_path).name,
                'language': language,
            }
        )

        # Parse with appropriate parser
        try:
            nodes = self._parse_document(document, ext, language)
        except Exception as e:
            print(f"Parsing error for {file_path}: {e}")
            # Fallback to sentence splitter
            nodes = self.sentence_splitter.get_nodes_from_documents([document])

        # Extract text content from nodes
        chunks = [node.text for node in nodes if node.text.strip()]

        return chunks

    def chunk_file_detailed(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Chunk a file with detailed metadata (for ingestion.py compatibility)

        Args:
            file_path: Path to file to chunk

        Returns:
            List of chunk dictionaries with 'content', 'type', 'start_line', 'end_line', 'language'
        """
        # Read file
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            return [{
                'content': f"Error reading {os.path.basename(file_path)}: {e}",
                'type': 'error',
                'start_line': 0,
                'end_line': 0,
                'language': ''
            }]

        if not content.strip():
            return []

        # Get file extension and language
        ext = Path(file_path).suffix.lower()
        language = self.language_map.get(ext, '')

        # Create LlamaIndex Document
        document = Document(
            text=content,
            metadata={
                'file_path': str(file_path),
                'file_name': Path(file_path).name,
                'language': language,
            }
        )

        # Parse with appropriate parser
        try:
            nodes = self._parse_document(document, ext, language)
        except Exception as e:
            print(f"Parsing error for {file_path}: {e}")
            nodes = self.sentence_splitter.get_nodes_from_documents([document])

        # Convert to detailed format
        chunks = []
        for node in nodes:
            if not node.text.strip():
                continue

            chunk = {
                'content': node.text,
                'type': self._classify_chunk_type(node.text, language),
                'start_line': node.metadata.get('start_line', 0),
                'end_line': node.metadata.get('end_line', 0),
                'language': language
            }
            chunks.append(chunk)

        return chunks

    def _parse_document(self, document: Document, ext: str, language: str) -> List[BaseNode]:
        """
        Parse document using appropriate LlamaIndex parser

        Args:
            document: LlamaIndex Document object
            ext: File extension
            language: Programming language

        Returns:
            List of LlamaIndex nodes
        """
        if ext in self.code_exts and language:
            # Code-aware parsing (respects AST structure)
            self.code_splitter.language = language
            return self.code_splitter.get_nodes_from_documents([document])

        elif ext in self.markdown_exts:
            # Markdown-aware parsing (respects headers)
            return self.markdown_parser.get_nodes_from_documents([document])

        elif ext in self.json_exts:
            # JSON structure-aware parsing
            return self.json_parser.get_nodes_from_documents([document])

        elif ext in self.yaml_exts:
            # YAML → JSON → structure-aware parsing
            try:
                data = yaml.safe_load(document.text)
                json_text = json.dumps(data, indent=2)
                json_document = Document(
                    text=json_text,
                    metadata=document.metadata
                )
                return self.json_parser.get_nodes_from_documents([json_document])
            except yaml.YAMLError:
                # Fallback to sentence splitter if YAML is malformed
                return self.sentence_splitter.get_nodes_from_documents([document])

        else:
            # Generic sentence-based parsing
            return self.sentence_splitter.get_nodes_from_documents([document])

    def _classify_chunk_type(self, content: str, language: str) -> str:
        """
        Classify chunk type based on content

        Args:
            content: Chunk text
            language: Programming language

        Returns:
            Chunk type string
        """
        content_lower = content.lower()

        # Code classification
        if language:
            if any(kw in content_lower for kw in ['def ', 'function ', 'class ', 'fn ', 'func ']):
                return 'function_definition'
            elif any(kw in content_lower for kw in ['import ', 'from ', 'require', 'use ']):
                return 'imports'
            elif any(kw in content_lower for kw in ['if ', 'for ', 'while ', 'try:', 'catch']):
                return 'logic'
            else:
                return 'code'

        # Markdown classification
        elif content.strip().startswith('#'):
            return 'header'
        elif '```' in content:
            return 'code_block'

        return 'text'

    def get_chunking_stats(self) -> Dict[str, Any]:
        """
        Get statistics about supported languages

        Returns:
            Dictionary with statistics
        """
        return {
            'supported_languages': len(self.language_map),
            'code_languages': len(self.code_exts),
            'markdown_formats': len(self.markdown_exts),
            'parser_types': 3  # code, markdown, sentence
        }


# Convenience function for direct import compatibility
def chunk_file(file_path: str) -> List[str]:
    """
    Convenience function - chunk a file into text segments

    Args:
        file_path: Path to file

    Returns:
        List of chunk strings
    """
    chunker = LlamaChunker()
    return chunker.chunk_file(file_path)
