import os
from typing import List, Dict, Any, Optional
import hashlib
from pathlib import Path

try:
    import pinecone
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    print("❌ Pinecone package not found. Install with: pip install pinecone-client")
    raise


class PineconeClient:
    """Pinecone client for storing and retrieving code embeddings"""

    def __init__(self, api_key: Optional[str] = None, environment: Optional[str] = None,
                 index_name: Optional[str] = None, namespace: Optional[str] = None):
        """Initialize Pinecone client"""
        self.api_key = api_key or os.getenv('PINECONE_API_KEY')
        self.environment = environment or os.getenv('PINECONE_ENVIRONMENT', 'us-east-1-aws')
        self.index_name = index_name or os.getenv('PINECONE_INDEX_NAME', 'model-earth')
        self.namespace = namespace or os.getenv('PINECONE_NAMESPACE', 'codechat-main')

        if not self.api_key:
            raise ValueError("PINECONE_API_KEY is required")

        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Pinecone client and ensure index exists"""
        try:
            self.pc = Pinecone(api_key=self.api_key)

            # Check if index exists, create if not
            if self.index_name not in self.pc.list_indexes().names():
                print(f"📝 Creating Pinecone index '{self.index_name}'...")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI ada-002 dimension
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region=self.environment.split('-')[0])
                )
                print(f"✅ Index '{self.index_name}' created")
            else:
                print(f"✅ Connected to Pinecone index '{self.index_name}'")

            self.index = self.pc.Index(self.index_name)
            print("✅ Pinecone client initialized")

        except Exception as e:
            print(f"❌ Failed to initialize Pinecone client: {e}")
            raise

    def store_chunk(self, chunk: Dict[str, Any], chunk_summary: str,
                   embedding: List[float], repo_name: str, file_path: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Store a single code chunk in Pinecone with enhanced metadata

        Args:
            chunk: Chunk dictionary with content and metadata
            chunk_summary: AI-generated summary of the chunk
            embedding: Embedding vector for the chunk
            repo_name: Name of the repository
            file_path: Path to the file
            metadata: Enhanced metadata from SmartChunker

        Returns:
            True if stored successfully, False otherwise
        """
        try:
            # Create unique ID for the chunk
            chunk_id = chunk.get('id', '')
            if not chunk_id:
                # Generate ID from content hash if not provided
                content_hash = hashlib.md5(chunk.get('content', '').encode()).hexdigest()[:16]
                chunk_id = f"{repo_name}_{Path(file_path).name}_{content_hash}"

            # Prepare enhanced metadata
            base_metadata = {
                'repo_name': repo_name,
                'file_path': file_path,
                'chunk_content': chunk.get('content', ''),
                'chunk_summary': chunk_summary,
                'chunk_id': chunk_id,
                'language': self._detect_language(file_path),
                'timestamp': str(Path(file_path).stat().st_mtime) if Path(file_path).exists() else '',
                'line_start': chunk.get('start_line', 0),
                'line_end': chunk.get('end_line', 0),
                'chunk_type': chunk.get('type', 'code')
            }

            # Add enhanced metadata if provided
            if metadata:
                enhanced_metadata = {
                    'semantic_types': metadata.get('semantic_types', ''),
                    'functions': metadata.get('functions', ''),
                    'classes': metadata.get('classes', ''),
                    'complexity': metadata.get('complexity', 0),
                    'token_count': metadata.get('token_count', 0),
                    'language_detected': metadata.get('language', ''),
                    'content_length': metadata.get('content_length', len(chunk.get('content', '')))
                }
                base_metadata.update(enhanced_metadata)

            # Prepare vector data
            vector_data = {
                'id': chunk_id,
                'values': embedding,
                'metadata': base_metadata
            }

            # Upsert to Pinecone
            self.index.upsert(vectors=[vector_data], namespace=self.namespace)

            print(f"   ✅ Stored chunk {chunk_id} in Pinecone")
            return True

        except Exception as e:
            print(f"   ❌ Failed to store chunk in Pinecone: {e}")
            return False

    def store_multiple_chunks(self, chunks_data: List[Dict[str, Any]]) -> int:
        """
        Store multiple chunks in Pinecone

        Args:
            chunks_data: List of dictionaries with chunk, summary, embedding, repo_name, file_path

        Returns:
            Number of chunks successfully stored
        """
        stored_count = 0

        for chunk_data in chunks_data:
            success = self.store_chunk(
                chunk=chunk_data['chunk'],
                chunk_summary=chunk_data['summary'],
                embedding=chunk_data['embedding'],
                repo_name=chunk_data['repo_name'],
                file_path=chunk_data['file_path']
            )
            if success:
                stored_count += 1

        return stored_count

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for chunks (for lambda compatibility)

        Args:
            chunks: List of chunk dictionaries

        Returns:
            List of chunks with embeddings added
        """
        # This is a compatibility method for the lambda
        # In practice, embeddings should be generated before calling this
        for chunk in chunks:
            if 'embedding' not in chunk:
                # Generate a simple hash-based "embedding" as placeholder
                content = chunk.get('content', '')
                content_hash = hashlib.md5(content.encode()).hexdigest()
                # Convert hash to list of floats (not real embeddings)
                chunk['embedding'] = [float(int(content_hash[i:i+2], 16)) for i in range(0, min(32, len(content_hash)), 2)]

        return chunks

    def upsert_vectors(self, repo: str, commit_sha: str, vectors: List[Dict[str, Any]]) -> bool:
        """
        Upsert vectors to Pinecone (for lambda compatibility)

        Args:
            repo: Repository name
            commit_sha: Commit SHA
            vectors: List of vector dictionaries

        Returns:
            True if successful
        """
        try:
            upsert_data = []
            for vector in vectors:
                vector_id = vector.get('id', f"{repo}_{commit_sha}_{len(upsert_data)}")
                upsert_data.append({
                    'id': vector_id,
                    'values': vector.get('embedding', []),
                    'metadata': {
                        'repo_name': repo,
                        'commit_sha': commit_sha,
                        'file_path': vector.get('file_path', ''),
                        'chunk_content': vector.get('content', ''),
                        'chunk_summary': vector.get('summary', ''),
                        'language': self._detect_language(vector.get('file_path', ''))
                    }
                })

            if upsert_data:
                self.index.upsert(vectors=upsert_data, namespace=self.namespace)
                print(f"✅ Upserted {len(upsert_data)} vectors to Pinecone")
                return True
            else:
                print("⚠️  No vectors to upsert")
                return False

        except Exception as e:
            print(f"❌ Failed to upsert vectors: {e}")
            return False

    def search(self, query_embedding: List[float], top_k: int = 5, live: bool = False) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using embedding.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            live: If True, filters for vectors with `live:true` metadata.

        Returns:
            List of similar chunks with scores and metadata.
        """
        filter_metadata = {'live': True} if live else None
        return self.search_similar(query_embedding, top_k, filter_metadata)

    def search_similar(self, query_embedding: List[float], top_k: int = 5,
                      filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using embedding

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of similar chunks with scores and metadata
        """
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=self.namespace,
                filter=filter_metadata,
                include_metadata=True,
                include_values=False
            )

            # Format results
            formatted_results = []
            for match in results.matches:
                result = {
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.metadata or {}
                }
                formatted_results.append(result)

            return formatted_results

        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            stats = self.index.describe_index_stats()
            namespace_stats = stats.namespaces.get(self.namespace, {})

            return {
                'total_vectors': namespace_stats.get('vector_count', 0),
                'index_name': self.index_name,
                'namespace': self.namespace,
                'dimension': stats.dimension
            }
        except Exception as e:
            print(f"❌ Failed to get stats: {e}")
            return {}

    def delete_vectors(self, vector_ids: List[str]) -> bool:
        """Delete vectors by IDs"""
        try:
            self.index.delete(ids=vector_ids, namespace=self.namespace)
            print(f"✅ Deleted {len(vector_ids)} vectors")
            return True
        except Exception as e:
            print(f"❌ Failed to delete vectors: {e}")
            return False

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext = Path(file_path).suffix.lower()
        lang_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.scala': 'scala',
            '.html': 'html',
            '.css': 'css',
            '.sql': 'sql',
            '.yml': 'yaml',
            '.yaml': 'yaml',
            '.json': 'json',
            '.xml': 'xml',
            '.md': 'markdown'
        }
        return lang_map.get(ext, 'unknown')

    def close(self):
        """Close the Pinecone client connection"""
        # Pinecone client doesn't need explicit closing
        pass
