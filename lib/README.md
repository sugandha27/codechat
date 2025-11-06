# Lib - Shared Utilities

Shared utilities used across multiple CodeChat components (ingestion, scripts, etc.).

## Files

```
lib/
├── __init__.py          # Package exports
└── pine.py              # Pinecone client wrapper
```

## PineconeClient

Wrapper around the Pinecone API for vector database operations.

### Usage

```python
from lib.pine import PineconeClient

# Initialize client
client = PineconeClient(
    api_key="your-api-key",
    index_name="repo-chunks"
)

# Upsert vectors
client.upsert(
    vectors=[{
        "id": "chunk-1",
        "values": [0.1, 0.2, ...],  # 1536-dim embedding
        "metadata": {
            "file_path": "path/to/file.py",
            "repo_name": "modelearth/webroot"
        }
    }],
    namespace="modelearth-webroot"
)

# Query vectors
results = client.query(
    vector=[0.1, 0.2, ...],
    top_k=10,
    namespace="modelearth-webroot",
    filter={"repo_name": "modelearth/webroot"}
)

# Delete vectors by filter
client.delete(
    filter={"file_path": "path/to/file.py"},
    namespace="modelearth-webroot"
)

# Fetch specific vectors
vectors = client.fetch(
    ids=["chunk-1", "chunk-2"],
    namespace="modelearth-webroot"
)
```

### Features

- **Dual SDK Support**: Handles both serverless and classic Pinecone clients
- **Automatic Index Creation**: Creates index if it doesn't exist
- **Namespace Management**: Isolates vectors by repository
- **Error Handling**: Graceful degradation for API errors

### Index Configuration

```python
# Serverless (default)
ServerlessSpec(
    cloud="aws",
    region="us-east-1"
)

# Classic (fallback)
metric="cosine"
dimension=1536
pod_type="p1.x1"
```

## Used By

- **ingestion/vector_db_sync.py** - Vector database sync
- **scripts/restore.py** - Disaster recovery rollback
- **src/lambdas/query_handler/** - Query API (future)

## Dependencies

```python
# Serverless SDK (preferred)
from pinecone import Pinecone, ServerlessSpec

# Classic SDK (fallback)
import pinecone
```

## Environment Variables

```bash
PINECONE_API_KEY=your-key      # Required
PINECONE_INDEX=repo-chunks     # Optional (default)
PINECONE_ENVIRONMENT=us-east-1 # Optional (classic only)
```

## Design Decisions

### Why a Wrapper?

1. **SDK Compatibility**: Supports both serverless and classic Pinecone APIs
2. **Consistent Interface**: Single API for all vector operations
3. **Reusability**: Shared across ingestion, scripts, and backend
4. **Error Handling**: Centralized retry logic and error messages

### Why in lib/?

- Not executable (so not in `scripts/`)
- Not ingestion-specific (used by restore.py)
- Not core business logic (so not in `src/`)
- Shared utility used across multiple components

## Related Documentation

- [Main README](../README.md) - System overview
- [Ingestion Pipeline](../ingestion/README.md) - Primary user of PineconeClient
- [Restore Script](../scripts/restore.py) - Disaster recovery using PineconeClient
