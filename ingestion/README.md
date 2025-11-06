# Ingestion Pipeline

Automated vector database synchronization using LlamaIndex for intelligent code chunking and OpenAI for embeddings.

## Overview

The ingestion pipeline keeps the Pinecone vector database in sync with repository changes by:
1. Detecting changed files via git commit ranges
2. Chunking code using language-aware parsers
3. Generating embeddings with OpenAI
4. Upserting/deleting vectors in Pinecone

**Runs automatically** via GitHub Actions on every push/PR merge to `main`.

## Files

```
ingestion/
├── vector_db_sync.py      # Main sync script
├── llama_chunker.py       # LlamaIndex-based chunking
├── test_vectordb_sync.py  # End-to-end tests
└── requirements.txt       # Dependencies
```

## Usage

### Automatic (CI/CD)

The GitHub Actions workflow automatically runs on push/PR merge:

```yaml
# .github/workflows/vector_db_sync.yml
- run: python codechat/ingestion/vector_db_sync.py --skip-on-missing-keys
```

**Auto-detection:**
- Reads `GITHUB_EVENT_PATH` to extract commit range
- Handles push and pull_request events
- Validates API keys before running
- Exits gracefully if keys are missing (--skip-on-missing-keys)

### Manual Sync

```bash
# Sync specific commit range
python ingestion/vector_db_sync.py \
  --from-commit HEAD~10 \
  --to-commit HEAD

# Sync single commit
python ingestion/vector_db_sync.py \
  --from-commit abc123^ \
  --to-commit abc123

# Bulk ingestion (all files from empty tree)
python ingestion/vector_db_sync.py \
  --from-commit 4b825dc642cb6eb9a060e54bf8d69288fbee4904 \
  --to-commit HEAD

# Explicit file list with status prefixes
python ingestion/vector_db_sync.py \
  --files A:path/to/new.py M:path/to/modified.js D:path/to/deleted.md

# Retry failed files
python ingestion/vector_db_sync.py \
  --retry-errors codechat/.vector_sync_errors.jsonl
```

### Command-Line Options

| Option | Description |
|--------|-------------|
| `--from-commit` | Starting commit/ref for git diff |
| `--to-commit` | Ending commit/ref (default: HEAD) |
| `--repo-root` | Git repository root (default: `.`) |
| `--files` | Explicit file list with optional status prefix (`A:`, `M:`, `D:`) |
| `--retry-errors` | Retry files from error log |
| `--errors-out` | Error log output path (default: `codechat/.vector_sync_errors.jsonl`) |
| `--skip-on-missing-keys` | Exit gracefully if API keys missing (for CI/CD) |

## Chunking Strategy

### LlamaIndex Node Parsers

The pipeline uses LlamaIndex's built-in parsers for intelligent, language-aware chunking:

#### 1. **CodeSplitter** - AST-Aware Code Chunking

```python
CodeSplitter(
    language="python",          # Auto-detected per file
    chunk_lines=50,             # Target lines per chunk
    chunk_lines_overlap=15,     # Context overlap
    max_chars=2000              # Hard limit
)
```

**Supported languages:**
- Python, JavaScript, TypeScript, Java, C++, Go, Rust
- PHP, Ruby, Swift, Kotlin, Scala
- C#, C, Shell, Lua, Perl

**Features:**
- Respects function/class boundaries
- Preserves imports and context
- Maintains syntactic validity

#### 2. **MarkdownNodeParser** - Header-Aware Parsing

```python
MarkdownNodeParser()
```

**Features:**
- Splits on markdown headers (`#`, `##`, `###`)
- Maintains document structure
- Preserves code blocks and lists

#### 3. **JSONNodeParser** - Structure-Aware JSON/YAML

```python
JSONNodeParser()
```

**Features:**
- Depth-first traversal of JSON structure
- Preserves nested relationships
- Handles large JSON files intelligently

**YAML Support:**
- Converts YAML → JSON using PyYAML
- Then applies JSONNodeParser

#### 4. **SentenceSplitter** - Generic Text Chunking

```python
SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=200
)
```

**Used for:**
- Plain text files
- Configuration files (`.conf`, `.ini`, `.cfg`)
- Log files
- Any unrecognized file types

### File Type Dispatch

```python
# Automatic language detection by extension
.py, .js, .ts, .java, .cpp, .go, .rs  → CodeSplitter
.md, .markdown                         → MarkdownNodeParser
.json                                  → JSONNodeParser
.yaml, .yml                            → YAML → JSON → JSONNodeParser
.txt, .log, .conf                      → SentenceSplitter
.csv, .tsv                             → Summary (first 20 lines)
```

### Special Handling

**CSV/TSV Files:**
```python
# Preview-based summary (no pandas dependency)
summary = f"""CSV File: {filename}
First 20 lines preview:
{content}
"""
```

**Binary/Large Files:**
```python
# Metadata-only summary
summary = f"""File: {filename}
Size: {size} MB
Type: {file_type}
"""
```

## Change Detection

### Git-Based Incremental Sync

The sync script computes changed files between commits:

```python
git diff --name-status <from> <to>
```

**Status codes:**
- `A` - Added (new file)
- `M` - Modified (existing file changed)
- `D` - Deleted (file removed)
- `R` - Renamed (handled as Delete old + Add new)

### Submodule Support

Automatically expands submodule pointer changes:

```bash
# Detect submodule commits
git diff --submodule=short <from> <to>

# List files in changed commits
git ls-tree -r --name-only <sha>

# Compute file-level diffs
git diff --name-status <old-sha> <new-sha>
```

**Example:**
```
Submodule codechat abc123..def456
  → Expands to: A codechat/file1.py, M codechat/file2.js
```

## Vector Operations

### Metadata Structure

Each chunk is stored with comprehensive metadata:

```python
{
    "id": "uuid",
    "values": [1536-dim embedding],
    "metadata": {
        "repo_name": "modelearth/webroot",
        "file_path": "codechat/ingestion/vector_db_sync.py",
        "file_type": "python",
        "chunk_content": "def function()...",
        "chunk_summary": "AI-generated summary",
        "chunk_id": "0",
        "language": "python",
        "timestamp": "2024-11-06T12:00:00",
        "commit_sha": "abc123"
    }
}
```

### Operations

**Add (A):**
1. Chunk file with LlamaChunker
2. Generate embedding for each chunk
3. Upsert vectors to Pinecone

**Modify (M):**
1. Delete existing vectors (by file_path filter)
2. Re-chunk and re-embed updated content
3. Upsert new vectors

**Delete (D):**
1. Delete vectors by file_path filter
2. Remove from namespace

**Rename (R):**
1. Delete vectors for old path
2. Add vectors for new path

### Batch Processing

```python
BATCH_SIZE = 10  # Vectors per upsert
```

**Benefits:**
- Efficient API usage
- Progress tracking with tqdm
- Handles large file sets

## Error Handling

### Error Log Format

Failed operations are logged to `.vector_sync_errors.jsonl`:

```jsonl
{"file_path": "path/to/file.py", "operation": "chunk", "message": "Parse error", "status": "M"}
{"file_path": "path/to/file.js", "operation": "embed", "message": "API error", "status": "A"}
```

### Retry Mechanism

```bash
# Retry all failed files
python ingestion/vector_db_sync.py --retry-errors

# Specify custom error log
python ingestion/vector_db_sync.py --retry-errors /path/to/errors.jsonl
```

### CI/CD Error Reporting

GitHub Actions workflow includes error summary:

```yaml
- name: Error summary
  if: always()
  run: |
    if [ -f codechat/.vector_sync_errors.jsonl ]; then
      tail -n 200 codechat/.vector_sync_errors.jsonl >> $GITHUB_STEP_SUMMARY
    fi

- name: Upload errors
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: vector-sync-errors
    path: codechat/.vector_sync_errors.jsonl
```

## Testing

### End-to-End Test

```bash
python ingestion/test_vectordb_sync.py
```

**Test scenarios:**
1. **Add** - Create new file and index
2. **Modify** - Update file and reindex
3. **Rename** - Move file (delete old + add new)
4. **Delete** - Remove file and delete vectors

**Requirements:**
- Valid `OPENAI_API_KEY`
- Valid `PINECONE_API_KEY`
- Keys in `codechat/.env` or environment

**Test artifacts:**
- Creates temporary files in `codechat/`
- Uses unique namespace: `vector-sync-test-{uuid}`
- Cleans up vectors and files after completion

### Test Output

```
[info] Auto-detected commit range: abc123..def456
[info] Starting VectorDB sync for modelearth-webroot
[info] Using LlamaIndex for intelligent code-aware chunking

[info] Deleting vectors for 3 files...
[info] Deleted vectors for: path/to/file.py

[info] Upserting 25 chunks in batches of 10...
Upserting: 100%|████████| 25/25 [00:05<00:00, 4.8it/s]

[info] Sync Complete:
  - Files processed: 15
  - Files skipped: 2
  - Files with errors: 0
  - Vectors deleted: 3 files
  - Chunks upserted: 25
  - Embedding model: text-embedding-3-small
```

## Dependencies

See [requirements.txt](requirements.txt):

```txt
openai>=1.12.0              # Embeddings
pinecone-client>=3.0.0      # Vector storage
llama-index-core>=0.12.0    # Chunking
tree-sitter-language-pack   # Code parsing
PyYAML>=6.0                 # YAML support
tiktoken>=0.5.0             # Token counting
tqdm>=4.66.0                # Progress bars
```

## Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...

# Optional (defaults provided)
PINECONE_INDEX=repo-chunks
GITHUB_REPOSITORY=owner/repo  # Auto-set in Actions
GITHUB_SHA=abc123              # Auto-set in Actions
```

## Performance

**Typical metrics:**
- Chunking: ~200 files/minute
- Embedding: ~50 chunks/second (batch API)
- Upsert: ~100 vectors/second (batched)

**Example run:**
```
50 files, 500 chunks → ~15 seconds
```

## Disaster Recovery

See [scripts/restore.py](../scripts/restore.py):

```bash
# Rollback to specific commit
python scripts/restore.py abc123 --namespace modelearth-webroot

# Verify vectors before deletion
python scripts/restore.py abc123 --dry-run
```

## Architecture Decisions

### Why LlamaIndex?

**Before:**
- 10,000 lines of custom chunking code
- 51 language-specific files
- Manual tree-sitter integration
- Hard to maintain and extend

**After:**
- ~300 lines with LlamaIndex
- Built-in support for 20+ languages
- AST-aware splitting
- Active maintenance by LlamaIndex team

### Why Incremental Sync?

**Git-based change detection:**
- Only process changed files
- Fast CI/CD runs (seconds vs minutes)
- Preserves existing vectors
- Handles renames correctly

**Alternative (full re-index):**
- Would process 1000s of files every push
- Expensive API calls
- Slow workflow runs
- No better accuracy

## Troubleshooting

### Common Issues

**1. "No module named 'llama_index'"**
```bash
pip install -r ingestion/requirements.txt
```

**2. "RuntimeError: GITHUB_EVENT_PATH not set"**
```bash
# Provide explicit commit range
python ingestion/vector_db_sync.py --from-commit HEAD~1 --to-commit HEAD
```

**3. "Missing required API keys"**
```bash
# Check environment
echo $OPENAI_API_KEY
echo $PINECONE_API_KEY

# Or use .env file
cp .env.example .env
# Edit .env
```

**4. "Pinecone index not found"**
```bash
# Create index (one-time setup)
python -c "
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key='...')
pc.create_index(
    name='repo-chunks',
    dimension=1536,
    metric='cosine',
    spec=ServerlessSpec(cloud='aws', region='us-east-1')
)
"
```

## Contributing

1. Make changes to chunking logic in `llama_chunker.py`
2. Run tests: `python ingestion/test_vectordb_sync.py`
3. Test manually: `python ingestion/vector_db_sync.py --from-commit HEAD~1`
4. Create PR - CI will test your changes

## Related Documentation

- [Main README](../README.md) - System overview
- [LlamaChunker](llama_chunker.py) - Chunking implementation
- [Pinecone Client](../lib/pine.py) - Vector DB wrapper
- [Restore Script](../scripts/restore.py) - Rollback utility
