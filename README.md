# CodeChat - Repository-Intelligent Code Search

AI-powered semantic search for code repositories using vector embeddings and LLM-based query understanding.

## Architecture Overview

CodeChat consists of three main components:

1. **Ingestion Pipeline** (`ingestion/`) - Automated vector database sync via GitHub Actions
2. **Backend API** (`backend/`, `src/lambdas/`) - AWS Lambda functions for query handling
3. **Frontend UI** (`chat/`) - Material Design chat interface

```
┌─────────────────────┐
│  GitHub Actions     │
│  (Push/PR events)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Ingestion Pipeline  │
│ - LlamaIndex chunking│
│ - OpenAI embeddings │
│ - Pinecone storage  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Pinecone VectorDB  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Lambda Backend    │
│  - Query handler    │
│  - Repository list  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Chat Frontend     │
│  - Material Design  │
│  - Multi-LLM support│
└─────────────────────┘
```

## Quick Start

### 1. Local Development

```bash
# Set up environment
conda create -n model-earth-codechat python=3.11
conda activate model-earth-codechat

# Install dependencies
pip install -r backend/requirements.txt  # For backend
pip install -r ingestion/requirements.txt  # For ingestion

# Set API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY, PINECONE_API_KEY

# Run local dev server
python scripts/dev_server.py --mock
```

### 2. Test Ingestion Pipeline

```bash
# Run full ingestion test (requires API keys)
python ingestion/test_vectordb_sync.py

# Manual ingestion from commit range
python ingestion/vector_db_sync.py --from-commit HEAD~10 --to-commit HEAD

# Bulk ingestion (all files)
python ingestion/vector_db_sync.py --from-commit 4b825dc642cb6eb9a060e54bf8d69288fbee4904
```

### 3. Deploy to AWS

```bash
# Deploy Lambda functions and infrastructure
cd backend
./scripts/deploy_lambda.sh

# Or use Terraform directly
cd backend/infra
terraform init
terraform apply
```

## Directory Structure

```
codechat/
├── ingestion/          # Vector DB sync pipeline
│   ├── vector_db_sync.py    # Main sync script (CI/CD)
│   ├── llama_chunker.py     # LlamaIndex-based chunking
│   ├── test_vectordb_sync.py # E2E tests
│   └── requirements.txt     # Ingestion dependencies
│
├── lib/                # Shared utilities
│   └── pine.py         # Pinecone client wrapper
│
├── backend/            # AWS infrastructure
│   ├── infra/          # Terraform configs
│   └── lambda_layers/  # Python dependencies
│
├── src/lambdas/        # Lambda function code
│   ├── query_handler/  # Main query API
│   └── get_repositories/ # Repository list API
│
├── chat/               # Frontend UI
│   ├── index.html      # Chat interface
│   ├── script.js       # Application logic
│   └── styles.css      # Material Design styles
│
├── scripts/            # Utility scripts
│   ├── dev_server.py   # Local development server
│   ├── restore.py      # Vector DB rollback tool
│   └── deploy_lambda.py # Deployment automation
│
└── config/             # Configuration files
    └── repositories.yml # Repository metadata
```

## Core Components

### Ingestion Pipeline

**Automated vector database sync triggered by GitHub Actions on every push/PR merge:**

- **Intelligent Chunking** (LlamaIndex):
  - AST-aware code splitting with `CodeSplitter`
  - Header-based markdown parsing with `MarkdownNodeParser`
  - Structure-aware JSON/YAML with `JSONNodeParser`
  - Generic text chunking with `SentenceSplitter`

- **Incremental Updates**:
  - Git-based change detection (commit range diffs)
  - Submodule-aware file tracking
  - Add (A), Modify (M), Delete (D), Rename (R) operations
  - Error tracking and retry mechanism

- **Vector Storage**:
  - OpenAI `text-embedding-3-small` (1536 dimensions)
  - Pinecone serverless with namespace isolation
  - Per-repository metadata (file path, language, commit SHA)

See [ingestion/README.md](ingestion/README.md) for detailed documentation.

### Backend API

**AWS Lambda functions with API Gateway:**

- **Query Handler** (`POST /query`):
  - Repository-scoped semantic search
  - Multi-LLM support (Bedrock, OpenAI, Anthropic)
  - Context-aware response generation
  - CORS-enabled REST API

- **Repository List** (`GET /repositories`):
  - Available repositories for search
  - Metadata from S3 configuration

**Infrastructure**:
- Python 3.13 runtime
- 1024 MB memory, 300s timeout
- Shared dependency layers
- CloudWatch logging

### Frontend UI

**Material Design chat interface:**

- Repository selection dropdown
- LLM provider options
- Real-time search with typing indicators
- Conversation history
- Dark theme

Hosted at: `https://model.earth/chat/`

See [chat/README.md](chat/README.md) for UI documentation.

## Configuration

### Environment Variables

```bash
# Required for ingestion
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...

# Optional (have defaults)
PINECONE_INDEX=repo-chunks
PINECONE_ENVIRONMENT=us-east-1-aws
```

### Repository Configuration

Edit `config/repositories.yml`:

```yaml
repositories:
  - name: "modelearth/webroot"
    description: "Main website repository"
    priority: "high"
```

## CI/CD Pipeline

**GitHub Actions workflow** (`.github/workflows/vector_db_sync.yml`):

```yaml
# Triggers on push to main and merged PRs
on:
  push:
    branches: [main]
  pull_request:
    types: [closed]

# Auto-detects commit range and syncs changed files
- run: python codechat/ingestion/vector_db_sync.py --skip-on-missing-keys
```

**Workflow features:**
- Auto-detects commit range from GitHub event payload
- Validates API keys before running
- Uploads error logs as artifacts
- Runs on every code change to keep vectors in sync

## Development

### Running Tests

```bash
# Ingestion pipeline E2E test
python ingestion/test_vectordb_sync.py

# Backend API tests (if available)
python -m pytest tests/
```

### Local Backend

```bash
# Mock mode (no API calls)
python scripts/dev_server.py --mock

# Real mode (requires API keys)
python scripts/dev_server.py
```

Access at `http://127.0.0.1:8080`

### Disaster Recovery

Rollback vector database to a previous commit:

```bash
python scripts/restore.py <commit-sha> [--namespace repo-name]
```

## Recent Changes

**November 2024 - Ingestion Pipeline Refactor:**
- Replaced 10,000 lines of custom chunking with LlamaIndex (~300 lines)
- Deleted unused `src/core/chunking/` directory (51 files)
- Reorganized into `ingestion/`, `lib/`, `scripts/` structure
- Simplified CI/CD workflow from 28 lines to 3 lines
- Added auto-detection of GitHub Actions environment
- Moved validation logic from bash to Python

## Performance

**Ingestion:**
- ~100-200 files/minute processing speed
- Batch upserts (10 chunks per batch)
- Efficient git diff-based change detection

**Query:**
- Sub-second vector search
- LLM response generation: 2-5 seconds
- CloudWatch metrics available

## Troubleshooting

### Ingestion Issues

```bash
# Check recent workflow runs
gh run list --workflow=vector_db_sync.yml

# View logs
gh run view <run-id> --log

# Manually trigger sync
python ingestion/vector_db_sync.py --from-commit HEAD~1 --to-commit HEAD
```

### API Issues

```bash
# Check Lambda logs
aws logs tail /aws/lambda/codechat-query-handler --follow

# Test endpoint
curl -X POST https://api-url/query \
  -d '{"query": "test", "repo_name": "modelearth/webroot"}'
```

## Contributing

1. Make changes in feature branch
2. Test locally: `python ingestion/test_vectordb_sync.py`
3. Create PR - vector sync runs automatically on merge
4. Monitor workflow: https://github.com/modelearth/webroot/actions

## Links

- **Chat UI**: https://model.earth/chat/
- **Main Repository**: https://github.com/modelearth/webroot
- **Documentation**: [docs/](docs/)

---

*For component-specific documentation, see README files in subdirectories.*
