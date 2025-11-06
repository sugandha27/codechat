"""
Vector DB sync for the webroot superproject and its submodules.

Invocation modes
- changed_files.txt: python codechat/vectordb_sync.py <changed_files.txt>
  (lines from `git diff --name-status`)
- explicit files: python codechat/vectordb_sync.py --files A:path M:path D:path â€¦
  (status prefix optional; default is M if omitted)
- commit range replay (recommended):
  python codechat/vectordb_sync.py --from-commit <rev> [--to-commit HEAD] [--repo-root .]
  Expands submodule pointer changes into file-level A/M/D across changed submodules.

Behavior
- A/M: pre-delete vectors for the path (idempotent), then chunk + embed + upsert
- D: delete vectors for the path
- Rename: expanded to D old + M new (superproject and submodules)
- Embeddings are content-only; file path is stored in metadata
- Strict failure: unexpected errors fail the run; errors recorded to JSONL

Env vars
- PINECONE_API_KEY (required)
- OPENAI_API_KEY (required)
- PINECONE_CLOUD (serverless; default: aws)
- PINECONE_REGION (serverless; default: us-east-1)
- PINECONE_ENV (classic fallback; default: us-west1-gcp)
- PINECONE_INDEX (optional, default: repo-chunks)
"""

# pyright: basic

import os
import sys
import uuid
import re
import json
from pathlib import Path
from typing import List, Tuple, Optional
from openai import OpenAI
import tiktoken
from tqdm import tqdm
import argparse
import subprocess

# Import unified chunker (same directory)
from llama_chunker import LlamaChunker

# Constants
MAX_TOKENS = 8192
INDEX_NAME = "repo-chunks"
DIMENSION = 1536
METRIC = "cosine"
BATCH_SIZE = 10

# Prefer serverless Pinecone SDK; fallback to classic client if not available
USE_SERVERLESS = False
pc = None  # type: ignore
pinecone_client = None  # type: ignore
try:
    from pinecone import Pinecone, ServerlessSpec  # type: ignore

    USE_SERVERLESS = True
except Exception:
    try:
        import pinecone as pinecone_client  # type: ignore

        USE_SERVERLESS = False
    except Exception:
        pinecone_client = None  # type: ignore
        USE_SERVERLESS = False

# Clients are initialized in main() to avoid import-time failures
index = None
openai_client = None
tokenizer = None
chunker = None  # LlamaChunker instance


def count_tokens(text: str) -> int:
    if tokenizer is None:
        # Should not happen (tokenizer initialized in main) but be safe
        return len(text.split())
    return len(tokenizer.encode(text, allowed_special="all"))


def re_chunk_if_oversize(sections: List[str], max_tokens: int = MAX_TOKENS) -> List[str]:
    final_chunks: List[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue

        tokens = count_tokens(section)
        if tokens <= max_tokens:
            final_chunks.append(section)
        else:
            split_points = re.split(r'(?<=[.!?])\s+', section)
            current_chunk = ""

            for part in split_points:
                test_chunk = current_chunk + (" " + part if current_chunk else part)
                if count_tokens(test_chunk) <= max_tokens:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        final_chunks.append(current_chunk.strip())
                    current_chunk = part

            if current_chunk.strip():
                final_chunks.append(current_chunk.strip())

            really_final: List[str] = []
            for chunk in final_chunks:
                if count_tokens(chunk) <= max_tokens:
                    really_final.append(chunk)
                else:
                    char_chunks = re.findall(r'.{1,3000}(?:\s+|$)', chunk)
                    really_final.extend([s.strip() for s in char_chunks if s.strip()])
            final_chunks = really_final

    return final_chunks


def detect_file_type(filepath: str) -> str:
    path = Path(filepath)
    ext = path.suffix.lower()

    if (ext == '' or ext in {'.sh'}) and path.exists():
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
                if first_line.startswith('#!'):
                    if 'python' in first_line:
                        return 'python'
                    elif any(shell in first_line for shell in ['bash', 'sh', 'zsh']):
                        return 'bash'
        except Exception:
            pass

    return ext.lstrip('.')


# Chunking functions replaced by LlamaChunker
# chunk_code_tree_sitter, chunk_markdown, chunk_json_yaml removed


# chunk_ipynb removed - LlamaIndex JSONNodeParser handles .ipynb files as JSON

def chunk_csv_tsv(filepath: Path):
    """Simplified CSV/TSV handling without pandas dependency"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [f.readline() for _ in range(20)]  # Read first 20 lines

        content = ''.join(lines)
        summary = f"""CSV/TSV File: {filepath.name}
First 20 lines preview:
{content}
"""
        return re_chunk_if_oversize([summary]), True

    except Exception as e:
        return [f"Error reading CSV {filepath}: {e}"], False


def chunk_as_summary(filepath: Path):
    path = Path(filepath)
    file_type = detect_file_type(str(filepath))

    try:
        stat = path.stat()
        size_mb = stat.st_size / (1024 * 1024)

        summary = f"""{file_type.upper()} file: {path.name}
Path: {filepath}
Size: {size_mb:.2f} MB
Type: {file_type}
"""
        if size_mb < 1 and file_type in {'txt', 'log', 'conf', 'ini', 'cfg'}:
            try:
                preview = path.read_text(encoding='utf-8', errors='ignore')[:500]
                summary += f"\nPreview:\n{preview}..."
            except Exception:
                pass

        return re_chunk_if_oversize([summary]), True

    except Exception as e:
        return [f"Error accessing {filepath}: {e}"], False


def dispatch_chunking(filepath: Path):
    """Simplified chunking - LlamaIndex replaces all custom AST parsing"""
    path = Path(filepath)
    ext = path.suffix.lower()

    # CSV/TSV: simple preview (no pandas needed)
    if ext in {".csv", ".tsv"}:
        return chunk_csv_tsv(path)

    # LlamaIndex handles: py, js, ts, java, cpp, go, rust, etc. (code)
    #                     md, txt, rst (markdown)
    #                     json, yaml, yml, ipynb (structured data)
    #                     html, css, xml (markup)
    try:
        chunks = chunker.chunk_file(str(filepath))
        if chunks:
            return chunks, True
    except Exception as e:
        print(f"LlamaChunker error for {filepath}: {e}")

    # Fallback: binary/unsupported files get metadata summary
    return chunk_as_summary(path)


def get_embedding(text: str) -> List[float]:
    if not text or not text.strip():
        raise RuntimeError("Empty text provided for embedding")
    try:
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        if not embedding:
            raise RuntimeError("Received empty embedding from API")
        return embedding
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}")


def get_accurate_line_range(chunk: str, full_text: str) -> str:
    if not full_text or not chunk:
        return "L1-L1"

    try:
        chunk_clean = chunk.strip()
        if not chunk_clean:
            return "L1-L1"

        full_lines = full_text.split('\n')
        chunk_lines = chunk_clean.split('\n')

        first_chunk_line = chunk_lines[0].strip()
        if not first_chunk_line:
            return "L1-L1"

        for i, line in enumerate(full_lines):
            if first_chunk_line in line.strip() or line.strip() in first_chunk_line:
                start_line = i + 1
                end_line = start_line + len(chunk_lines) - 1
                return f"L{start_line}-L{end_line}"

        for i, line in enumerate(full_lines):
            if len(line.strip()) > 10 and line.strip() in chunk:
                start_line = i + 1
                chunk_line_count = chunk.count('\n') + 1
                end_line = start_line + chunk_line_count - 1
                return f"L{start_line}-L{end_line}"

        return "L1-L1"

    except Exception as e:
        print(f"[warn] Error calculating line range: {e}")
        return "L1-L1"


def process_file(filepath: str, status: str, repo_name: str):
    try:
        if not Path(filepath).exists():
            print(f"[warn] File not found: {filepath}")
            return []

        chunks, should_embed = dispatch_chunking(Path(filepath))
        chunk_entries = []

        full_text = ""
        if should_embed:
            try:
                full_text = Path(filepath).read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                print(f"[warn] Could not read full text for {filepath}: {e}")

        for i, chunk in enumerate(chunks):
            if not chunk or not chunk.strip():
                continue

            chunk_id = str(uuid.uuid4())
            vector: List[float] = []

            if should_embed and count_tokens(chunk) <= MAX_TOKENS:
                vector = get_embedding(chunk)

            chunk_entry = {
                "id": chunk_id,
                "values": vector,
                "metadata": {
                    "repo_name": repo_name,
                    "file_path": str(filepath),
                    "file_type": detect_file_type(filepath),
                    "chunk_type": "summary" if not should_embed else "content",
                    "chunk_index": i,
                    "chunk_id": chunk_id,
                    "content": chunk,
                    "line_range": get_accurate_line_range(chunk, full_text),
                    "embedded": bool(vector),
                    "should_embed": bool(should_embed),
                    "status": status,
                    "token_count": count_tokens(chunk)
                }
            }
            chunk_entries.append(chunk_entry)

        return chunk_entries

    except Exception as e:
        print(f"[error] Failed to process {filepath}: {e}")
        raise


def safe_delete_vectors(file_path: str, repo_name: str) -> None:
    try:
        _ = index.delete(
            filter={"repo_name": repo_name, "file_path": file_path},
            namespace=repo_name
        )
        print(f"[info] Deleted vectors for: {file_path}")
    except Exception as e:
        msg = str(e).lower()
        if "namespace not found" in msg or "code\":5" in msg:
            # Namespace hasn't been created yet; deletion is a no-op
            print(f"[info] Namespace absent on delete; skipping: {file_path}")
            return
        raise


def safe_upsert_batch(batch: List[dict], repo_name: str) -> int:
    for entry in batch:
        md = entry.get("metadata", {})
        if md.get("chunk_type") == "content" and not entry.get("values"):
            raise RuntimeError(f"Attempted to upsert empty embedding: {md.get('file_path')}")
    index.upsert(vectors=batch, namespace=repo_name)
    return len(batch)


def detect_github_commit_range() -> Tuple[str, str]:
    """
    Auto-detect commit range from GitHub Actions environment.
    Returns (from_commit, to_commit) tuple.
    Raises RuntimeError if not in GitHub Actions or if detection fails.
    """
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    event_name = os.environ.get("GITHUB_EVENT_NAME")
    github_sha = os.environ.get("GITHUB_SHA")

    if not event_path:
        raise RuntimeError("GITHUB_EVENT_PATH not set. Cannot auto-detect commit range. "
                          "Provide --from-commit or run in GitHub Actions environment.")

    if not os.path.exists(event_path):
        raise RuntimeError(f"GitHub event file not found: {event_path}")

    try:
        with open(event_path, 'r', encoding='utf-8') as f:
            event = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to parse GitHub event file: {e}")

    if event_name == "push":
        base = event.get("before", "")
        head = github_sha or event.get("after", "")
    elif event_name == "pull_request":
        pr = event.get("pull_request", {})
        base = pr.get("base", {}).get("sha", "")
        head = pr.get("merge_commit_sha") or github_sha or ""
    else:
        raise RuntimeError(f"Unsupported GitHub event type: {event_name}. "
                          "Only 'push' and 'pull_request' events are supported. "
                          "Use --from-commit for manual runs.")

    # Handle empty tree / first commit
    if not base or base == "0000000000000000000000000000000000000000":
        if head:
            base = f"{head}^"
        else:
            raise RuntimeError("Cannot determine base commit: both base and head are missing from event data")

    if not head:
        raise RuntimeError("Cannot determine head commit: missing from GitHub event data")

    return (base, head)


def parse_args():
    parser = argparse.ArgumentParser(description="VectorDB sync")
    parser.add_argument("changed_files", nargs="?", help="Path to changed_files.txt from git diff --name-status")
    parser.add_argument("--files", nargs="*",
                        help="Explicit files to process. Accepts optional status prefix: A:path, M:path, D:path. Default is M if omitted.")
    parser.add_argument("--retry-errors", dest="retry_errors", nargs="?", const="codechat/.vector_sync_errors.jsonl",
                        help="Retry files listed in an errors file (default: codechat/.vector_sync_errors.jsonl)")
    parser.add_argument("--errors-out", dest="errors_out", default="codechat/.vector_sync_errors.jsonl",
                        help="Path to write JSONL errors")
    parser.add_argument("--from-commit", dest="from_commit",
                        help="Compute changes from this commit/ref to HEAD or --to-commit. "
                             "For bulk ingestion of all files, use: 4b825dc642cb6eb9a060e54bf8d69288fbee4904 (empty tree)")
    parser.add_argument("--to-commit", dest="to_commit", default="HEAD", help="End commit/ref for diff (default: HEAD)")
    parser.add_argument("--repo-root", dest="repo_root", default=".",
                        help="Path to the git superproject root (default: current dir)")
    parser.add_argument("--skip-on-missing-keys", action="store_true",
                        help="Exit gracefully (code 0) if API keys are missing instead of raising an error")
    return parser.parse_args()


def append_error(path: str, file_path: str, operation: str, message: str, status: Optional[str] = None) -> None:
    try:
        rec = {"file_path": file_path, "operation": operation, "message": message}
        if status:
            rec["status"] = status
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        # Swallow error; rely on idempotent commit-range replay for recovery
        pass


def _run_git(args: List[str], cwd: str) -> str:
    cp = subprocess.run(["git"] + args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {cp.stderr.strip()}")
    return cp.stdout


def _parse_submodule_short(diff_text: str) -> List[Tuple[str, str, str]]:
    results: List[Tuple[str, str, str]] = []
    for line in diff_text.splitlines():
        m = re.match(r"^Submodule\s+([^\s]+)\s+([0-9a-f]{7,})\.\.([0-9a-f]{7,}).*$", line.strip())
        if m:
            results.append((m.group(1), m.group(2), m.group(3)))
    return results


def compute_changes_from_git(repo_root: str, from_rev: str, to_rev: str) -> List[Tuple[str, str]]:
    changes: List[Tuple[str, str]] = []
    # Superproject changes (expand renames)
    super_out = _run_git(["diff", "--name-status", from_rev, to_rev], repo_root)
    for line in super_out.splitlines():
        cols = line.strip().split("\t")
        if not cols:
            continue
        st = cols[0]
        if st.startswith("R") and len(cols) >= 3:
            # rename: old -> new
            changes.append(("D", cols[1]))
            changes.append(("M", cols[2]))
        elif len(cols) >= 2:
            changes.append((st, cols[1]))

    # Submodule pointer changes
    sub_out = _run_git(["diff", "--submodule=short", from_rev, to_rev], repo_root)
    for sub_path, oldsha, newsha in _parse_submodule_short(sub_out):
        sub_abs = str(Path(repo_root) / sub_path)
        sub_gitdir = str(Path(repo_root) / ".git" / "modules" / sub_path)
        added = (oldsha == "0000000" or re.fullmatch(r"0+", oldsha) is not None)
        deleted = (newsha == "0000000" or re.fullmatch(r"0+", newsha) is not None)
        try:
            if added:
                # List all files at newsha as added
                if Path(sub_abs).is_dir():
                    out = _run_git(["-C", sub_abs, "ls-tree", "-r", "--name-only", newsha], repo_root)
                else:
                    out = _run_git(["--git-dir", sub_gitdir, "ls-tree", "-r", "--name-only", newsha], repo_root)
                for f in out.splitlines():
                    if f.strip():
                        changes.append(("A", f"{sub_path}/{f}"))
            elif deleted:
                if Path(sub_gitdir).is_dir():
                    out = _run_git(["--git-dir", sub_gitdir, "ls-tree", "-r", "--name-only", oldsha], repo_root)
                else:
                    out = _run_git(["-C", sub_abs, "ls-tree", "-r", "--name-only", oldsha], repo_root)
                for f in out.splitlines():
                    if f.strip():
                        changes.append(("D", f"{sub_path}/{f}"))
            else:
                # Regular diff inside submodule
                if Path(sub_abs).is_dir():
                    out = _run_git(["-C", sub_abs, "diff", "--name-status", oldsha, newsha], repo_root)
                else:
                    out = _run_git(["--git-dir", sub_gitdir, "diff", "--name-status", oldsha, newsha], repo_root)
                for line in out.splitlines():
                    cols = line.strip().split("\t")
                    if not cols:
                        continue
                    st = cols[0]
                    if st.startswith("R") and len(cols) >= 3:
                        changes.append(("D", f"{sub_path}/{cols[1]}"))
                        changes.append(("M", f"{sub_path}/{cols[2]}"))
                    elif len(cols) >= 2:
                        changes.append((st, f"{sub_path}/{cols[1]}"))
        except Exception as e:
            append_error("codechat/.vector_sync_errors.jsonl", sub_path, "diff-submodule", str(e))

    return changes


def main_entry():
    args = parse_args()

    # Validate required API keys
    missing_keys = []
    if not os.environ.get("PINECONE_API_KEY"):
        missing_keys.append("PINECONE_API_KEY")
    if not os.environ.get("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")

    if missing_keys:
        if args.skip_on_missing_keys:
            print(f"[skip] Missing required API keys: {', '.join(missing_keys)}; skipping VectorDB sync.")
            sys.exit(0)
        else:
            raise RuntimeError(f"Missing required API keys: {', '.join(missing_keys)}")

    errors_out = args.errors_out
    files_to_process: List[Tuple[str, str]] = []

    if args.files:
        def parse_token(tok: str) -> tuple:
            if ":" in tok:
                st, path = tok.split(":", 1)
                st = st.strip().upper()
                if st not in {"A", "M", "D"}:
                    raise RuntimeError(f"Invalid status prefix in --files token: {tok}")
                return (st, path)
            return ("M", tok)

        for tok in args.files:
            files_to_process.append(parse_token(tok))
    elif args.retry_errors:
        src = Path(args.retry_errors)
        if src.exists():
            for line in src.read_text(encoding="utf-8").splitlines():
                try:
                    obj = json.loads(line)
                    fp = obj.get("file_path")
                    st = (obj.get("status") or "M").upper()
                    if st not in {"A", "M", "D"}:
                        st = "M"
                    if fp:
                        files_to_process.append((st, fp))
                except Exception:
                    continue
        else:
            raise RuntimeError(f"Errors file not found: {args.retry_errors}")
    else:
        from_commit = args.from_commit
        to_commit = args.to_commit or "HEAD"

        # Auto-detect commit range from GitHub Actions if not explicitly provided
        if not from_commit and not args.changed_files:
            from_commit, to_commit = detect_github_commit_range()
            print(f"[info] Auto-detected commit range from GitHub Actions: {from_commit}..{to_commit}")

        if from_commit:
            changes = compute_changes_from_git(args.repo_root, from_commit, to_commit)
            for st, fp in changes:
                if st.startswith('R'):
                    continue
                files_to_process.append((st, fp))
        elif args.changed_files:
            # Parse changed files
            with open(args.changed_files, 'r', encoding='utf-8') as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            for line in lines:
                parts = line.split(None, 2)
                status = parts[0]
                if status.startswith('R'):
                    if len(parts) >= 3:
                        old_fp = parts[1];
                        new_fp = parts[2]
                        files_to_process.append(("D", old_fp))
                        files_to_process.append(("M", new_fp))
                    else:
                        raise RuntimeError(f"Malformed rename line: {line}")
                else:
                    fp = parts[1] if len(parts) > 1 else ""
                    if fp:
                        files_to_process.append((status, fp))
        else:
            raise RuntimeError("Usage: provide one of: --files, --retry-errors, --from-commit, or changed_files path")

    run_sync(files_to_process, errors_out)


def run_sync(files_to_process: List[Tuple[str, str]], errors_out: str):
    # Environment context
    repo_name = os.getenv("GITHUB_REPOSITORY", "unknown").split("/")[-1]
    commit_sha = os.getenv("GITHUB_SHA", "unknown")

    # Initialize external clients and tokenizer lazily
    global index, openai_client, tokenizer, pc
    index_name = os.getenv("PINECONE_INDEX", INDEX_NAME)
    api_key = os.environ.get("PINECONE_API_KEY", "")

    if USE_SERVERLESS:
        from pinecone import Pinecone, ServerlessSpec  # type: ignore
        pc = Pinecone(api_key=api_key)
        # Ensure index exists (serverless)
        try:
            idxs = pc.list_indexes()
            names = []
            if hasattr(idxs, "indexes"):
                names = [ix.name for ix in idxs.indexes]
            else:
                names = [getattr(ix, "name", None) or ix.get("name") for ix in idxs]  # type: ignore
            if index_name not in set([n for n in names if n]):
                cloud = os.getenv("PINECONE_CLOUD", "aws")
                region = os.getenv("PINECONE_REGION", "us-east-1")
                print(
                    f"[info] Creating Pinecone serverless index '{index_name}' (dim={DIMENSION}, metric={METRIC}, cloud={cloud}, region={region})")
                try:
                    pc.create_index(
                        name=index_name,
                        dimension=DIMENSION,
                        metric=METRIC,
                        spec=ServerlessSpec(cloud=cloud, region=region),
                    )
                except Exception as e:
                    print(f"[warn] create_index failed or already exists: {e}")
        except Exception as e:
            print(f"[warn] Could not list/create serverless index: {e}")
        index = pc.Index(index_name)
    else:
        if pinecone_client is None:
            raise RuntimeError(
                "Pinecone SDK not available. Install 'pinecone' for serverless or 'pinecone-client' for classic.")
        pinecone_client.init(
            api_key=api_key,
            environment=os.getenv("PINECONE_ENV", "us-west1-gcp")
        )
        # Ensure index exists (classic client)
        try:
            existing = []
            try:
                existing = pinecone_client.list_indexes()
            except Exception:
                existing = []
            if isinstance(existing, dict) and "indexes" in existing:
                existing = [ix.get("name") for ix in existing.get("indexes", [])]
            if index_name not in set(existing or []):
                print(f"[info] Creating Pinecone classic index '{index_name}' (dim={DIMENSION}, metric={METRIC})")
                try:
                    pinecone_client.create_index(index_name, dimension=DIMENSION, metric=METRIC)
                except Exception as e:
                    print(f"[warn] create_index failed: {e}")
        except Exception as e:
            print(f"[warn] Could not verify/create classic index '{index_name}': {e}")
        index = pinecone_client.Index(index_name)

    oai_key = os.environ.get("OPENAI_API_KEY", "")
    if not oai_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    openai_client = OpenAI(api_key=oai_key)
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Initialize LlamaChunker for unified chunking
    global chunker
    chunker = LlamaChunker()

    print(f"[info] Starting VectorDB sync for {repo_name} (commit: {commit_sha[:8]})")
    print(f"[info] Using LlamaIndex for intelligent code-aware chunking")

    to_upsert: List[dict] = []
    delete_operations: List[str] = []
    file_stats = {"processed": 0, "errors": 0, "skipped": 0}
    failures: List[dict] = []

    # Process files
    for status, filepath in files_to_process:
        try:
            if status == "D":
                delete_operations.append(filepath)
                continue
            if not Path(filepath).exists():
                file_stats["skipped"] += 1
                raise FileNotFoundError(f"File marked as {status} but not found: {filepath}")
            if status in ("A", "M"):
                delete_operations.append(filepath)
            chunks = process_file(filepath, status, repo_name)
            to_upsert.extend(chunks)
            file_stats["processed"] += 1
        except Exception as e:
            file_stats["errors"] += 1
            failures.append({"file_path": filepath, "operation": "process", "message": str(e), "status": status})
            append_error(errors_out, filepath, "process", str(e), status=status)

    print(f"\n[info] Deleting vectors for {len(delete_operations)} files...")
    for filepath in delete_operations:
        try:
            safe_delete_vectors(filepath, repo_name)
        except Exception as e:
            file_stats["errors"] += 1
            failures.append({"file_path": filepath, "operation": "delete", "message": str(e), "status": "D"})
            append_error(errors_out, filepath, "delete", str(e), status="D")

    upserted_ids: List[str] = []
    total_upserted = 0
    if to_upsert:
        print(f"\n[info] Upserting {len(to_upsert)} chunks in batches of {BATCH_SIZE}...")
        for i in tqdm(range(0, len(to_upsert), BATCH_SIZE), desc="Upserting"):
            batch = to_upsert[i:i + BATCH_SIZE]
            try:
                upserted_count = safe_upsert_batch(batch, repo_name)
                total_upserted += upserted_count
                for it in batch:
                    uid = it.get('id')
                    if isinstance(uid, str):
                        upserted_ids.append(uid)
            except Exception as e:
                file_stats["errors"] += 1
                paths_statuses = set(
                    (it.get("metadata", {}).get("file_path"), it.get("metadata", {}).get("status", "M")) for it in
                    batch)
                for fp, st in paths_statuses:
                    failures.append(
                        {"file_path": fp or "<unknown>", "operation": "upsert", "message": str(e), "status": st})
                    append_error(errors_out, fp or "<unknown>", "upsert", str(e), status=st or "M")

    print(f"\n[info] Sync Complete for {repo_name}:")
    print(f"  - Files processed: {file_stats['processed']}")
    print(f"  - Files skipped: {file_stats['skipped']}")
    print(f"  - Files with errors: {file_stats['errors']}")
    print(f"  - Vectors deleted: {len(delete_operations)} files")
    print(f"  - Chunks upserted: {total_upserted}")
    print(f"  - Embedding model: text-embedding-3-small")
    if failures:
        print(
            f"[error] {len(failures)} failures encountered. See {errors_out} for details. Use --retry-errors to re-run.")
        for f in failures:
            print(
                f"  - failure: op={f.get('operation')} status={f.get('status')} file={f.get('file_path')} message={f.get('message')}")
        raise SystemExit(1)
    return {"namespace": repo_name, "upserted_ids": upserted_ids}


if __name__ == "__main__":
    main_entry()


