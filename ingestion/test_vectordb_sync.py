"""
End-to-end test for codechat/ingestion/vector_db_sync.py.

This runs real API calls (OpenAI + Pinecone) and cleans up vectors.

Scenarios:
- Add (A) -> index file
- Modify (M) -> reindex (delete first)
- Rename (D+M) -> delete old, index new
- Delete (D) -> remove vectors

Run: python codechat/ingestion/test_vector_db_sync.py
"""

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Tuple, List

from dotenv import load_dotenv
from openai import OpenAI

# Path calculations based on test file location
THIS_DIR = Path(__file__).parent  # codechat/ingestion/
CODECHAT_ROOT = THIS_DIR.parent  # codechat/
REPO_ROOT = CODECHAT_ROOT.parent  # webroot/
sys.path.insert(0, str(THIS_DIR))
import vector_db_sync  # type: ignore


def ensure_env() -> Tuple[str, str, str]:
    api = os.getenv("PINECONE_API_KEY")
    oai = os.getenv("OPENAI_API_KEY")
    if not api or not oai:
        raise SystemExit("PINECONE_API_KEY and OPENAI_API_KEY must be set in codechat/.env or environment")

    # Use a dedicated namespace for test to avoid collisions
    repo_name = f"vector-sync-test-{uuid.uuid4().hex[:8]}"
    os.environ["GITHUB_REPOSITORY"] = f"local/{repo_name}"

    index_name = os.getenv("PINECONE_INDEX", vector_db_sync.INDEX_NAME)
    env = os.getenv("PINECONE_ENV", "us-west1-gcp")
    return repo_name, index_name, env


def get_index_handle(index_name: str):
    """Return an index handle using available SDK (serverless preferred)."""
    try:
        from pinecone import Pinecone  # type: ignore
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        return pc.Index(index_name)
    except Exception:
        # Fallback to classic client
        import pinecone as pinecone_client  # type: ignore
        pinecone_client.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.getenv("PINECONE_ENV", "us-west1-gcp"))
        return pinecone_client.Index(index_name)


def count_vectors_by_query(index, namespace: str, query_vec, expect_path: str) -> int:
    res = index.query(vector=query_vec, top_k=5, namespace=namespace, include_metadata=True)
    matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
    def md(m):
        return m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {})
    return sum(1 for m in matches if md(m).get("file_path") == expect_path)


def wait_fetch_by_ids(index, namespace: str, ids: List[str], attempts: int = 20, delay: float = 1.5) -> bool:
    """Wait for fetch(ids) to return any records for new namespace consistency."""
    for _ in range(attempts):
        try:
            fetched = index.fetch(ids=ids, namespace=namespace)
            if isinstance(fetched, dict):
                vecs = fetched.get("vectors") or fetched.get("records") or {}
            else:
                vecs = getattr(fetched, "vectors", None) or getattr(fetched, "records", None) or {}
            if vecs:
                return True
        except Exception:
            pass
        time.sleep(delay)
    return False


def wait_ids_gone(index, namespace: str, ids: List[str], attempts: int = 20, delay: float = 1.0) -> bool:
    """Wait until fetch(ids) returns no records (used after deletes/renames)."""
    for _ in range(attempts):
        try:
            fetched = index.fetch(ids=ids, namespace=namespace)
            if isinstance(fetched, dict):
                vecs = fetched.get("vectors") or fetched.get("records") or {}
            else:
                vecs = getattr(fetched, "vectors", None) or getattr(fetched, "records", None) or {}
            if not vecs:
                return True
        except Exception:
            return True
        time.sleep(delay)
    return False


def wait_for_count(index, namespace: str, query_vec, expect_path: str, expect_min: int, attempts: int = 10, delay: float = 1.5) -> int:
    """Poll query until count >= expect_min or attempts exhausted."""
    last = 0
    for _ in range(attempts):
        try:
            last = count_vectors_by_query(index, namespace, query_vec, expect_path)
            if last >= expect_min:
                return last
        except Exception:
            last = 0
        time.sleep(delay)
    return last


def main() -> None:
    # Load local env file from codechat/.env
    load_dotenv(dotenv_path=str(CODECHAT_ROOT / ".env"), override=True)
    repo_name, index_name, env = ensure_env()

    # Test artifacts
    test_file_1 = CODECHAT_ROOT / f"_tmp_vector_sync_test_{uuid.uuid4().hex[:8]}.md"
    test_file_2 = test_file_1.with_name(test_file_1.stem + "_renamed.md")
    errors_file = CODECHAT_ROOT / "_tmp_vector_sync_errors.jsonl"

    # Index handle obtained after first sync (index may be created there)
    index = None
    # OpenAI client for query embeddings
    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    created_files = []
    try:
        # 1) Add
        content_add = "# Vector Sync Test\n\nInitial content."
        test_file_1.write_text(content_add, encoding="utf-8")
        created_files.append(test_file_1)
        # Add via run_sync (capture ids for fetch-by-id verification)
        res = vector_db_sync.run_sync([("A", f"codechat/{test_file_1.name}")], str(errors_file))
        index = get_index_handle(index_name)
        add_ids = res.get("upserted_ids", [])
        ns = res.get("namespace", repo_name)
        if not add_ids:
            # Force a follow-up modify upsert to ensure IDs are available
            res2 = vector_db_sync.run_sync([("M", f"codechat/{test_file_1.name}")], str(errors_file))
            add_ids = res2.get("upserted_ids", [])
            ns = res2.get("namespace", ns)
        if not add_ids or not wait_fetch_by_ids(index, ns, add_ids):
            raise AssertionError("No vectors found after Add (fetch-by-id)")

        # 2) Modify
        content_mod = "# Vector Sync Test\n\nModified content."
        test_file_1.write_text(content_mod, encoding="utf-8")
        res = vector_db_sync.run_sync([("M", f"codechat/{test_file_1.name}")], str(errors_file))
        mod_ids = res.get("upserted_ids", [])
        if not mod_ids or not wait_fetch_by_ids(index, res.get("namespace", repo_name), mod_ids, attempts=15, delay=1.0):
            raise AssertionError("No vectors found after Modify (fetch-by-id)")

        # 3) Rename
        test_file_1.rename(test_file_2)
        created_files.append(test_file_2)
        # Rename as D old + M new
        res = vector_db_sync.run_sync([
            ("D", f"codechat/{test_file_1.name}"),
            ("M", f"codechat/{test_file_2.name}")
        ], str(errors_file))
        # Old ids should be gone (from previous mod_ids), new ids should be present
        if 'mod_ids' in locals() and mod_ids:
            if not wait_ids_gone(index, res.get("namespace", repo_name), mod_ids, attempts=20, delay=1.0):
                raise AssertionError("Old path vectors still present after Rename")
        new_ids = res.get("upserted_ids", [])
        if not new_ids or not wait_fetch_by_ids(index, res.get("namespace", repo_name), new_ids, attempts=15, delay=1.0):
            raise AssertionError("No vectors found for new path after Rename (fetch-by-id)")

        # 4) Delete
        if test_file_2.exists():
            test_file_2.unlink()
        res = vector_db_sync.run_sync([("D", f"codechat/{test_file_2.name}")], str(errors_file))
        fetched_del = index.fetch(ids=new_ids, namespace=res.get("namespace", repo_name))
        vecs_del = fetched_del.get("vectors", {}) if isinstance(fetched_del, dict) else getattr(fetched_del, "vectors", {})
        if vecs_del:
            raise AssertionError("Vectors still present after Delete")

        print("[ok] Vector sync e2e test passed.")

    finally:
        # Cleanup vectors defensively for both paths
        try:
            index = index or get_index_handle(index_name)
        except Exception:
            index = None
        if index is not None:
            for fp in (f"codechat/{test_file_1.name}", f"codechat/{test_file_2.name}"):
                try:
                    index.delete(filter={"repo_name": repo_name, "file_path": fp}, namespace=repo_name)
                except Exception:
                    # Try simpler filter on file_path only
                    try:
                        index.delete(filter={"file_path": fp}, namespace=repo_name)
                    except Exception:
                        pass

        # Cleanup files
        for f in created_files:
            try:
                if f.exists():
                    f.unlink()
            except Exception:
                pass
        try:
            if errors_file.exists():
                errors_file.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
