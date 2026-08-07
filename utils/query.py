import argparse
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data_prep" / "chroma_db"

# Loaded once at module level - reused across every query instead of
# reloading the model from disk on each call. Matters once this is
# wired into a live call loop with multiple turns.
_model = SentenceTransformer(EMBEDDING_MODEL)


# If the single best-matching document has this many chunks or fewer,
# assume the caller wants the whole thing (e.g. a short syllabus or
# overview page) rather than just the top-k closest pieces of it.
WHOLE_DOC_THRESHOLD = 8
EXPAND_MAX_CHUNKS = 5
EXPAND_MAX_TOTAL_CHARS = 1500


def _open_collection(db_path: str, collection_name: str):
    resolved_path = str(Path(db_path).resolve())
    client = chromadb.PersistentClient(path=resolved_path)

    existing = {c.name for c in client.list_collections()}
    if collection_name not in existing:
        raise SystemExit(
            f"Collection '{collection_name}' does not exist at {resolved_path}.\n"
            f"Available collections there: {sorted(existing) or '(none)'}\n"
            f"Did you run build_vectordb.py with the same --db_path / --collection?"
        )

    collection = client.get_collection(collection_name)
    if collection.count() == 0:
        raise SystemExit(
            f"Collection '{collection_name}' at {resolved_path} exists but has 0 chunks.\n"
            f"Re-run build_vectordb.py pointed at this same --db_path to populate it."
        )
    return collection, resolved_path


def get_whole_document(collection, source_file: str):
    """Fetch every chunk belonging to one source file, in original order."""
    result = collection.get(
        where={"source_file": source_file},
        include=["documents", "metadatas"],
    )
    pairs = list(zip(result["documents"], result["metadatas"]))
    pairs.sort(key=lambda p: p[1]["chunk_index"])
    return pairs


def query(db_path: str, collection_name: str, question: str, top_k: int = 3):
    collection, resolved_path = _open_collection(db_path, collection_name)
    print(f"(querying collection '{collection_name}' at {resolved_path}, "
          f"{collection.count()} chunks total)")

    q_emb = _model.encode([question]).tolist()

    # Step 1: find the single closest chunk to identify which document
    # the question is actually about.
    top_hit = collection.query(query_embeddings=q_emb, n_results=1)
    top_meta = top_hit["metadatas"][0][0]
    source_file = top_meta["source_file"]
    doc_chunk_count = top_meta.get("chunk_count", 0)

    if doc_chunk_count and doc_chunk_count <= WHOLE_DOC_THRESHOLD:
        # Step 2a: short reference doc - pull ALL its chunks, in order.
        pairs = get_whole_document(collection, source_file)
        print(f"\n(returning all {len(pairs)} chunks from {source_file})")
        for i, (doc, meta) in enumerate(pairs):
            print(f"\n--- part {i + 1} (source: {meta['source_file']}, "
                  f"section: {meta['section']}, title: {meta.get('title')}) ---")
            print(doc[:400])
    else:
        # Step 2b: longer document - fall back to normal top-k semantic search.
        results = collection.query(query_embeddings=q_emb, n_results=top_k)
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            print(f"\n--- match {i + 1} (source: {meta['source_file']}, "
                  f"section: {meta['section']}, title: {meta.get('title')}) ---")
            print(results["documents"][0][i][:400])


def query_with_expansion(collection, query_embedding, top_k: int = 5):
    """Top-k semantic search, plus: for any hit that belongs to a small doc
    (few chunks, short overall), pull in its sibling chunks too so the
    answer isn't missing context from the rest of that short page."""
    initial = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    seen_ids = set(initial["ids"][0])
    seen_source_files = set()
    final_chunks = []

    for i, chunk_id in enumerate(initial["ids"][0]):
        meta = initial["metadatas"][0][i]
        text = initial["documents"][0][i]
        source_file = meta.get("source_file")

        final_chunks.append({"id": chunk_id, "text": text, "metadata": meta, "expanded": False})

        chunk_count = meta.get("chunk_count", 999)
        total_chars = meta.get("total_doc_chars", 999999)
        is_small_doc = chunk_count <= EXPAND_MAX_CHUNKS and total_chars <= EXPAND_MAX_TOTAL_CHARS

        if is_small_doc and source_file not in seen_source_files:
            seen_source_files.add(source_file)
            siblings = collection.get(where={"source_file": source_file})
            for j, sib_id in enumerate(siblings["ids"]):
                if sib_id not in seen_ids:
                    seen_ids.add(sib_id)
                    final_chunks.append({
                        "id": sib_id,
                        "text": siblings["documents"][j],
                        "metadata": siblings["metadatas"][j],
                        "expanded": True,  # pulled in via sibling expansion, not the original match
                    })

    return final_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db_path",
        default=str(DB_PATH)
    )
    parser.add_argument("--collection", default="doai_site")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--test_query", required=True)

    args = parser.parse_args()

    print("DB PATH =", args.db_path)

    query(
        args.db_path,
        args.collection,
        args.test_query,
        args.top_k,
    )