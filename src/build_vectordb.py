"""
build_vectordb.py

Load pre-chunked JSON files (already split into headed sections upstream)
and embed + store them in a persistent local ChromaDB collection.

Expected input: a folder of *.json files, one per source page, shaped like:

{
  "file": "About_Mtech.md",
  "frontmatter": {
    "title": "...",
    "source_type": "KU website",
    "category": "academic_program",
    "department": "Artificial Intelligence",
    "degree_level": "PostGraduate",
    "last_updated": "2026-07-25"
  },
  "chunk_count": 4,
  "warning": null,
  "chunks": [
    {"index": 0, "heading": "Program Overview", "char_count": 189, "text": "..."},
    ...
  ]
}

Usage:
    python build_vectordb.py --input doai_output_chunks --db_path ./chroma_db
    python build_vectordb.py --input doai_output_chunks --db_path ./chroma_db --test_query "tell me about ku canteen"
"""

import argparse
import hashlib
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Metadata fields pulled out of "frontmatter" and flattened onto every chunk
# belonging to that file. ChromaDB metadata values must be str/int/float/bool
# (no nested dicts), which is exactly why this flattening step exists.
FRONTMATTER_FIELDS = [
    "title",
    "source_type",
    "category",
    "department",
    "degree_level",
    "last_updated",
]


def load_chunked_documents(input_dir: Path):
    """Read every *.json file in input_dir and return (ids, texts, metadatas)."""
    all_ids, all_texts, all_metas = [], [], []
    seen = {}
    skipped = []

    for path in sorted(input_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            skipped.append((path.name, f"invalid JSON: {e}"))
            continue

        source_file = data.get("file", path.stem)
        chunks = data.get("chunks", [])
        if not chunks:
            skipped.append((path.name, "no 'chunks' array or it's empty"))
            continue

        frontmatter = data.get("frontmatter", {}) or {}
        doc_chunk_count = data.get("chunk_count", len(chunks))
        total_doc_chars = sum(ch.get("char_count", len(ch.get("text", ""))) for ch in chunks)

        for ch in chunks:
            text = ch.get("text", "")
            if not text.strip():
                continue

            idx = ch.get("index")
            chunk_id = hashlib.md5(f"{source_file}::{idx}".encode()).hexdigest()

            chunk_id = hashlib.md5(f"{source_file}::{idx}".encode()).hexdigest()

            if chunk_id in seen:
                print("\n" + "=" * 60)
                print("Duplicate ID detected!")
                print(f"ID          : {chunk_id}")
                print(f"Current JSON: {path.name}")
                print(f"Source file : {source_file}")
                print(f"Chunk index : {idx}")
                print("Previous occurrence:")
                print(seen[chunk_id])
                raise ValueError("Duplicate chunk ID found")
            else:
                seen[chunk_id] = {
                    "json_file": path.name,
                    "source_file": source_file,
                    "chunk_index": idx,
                }

            meta = {
                "source_file": source_file,
                "section": ch.get("heading") or "root",
                "chunk_index": idx,
                "char_count": ch.get("char_count", len(text)),
                "chunk_count": doc_chunk_count,
                "total_doc_chars": total_doc_chars,
            }
            for field in FRONTMATTER_FIELDS:
                # Chroma metadata can't store None - fall back to "" so the
                # field is still present and filterable.
                meta[field] = frontmatter.get(field) or ""

            all_ids.append(chunk_id)
            all_texts.append(text)
            all_metas.append(meta)

    return all_ids, all_texts, all_metas, skipped


def build_vectordb(input_dir: str, db_path: str, collection_name: str = "doai_site"):
    input_dir = Path(input_dir).resolve()
    db_path = str(Path(db_path).resolve())

    print(f"Loading chunked JSON files from {input_dir} ...")
    ids, texts, metas, skipped = load_chunked_documents(input_dir)

    if skipped:
        print(f"  Skipped {len(skipped)} file(s):")
        for name, reason in skipped:
            print(f"    - {name}: {reason}")

    print(f"  {len(texts)} chunks loaded from "
          f"{len({m['source_file'] for m in metas})} source file(s)")

    if not texts:
        print("  Nothing to embed - check --input path and that the JSON files "
              "match the expected shape. Aborting before touching the DB.")
        return None

    print(f"Loading embedding model '{EMBEDDING_MODEL}' ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding chunks ...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32).tolist()

    print(f"Writing to ChromaDB at {db_path} ...")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(collection_name)

    # upsert in batches so re-running after re-chunking updates rather than duplicates
    batch = 100
    for i in range(0, len(ids), batch):
        collection.upsert(
            ids=ids[i:i + batch],
            embeddings=embeddings[i:i + batch],
            documents=texts[i:i + batch],
            metadatas=metas[i:i + batch],
        )

    print(f"Done. Collection '{collection_name}' at {db_path} now has {collection.count()} chunks.")
    return collection



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data_prep\\doai_output_chunks", help="Folder with pre-chunked *.json files")
    parser.add_argument("--db_path", default="./chroma_db", help="ChromaDB persistence folder")
    parser.add_argument("--collection", default="doai_site")
    args = parser.parse_args()

    resolved_db_path = str(Path(args.db_path).resolve())
    print(f"(using db_path: {resolved_db_path})")

    collection = build_vectordb(args.input, resolved_db_path, args.collection)
