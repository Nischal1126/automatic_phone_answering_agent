import argparse
import json
import re
from pathlib import Path

import yaml

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?\n)---\s*\n?", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def parse_frontmatter(text: str):
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return {}, text
    body = text[m.end():]
    flat_meta = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            flat_meta[k] = v
        else:
            flat_meta[k] = str(v)
    return flat_meta, body


def split_by_headings(text: str):
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]

    sections = []
    heading_stack = []

    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        heading_stack = [h for h in heading_stack if h[0] < level]
        heading_stack.append((level, title))
        heading_path = " > ".join(t for _, t in heading_stack)

        if body:
            sections.append((heading_path, body))

    if not sections:
        return [("", text)]
    return sections


def recursive_split(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        slice_ = text[start:end]
        if end < len(text):
            last_break = max(slice_.rfind("\n\n"), slice_.rfind(". "))
            if last_break > size * 0.5:
                end = start + last_break + 1
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]


def chunk_document(body_text: str):
    chunks = []
    sections = split_by_headings(body_text)
    for heading_path, body in sections:
        for piece in recursive_split(body):
            chunks.append({
                "text": (f"{heading_path}\n{piece}" if heading_path else piece),
                "heading": heading_path,
            })
    return chunks


# ---------------------------------------------------------------------------
# Preview runner
# ---------------------------------------------------------------------------
def preview_folder(input_dir: str, full_text: bool = False, output_dir: str = None):
    input_dir = Path(input_dir)
    md_files = sorted(input_dir.glob("*.md"))

    if not md_files:
        print(f"No .md files found in {input_dir}")
        return []

    out_path = None
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

    all_results = []
    total_chunks = 0
    oversized_chunks = 0

    for md_file in md_files:
        raw_text = md_file.read_text(encoding="utf-8", errors="ignore")
        frontmatter, body = parse_frontmatter(raw_text)
        chunks = chunk_document(body)

        # Flag the "silent single blob" failure mode: everything collapsed into
        # one chunk with no heading path, but raw '#' characters are still in
        # it -- almost always means a heading had no body text below it (e.g.
        # a fact written directly into the heading itself).
        warning = None
        if len(chunks) == 1 and chunks[0]["heading"] == "" and "\n#" in ("\n" + chunks[0]["text"]):
            warning = ("Collapsed into a single chunk with no heading structure recognized. "
                       "This usually means a heading had no body text below it (the fact was "
                       "written into the heading line itself) -- check for headings with "
                       "nothing but a blank line after them.")

        file_result = {
            "file": md_file.name,
            "frontmatter": frontmatter,
            "chunk_count": len(chunks),
            "warning": warning,
            "chunks": [
                {"index": i, "heading": c["heading"] or None, "char_count": len(c["text"]), "text": c["text"]}
                for i, c in enumerate(chunks)
            ],
        }
        all_results.append(file_result)
        total_chunks += len(chunks)

        # Write this file's own chunk results to <stem>.json in the output folder
        if out_path:
            json_path = out_path / f"{md_file.stem}.json"
            json_path.write_text(json.dumps(file_result, indent=2, default=str), encoding="utf-8")

        print(f"\n{'=' * 70}")
        print(f"FILE: {md_file.name}" + (f"  ->  {out_path / (md_file.stem + '.json')}" if out_path else ""))
        print(f"{'=' * 70}")
        if frontmatter:
            print(f"Frontmatter: {frontmatter}")
        if warning:
            print(f"WARNING: {warning}")
        print(f"Chunks produced: {len(chunks)}\n")

        for i, c in enumerate(chunks):
            heading = c["heading"] or "(no heading)"
            char_count = len(c["text"])
            if char_count > CHUNK_SIZE * 1.1:
                oversized_chunks += 1
            print(f"  [{i}] {heading}  ({char_count} chars)")
            if full_text:
                print(f"      {c['text']}")
            else:
                preview = c["text"][:150].replace(chr(10), " ")
                print(f"      {preview}{'...' if len(c['text']) > 150 else ''}")

    print(f"\n{'=' * 70}")
    print(f"TOTAL: {len(md_files)} file(s), {total_chunks} chunk(s), "
          f"{oversized_chunks} chunk(s) over {CHUNK_SIZE} chars")
    if out_path:
        print(f"Per-file JSON written to: {out_path}/")
    print(f"{'=' * 70}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder containing .md files")
    parser.add_argument("--output", default=None,
                         help="Folder to write one <filename>.json per <filename>.md into "
                              "(e.g. about_mtech.md -> <output>/about_mtech.json)")
    parser.add_argument("--save", default=None, help="Optional: also save ALL results combined into one JSON file")
    parser.add_argument("--full-text", action="store_true", help="Print full chunk text instead of a preview")
    args = parser.parse_args()

    results = preview_folder(args.input, full_text=args.full_text, output_dir=args.output)

    if args.save and results:
        Path(args.save).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"\nCombined results also saved to {args.save}")