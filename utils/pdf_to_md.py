"""
Extract text (and tables) from PDF file(s) and save as .md, ready to feed
into a vector DB ingestion pipeline (e.g. build_vectordb.py).

Accepts either a single PDF or a folder of PDFs.

Usage:
    python pdf_to_md.py mydoc.pdf --output extracted_md
    python pdf_to_md.py pdf_folder/ --output extracted_md
"""

import argparse
import datetime
from pathlib import Path

import pdfplumber


def table_to_markdown(table: list[list]) -> str:
    """Convert a pdfplumber-extracted table (list of rows) into a markdown table."""
    header = [(c or "").strip() for c in table[0]]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in table[1:]:
        cells = [(c or "").strip().replace("\n", " ") for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def extract_pdf_to_markdown(pdf_path: Path) -> tuple[str, int]:
    """Returns (markdown_text, page_count). Extracts plain text per page, plus
    any detected tables appended as clean markdown tables.

    Note: pdfplumber's page.extract_text() doesn't exclude table regions, so
    a table's cell contents may appear twice -- once as jumbled inline text,
    once as a clean markdown table below it. For text-heavy documents this
    rarely matters; for table-heavy documents, spot-check the output.
    """
    page_parts = []
    page_count = 0

    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            section = []

            text = page.extract_text()
            if text and text.strip():
                section.append(text.strip())

            tables = page.extract_tables()
            for table in tables:
                if table and any(any(cell for cell in row) for row in table):
                    section.append(table_to_markdown(table))

            if section:
                page_parts.append("\n\n".join(section))

    return "\n\n".join(page_parts).strip(), page_count


def convert_pdf(pdf_path: Path, output_dir: Path) -> bool:
    md_text, page_count = extract_pdf_to_markdown(pdf_path)

    if not md_text:
        print(f"[WARN] {pdf_path.name}: no extractable text found "
              f"({page_count} page(s)) -- likely a scanned/image-only PDF, needs OCR.")
        return False

    frontmatter = (
        "---\n"
        f'source_file: "{pdf_path.name}"\n'
        f"page_count: {page_count}\n"
        f"extracted_on: {datetime.date.today().isoformat()}\n"
        "source_type: scraped\n"
        "---\n\n"
    )

    out_path = output_dir / f"{pdf_path.stem}.md"
    out_path.write_text(frontmatter + md_text, encoding="utf-8")
    print(f"[OK] {pdf_path.name} -> {out_path.name} ({page_count} pages, {len(md_text)} chars)")
    return True


def run(input_path: str, output: str):
    input_path = Path(input_path)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        pdf_files = [input_path]
    elif input_path.is_dir():
        pdf_files = sorted(input_path.glob("*.pdf"))
    else:
        print(f"Path not found: {input_path}")
        return

    if not pdf_files:
        print(f"No PDF files found at {input_path}")
        return

    ok_count = 0
    for pdf_file in pdf_files:
        if convert_pdf(pdf_file, output_dir):
            ok_count += 1

    print(f"\nDone. {ok_count}/{len(pdf_files)} PDF(s) converted -> {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="A single .pdf file or a folder containing .pdf files")
    parser.add_argument("--output", default="extracted_md", help="Folder to save .md files into")
    args = parser.parse_args()

    run(args.input, args.output)