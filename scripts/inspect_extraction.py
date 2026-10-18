"""Extract text from each PDF and write a human-readable dump for manual inspection.

Usage (from repo root):
    python scripts/inspect_extraction.py

Output:
    data/inspect/<ID>.txt   full extracted text with ===== PAGE n ===== markers
    a summary table printed to the console

Goal: LOOK at the text before building anything on it. Garbage in the chunks
means garbage in retrieval, and you need to know which problems come from extraction.
"""
from pathlib import Path

import pymupdf  # PyMuPDF
import yaml

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "papers.yaml"
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "inspect"
LOW_TEXT_CHARS = 500  # pages below this are probably mostly figures/tables


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    papers = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["papers"]

    print(f"{'ID':<4} {'pages':>5} {'chars':>8} {'low-text pages':<20} first line of page 1")
    total_pages = 0
    for paper in papers:
        pdf_path = RAW_DIR / f"{paper['id']}.pdf"
        if not pdf_path.exists():
            print(f"{paper['id']:<4} MISSING — run download_papers.py first")
            continue

        doc = pymupdf.open(pdf_path)
        lines, low_pages, total_chars = [], [], 0
        for page_index, page in enumerate(doc):
            text = page.get_text("text")  # default reading order from the PDF content stream
            page_no = page_index + 1      # humans count pages from 1
            total_chars += len(text)
            if len(text) < LOW_TEXT_CHARS:
                low_pages.append(page_no)
            lines.append(f"\n===== PAGE {page_no} =====\n{text}")

        (OUT_DIR / f"{paper['id']}.txt").write_text("".join(lines), encoding="utf-8")
        first_line = doc[0].get_text("text").strip().split("\n")[0][:60]
        total_pages += len(doc)
        print(f"{paper['id']:<4} {len(doc):>5} {total_chars:>8} {str(low_pages)[:20]:<20} {first_line}")

    print(f"\nTotal pages: {total_pages}. Dumps written to {OUT_DIR}")


if __name__ == "__main__":
    main()
