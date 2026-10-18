"""Help locate where References and Appendix start in each paper (for papers.yaml overrides).

Usage:
    python scripts/find_appendix.py            all papers
    python scripts/find_appendix.py P4 P1      only these

For every page it prints lines that LOOK like a references heading or an appendix
section heading ("References", "Appendix", "A Additional Results", "B.1 Prompts", ...).
These are only candidates: open the PDF at that page and confirm before writing
references_start_page / appendix_start_page into papers.yaml.
"""
import re
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

# Case-sensitive on purpose: "supplementary." inside a sentence should not match.
# Lettered headings come in two styles: "A Details" (ACL/COLING) and "A. Details" (CVPR/arXiv).
KEYWORD_RE = re.compile(r"^(References|REFERENCES|Bibliography|Appendix|APPENDIX|Appendices"
                        r"|Supplementary|SUPPLEMENTARY)\b")
LETTERED_RE = re.compile(r"^[A-H](\.\d+)*\.?\s+[A-Z][A-Za-z]")


def is_candidate(line: str) -> bool:
    if len(line) >= 70:
        return False
    if KEYWORD_RE.match(line):
        return True
    # "A. Radford, J. Wu, ..." is an author line in the references, not a heading
    return bool(LETTERED_RE.match(line)) and "," not in line


def main() -> None:
    wanted = sys.argv[1:]
    for pdf_path in sorted(RAW_DIR.glob("P*.pdf"), key=lambda p: int(p.stem[1:])):
        if wanted and pdf_path.stem not in wanted:
            continue
        doc = pymupdf.open(pdf_path)
        print(f"\n===== {pdf_path.stem} ({len(doc)} pages) =====")
        for page_index, page in enumerate(doc):
            hits = [l.strip() for l in page.get_text("text").splitlines() if is_candidate(l.strip())]
            if hits:
                print(f"  p.{page_index + 1:>2}: " + " | ".join(hits[:4]))


if __name__ == "__main__":
    main()
