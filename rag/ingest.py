"""Ingestion: PDF -> cleaned lines -> section labels -> overlapping token chunks.

Output: data/index/chunks.jsonl, one JSON object per chunk, for ALL section kinds.
Which kinds get indexed is decided later (config: chunking.index_sections).

Steps per document
1. Extract text page by page with PyMuPDF (page numbers start at 1).
2. Drop noise lines, looking only at the first/last 3 lines of each page: bare page
   numbers, and running headers/footers (= short edge lines that repeat on >= 50% of
   the pages, digits ignored). Restricting to page edges avoids deleting body text that
   happens to repeat.
3. Label every line as main / references / appendix:
   - If papers.yaml gives references_start_page / appendix_start_page (from manual
     inspection), those win, at page granularity.
   - Otherwise a heuristic: a line that is exactly "References"/"Bibliography" starts the
     references; after that, a short line starting with "Appendix"/"Supplementary" starts
     the appendix. Appendices placed BEFORE the references are not detected.
4. Track the latest heading as the section name for citations: numbered ("3.2 Method")
   or lettered appendix style ("A. Experiment Details", "B.1 Prompts"). The tracker follows
   the effective kind (overrides first), so appendix chunks never show "References".
5. Clean text: re-join words hyphenated across line breaks, collapse whitespace.
6. Chunk each contiguous block of one kind with a sliding window over tokenizer tokens
   (size/overlap from config). The chunk text is sliced from the original string using the
   tokenizer's character offsets, so no text is altered by decoding. A chunk records every
   page it overlaps.
"""
import json
import re
from collections import Counter, defaultdict

import pymupdf

from rag.config import load_config, load_manifest, resolve

PAGE_NUM_RE = re.compile(r"^\d{1,3}$")
REFERENCES_RE = re.compile(r"^(\d+\.?\s*)?(references|bibliography)$", re.IGNORECASE)
APPENDIX_RE = re.compile(r"^([A-Z]\.?\s+)?(appendix|appendices|supplementary)\b", re.IGNORECASE)
HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z][A-Za-z][^.]{1,70})$")  # "3.2 Method"
LETTERED_HEADING_RE = re.compile(r"^[A-H](\.\d+)*\.?\s+[A-Z][A-Za-z]")  # "A. Details", "B.1 Prompts"


def is_heading(line: str) -> bool:
    if len(line) >= 70 or "," in line:  # "A. Radford, J. Wu" is an author line, not a heading
        return False
    return bool(HEADING_RE.match(line) or LETTERED_HEADING_RE.match(line) or APPENDIX_RE.match(line))
HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")


def _norm(line: str) -> str:
    """Normalise a line for header/footer matching: lowercase, digits -> '#'."""
    return re.sub(r"\d+", "#", line.strip().lower())


EDGE_LINES = 3  # headers/footers live in the first/last few lines of a page


def edge_indices(lines: list[str]) -> set[int]:
    nonempty = [i for i, l in enumerate(lines) if l.strip()]
    return set(nonempty[:EDGE_LINES] + nonempty[-EDGE_LINES:])


def find_repeated_lines(pages_lines: list[list[str]], min_fraction: float = 0.5) -> set[str]:
    """Short lines that sit at a page edge on >= min_fraction of pages = running header/footer."""
    n_pages = len(pages_lines)
    if n_pages < 4:  # too few pages to tell a header from real text
        return set()
    counts = Counter()
    for lines in pages_lines:
        counts.update({_norm(lines[i]) for i in edge_indices(lines) if len(lines[i].strip()) < 100})
    return {line for line, c in counts.items() if c / n_pages >= min_fraction}


def kind_from_overrides(page_no: int, paper: dict) -> str | None:
    """Page-level labels from papers.yaml. Works in either order:
    main -> references -> appendix   (most papers)
    main -> appendix -> references   (e.g. appendix placed before the bibliography)"""
    starts = [(paper.get("references_start_page"), "references"),
              (paper.get("appendix_start_page"), "appendix")]
    starts = [(page, kind) for page, kind in starts if page is not None]
    if not starts:
        return None
    kind = "main"
    for start, label in sorted(starts):
        if page_no >= start:
            kind = label
    return kind


def clean_text(text: str) -> str:
    text = HYPHEN_BREAK_RE.sub(r"\1\2", text)  # "repre-\nsentation" -> "representation"
    return re.sub(r"\s+", " ", text).strip()


def extract_segments(pdf_path, paper: dict) -> tuple[list[dict], Counter]:
    """Return segments = runs of lines sharing (page, kind, heading), plus stats."""
    doc = pymupdf.open(pdf_path)
    pages_lines = [page.get_text("text").splitlines() for page in doc]
    repeated = find_repeated_lines(pages_lines)

    stats = Counter()
    segments: list[dict] = []
    heuristic_kind, heading, previous_kind = "main", "", "main"

    for page_index, lines in enumerate(pages_lines):
        page_no = page_index + 1
        current = None
        edges = edge_indices(lines)
        for line_index, raw in enumerate(lines):
            line = raw.strip()
            if not line:
                continue
            at_edge = line_index in edges
            if at_edge and (PAGE_NUM_RE.match(line) or _norm(line) in repeated):
                stats["dropped_noise_lines"] += 1
                continue

            # --- section kind: heuristic state machine (used only if no overrides) ---
            if heuristic_kind == "main" and REFERENCES_RE.match(line):
                heuristic_kind = "references"
                stats["references_heading_page"] = page_no
            elif heuristic_kind == "references" and len(line) < 60 and APPENDIX_RE.match(line):
                heuristic_kind = "appendix"
                stats["appendix_heading_page"] = page_no

            kind = kind_from_overrides(page_no, paper) or heuristic_kind

            # --- section name for citations, based on the EFFECTIVE kind ---
            if kind == "references":
                heading = "References"
            elif kind != previous_kind and kind == "appendix":
                heading = "Appendix"  # until the first appendix heading line is seen
            if kind != "references" and is_heading(line):
                heading = line
            previous_kind = kind

            if current is None or (current["kind"], current["heading"]) != (kind, heading):
                current = {"page": page_no, "kind": kind, "heading": heading, "lines": []}
                segments.append(current)
            current["lines"].append(line)

    for seg in segments:
        seg["text"] = clean_text("\n".join(seg.pop("lines")))
    return [s for s in segments if s["text"]], stats


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.model_max_length = 10**9  # we only count tokens here; silence the length warning
    return tok


def chunk_block(segments: list[dict], tokenizer, size: int, overlap: int) -> list[dict]:
    """Sliding-window chunking over one contiguous block of segments of the same kind."""
    parts, spans, pos = [], [], 0
    for seg in segments:
        if parts:
            parts.append(" ")
            pos += 1
        parts.append(seg["text"])
        spans.append((pos, pos + len(seg["text"]), seg))
        pos += len(seg["text"])
    text = "".join(parts)

    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    chunks, step = [], size - overlap
    for start in range(0, len(offsets), step):
        window = offsets[start:start + size]
        c0, c1 = window[0][0], window[-1][1]
        overlapping = [seg for s0, s1, seg in spans if s0 < c1 and s1 > c0]
        chunks.append({
            "text": text[c0:c1],
            "pages": sorted({seg["page"] for seg in overlapping}),
            "heading": overlapping[0]["heading"] if overlapping else "",
            "n_tokens": len(window),
        })
        if start + size >= len(offsets):
            break
    return chunks


def ingest(cfg: dict) -> None:
    raw_dir = resolve(cfg["paths"]["raw_dir"])
    out_file = resolve(cfg["paths"]["chunks_file"])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    size = cfg["chunking"]["chunk_tokens"]
    overlap = cfg["chunking"]["overlap_tokens"]
    tokenizer = load_tokenizer(cfg["embedding"]["model"])

    all_chunks = []
    print(f"{'ID':<4} {'main':>5} {'refs':>5} {'appx':>5} {'noise':>6}  refs@  appx@  source")
    for paper in load_manifest(cfg):
        pdf_path = raw_dir / f"{paper['id']}.pdf"
        if not pdf_path.exists():
            print(f"{paper['id']:<4} MISSING {pdf_path}")
            continue
        segments, stats = extract_segments(pdf_path, paper)

        # group consecutive segments of the same kind, then chunk each group
        blocks = []
        for seg in segments:
            if blocks and blocks[-1][0] == seg["kind"]:
                blocks[-1][1].append(seg)
            else:
                blocks.append((seg["kind"], [seg]))

        per_kind = defaultdict(int)
        for kind, block_segments in blocks:
            for ch in chunk_block(block_segments, tokenizer, size, overlap):
                per_kind[kind] += 1
                ch.update({
                    "chunk_id": f"{paper['id']}-{kind[0]}{per_kind[kind]:03d}",
                    "doc_id": paper["id"],
                    "title": paper["title"],
                    "kind": kind,
                })
                all_chunks.append(ch)

        source = "overrides" if kind_from_overrides(1, paper) else "heuristic"
        print(f"{paper['id']:<4} {per_kind['main']:>5} {per_kind['references']:>5} "
              f"{per_kind['appendix']:>5} {stats['dropped_noise_lines']:>6}  "
              f"{str(stats.get('references_heading_page', '-')):>5}  "
              f"{str(stats.get('appendix_heading_page', '-')):>5}  {source}")

    with out_file.open("w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    print(f"\n{len(all_chunks)} chunks -> {out_file}")


if __name__ == "__main__":
    ingest(load_config())
