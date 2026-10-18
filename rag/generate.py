"""Answer generation with a local LLM through Ollama's /api/chat endpoint.

The LLM sees numbered passages [C1]..[Cn] and must cite them. After generation we map
each [Ck] back to the chunk that was actually retrieved, so every reference shown to
the user corresponds to retrieved evidence. Citations that point to a passage number
that was never provided are reported as invalid.
"""
import re
import time

import requests

SYSTEM_PROMPT = """You answer questions about a fixed collection of research papers.
Rules:
1. Use ONLY the information in the numbered context passages. Do not add facts from outside knowledge.
2. Cite the passage(s) supporting each claim right after it, e.g. [C1] or [C2][C4].
3. If the passages do not contain enough information to answer, say so plainly instead of guessing.
4. Answer in the same language as the question. Be concise."""

CLOSED_BOOK_PROMPT = """Answer the question about research papers from your own knowledge.
If you do not know, say so. Answer in the same language as the question. Be concise."""

CITATION_RE = re.compile(r"\[C(\d+)\]")
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def format_pages(pages: list[int]) -> str:
    return f"p.{pages[0]}" if len(pages) == 1 else f"pp.{pages[0]}-{pages[-1]}"


def build_context(hits: list[dict]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        c = hit["chunk"]
        header = (f"[C{i}] {c['doc_id']}: {c['title']} | {format_pages(c['pages'])}"
                  f" | section: {c['heading'] or '-'} | {c['kind']}")
        blocks.append(f"{header}\n{c['text']}")
    return "\n\n".join(blocks)


def call_ollama(cfg: dict, system: str, user: str) -> dict:
    g = cfg["generation"]
    payload = {
        "model": g["model"],
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": g["temperature"], "num_ctx": g["num_ctx"]},
    }
    if g.get("think") is not None:
        payload["think"] = g["think"]
    t0 = time.time()
    resp = requests.post(f"{g['host']}/api/chat", json=payload, timeout=g["timeout_s"])
    resp.raise_for_status()
    data = resp.json()
    content = THINK_RE.sub("", data["message"]["content"]).strip()
    return {
        "content": content,
        "seconds": round(time.time() - t0, 1),
        "prompt_tokens": data.get("prompt_eval_count"),
        "output_tokens": data.get("eval_count"),
    }


def answer(cfg: dict, question: str, hits: list[dict]) -> dict:
    user = f"Context passages:\n\n{build_context(hits)}\n\nQuestion: {question}"
    result = call_ollama(cfg, SYSTEM_PROMPT, user)

    cited = sorted({int(n) for n in CITATION_RE.findall(result["content"])})
    valid = [n for n in cited if 1 <= n <= len(hits)]
    result.update({
        "user_prompt": user,
        "cited_passages": valid,
        "invalid_citations": [n for n in cited if n not in valid],
        "sources": [{
            "label": f"C{n}",
            "chunk_id": hits[n - 1]["chunk"]["chunk_id"],
            "doc_id": hits[n - 1]["chunk"]["doc_id"],
            "title": hits[n - 1]["chunk"]["title"],
            "pages": hits[n - 1]["chunk"]["pages"],
            "section": hits[n - 1]["chunk"]["heading"],
        } for n in valid],
    })
    return result


def answer_closed_book(cfg: dict, question: str) -> dict:
    """Same LLM, no retrieved context: measures what the model already 'knows' (NOTES H2)."""
    return call_ollama(cfg, CLOSED_BOOK_PROMPT, f"Question: {question}")
