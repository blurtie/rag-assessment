"""Retrieval: dense (embeddings), BM25 (keywords), or hybrid (Reciprocal Rank Fusion).

Hybrid / RRF: each retriever returns a ranked candidate list; a chunk's fused score is
sum over retrievers of 1 / (rrf_k + rank). Only ranks are used, so the two very different
score scales (cosine vs BM25) never need to be compared directly.
"""
import json
import re

import numpy as np
from rank_bm25 import BM25Okapi

from rag.config import resolve
from rag.index import embed_texts, load_chunks, load_embedder

TOKEN_RE = re.compile(r"\w+")


def bm25_tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class Retriever:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        meta = json.loads(resolve(cfg["paths"]["index_meta_file"]).read_text(encoding="utf-8"))
        by_id = {c["chunk_id"]: c for c in load_chunks(cfg, only_indexed=False)}
        # keep exactly the order used when the embeddings were built
        self.chunks = [by_id[cid] for cid in meta["chunk_ids"]]
        self.vectors = np.load(resolve(cfg["paths"]["embeddings_file"]))
        assert len(self.chunks) == len(self.vectors), "index out of date: rerun `index`"
        self.bm25 = BM25Okapi([bm25_tokenize(c["text"]) for c in self.chunks])
        self._embedder = None  # loaded on first dense query

    def _dense_ranking(self, query: str, k: int) -> list[tuple[int, float]]:
        if self._embedder is None:
            self._embedder = load_embedder(self.cfg)
        q = embed_texts(self._embedder, [query], batch_size=1)[0]
        scores = self.vectors @ q
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]

    def _bm25_ranking(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(bm25_tokenize(query))
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]

    def search(self, query: str, method: str | None = None, top_k: int | None = None) -> list[dict]:
        r = self.cfg["retrieval"]
        method = method or r["method"]
        top_k = top_k or r["top_k"]
        cand_k = max(r["candidate_k"], top_k)

        rankings = {}
        if method in ("dense", "hybrid"):
            rankings["dense"] = self._dense_ranking(query, cand_k)
        if method in ("bm25", "hybrid"):
            rankings["bm25"] = self._bm25_ranking(query, cand_k)

        fused: dict[int, dict] = {}
        for name, ranked in rankings.items():
            for rank, (idx, score) in enumerate(ranked, start=1):
                entry = fused.setdefault(idx, {"score": 0.0})
                entry[f"{name}_rank"] = rank
                entry[f"{name}_score"] = round(score, 4)
                entry["score"] += 1.0 / (r["rrf_k"] + rank) if method == "hybrid" else score

        ordered = sorted(fused.items(), key=lambda kv: -kv[1]["score"])[:top_k]
        return [{"rank": i, "chunk": self.chunks[idx], **info}
                for i, (idx, info) in enumerate(ordered, start=1)]
