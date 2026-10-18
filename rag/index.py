"""Embed the selected chunks and save them as a plain NumPy matrix.

Why no vector database: the collection is a few hundred chunks. Exact search
(one matrix multiplication) is fast enough, has no approximation error, and every
step is visible. A vector DB (Chroma, FAISS) would matter at ~100k+ chunks.
"""
import json
import time

import numpy as np

from rag.config import load_config, resolve


def load_chunks(cfg: dict, only_indexed: bool = True) -> list[dict]:
    path = resolve(cfg["paths"]["chunks_file"])
    chunks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if only_indexed:
        keep = set(cfg["chunking"]["index_sections"])
        chunks = [c for c in chunks if c["kind"] in keep]
    return chunks


def load_embedder(cfg: dict):
    import torch
    from sentence_transformers import SentenceTransformer

    device = cfg["embedding"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, embedding on CPU (slower).")
        device = "cpu"
    model = SentenceTransformer(cfg["embedding"]["model"], device=device)
    if device == "cuda":
        model.half()  # fp16: about half the VRAM, negligible quality change for retrieval
    return model


def embed_texts(model, texts: list[str], batch_size: int) -> np.ndarray:
    vecs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=len(texts) > 1)
    return vecs.astype(np.float32)  # unit-length vectors: dot product == cosine similarity


def build_index(cfg: dict) -> None:
    chunks = load_chunks(cfg)
    model = load_embedder(cfg)
    t0 = time.time()
    vectors = embed_texts(model, [c["text"] for c in chunks], cfg["embedding"]["batch_size"])
    elapsed = time.time() - t0

    np.save(resolve(cfg["paths"]["embeddings_file"]), vectors)
    meta = {
        "embedding_model": cfg["embedding"]["model"],
        "index_sections": cfg["chunking"]["index_sections"],
        "chunk_tokens": cfg["chunking"]["chunk_tokens"],
        "overlap_tokens": cfg["chunking"]["overlap_tokens"],
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "dim": int(vectors.shape[1]),
        "embed_seconds": round(elapsed, 1),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    resolve(cfg["paths"]["index_meta_file"]).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Indexed {len(chunks)} chunks, dim={vectors.shape[1]}, {elapsed:.1f}s")


if __name__ == "__main__":
    build_index(load_config())
