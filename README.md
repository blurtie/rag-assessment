# RAG over research papers: from-scratch baseline

This repository is the **first version** of my RAG system for the Kalbe AI Engineer (Multimodal)
technical assessment, built **without a framework** to learn the fundamentals: PDF extraction,
page labelling, chunking, bge-m3 embeddings, dense / BM25 / hybrid (RRF) retrieval, generation
with qwen3:8b via Ollama, citation mapping, and evaluation.

**The submitted version is the LangChain rewrite:** <https://github.com/blurtie/kalbe-rag-lc>
(same papers, same evaluation questions, same LLM and embedding model). Its README compares the two.

- Design decisions, bugs found and fixed, observations and limitations: [`NOTES.md`](NOTES.md)
- Code: `rag/` (`config`, `ingest`, `index`, `retrieve`, `generate`, `cli`, `evaluate`)
- Labelled evaluation results: `outputs/eval/`

**Known issue:** P7's manual page boundaries in `papers.yaml` (references from p. 7, appendix from p. 8)
were taken from the Nature version of DreamerV3, while the indexed file is the 40-page arXiv preprint
(correct: references pp. 12–16, Methods + supplementary from p. 17). It was fixed only in the LangChain
version; re-running that version with the fix did not change any retrieval metric.

**AI assistance:** written with help from Claude (Anthropic); see the LangChain repo's README for details.
