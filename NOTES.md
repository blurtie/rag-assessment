# NOTES — Kalbe RAG Assessment

Bagian yang masih `[TODO]` belum saya periksa sendiri. Jangan diisi sebelum dicek.

## Keputusan
- **D1 (15 Okt) Opsi 1: RAG.** Alasan: kekuatan saya membangun sistem; diskusi
  membahas kode saya sendiri. Ditolak: Opsi 3 (butuh pengalaman riset), Opsi 2
  (diskusi mekanisme lebih sulit diprediksi).
- **D2 Koleksi: 10 paper reading list (P1–P10).** Alasan: relevan dengan posisi,
  reviewer bisa menilai benar/salah jawaban, ada pertanyaan lintas dokumen.
  Ditolak: dokumen BPOM (kurang relevan dengan posisi multimodal).
- **D3 PDF tidak di-commit, pakai download script.** Alasan: lisensi tiap paper
  berbeda; reviewer bisa mereproduksi koleksi.
- **D4 P7 pakai preprint arXiv, bukan versi Nature.** Alasan: versi Nature
  mungkin tidak bebas diunduh. Konsekuensi: isi/halaman bisa beda dari versi terbit.
- **D5 Tanpa LangChain/LlamaIndex.** Alasan: harus bisa trace satu pertanyaan
  melewati pipeline dan menjelaskan setiap langkah.
- **D6 Conda hanya untuk environment, dependency via pip + requirements.txt.**
  Alasan: reviewer tanpa Anaconda tetap bisa setup; menghindari konflik conda/pip.
- **D7 Judul dokumen diambil dari papers.yaml, bukan teks PDF.** Alasan: lihat O3.
- **D8 Versi library dikunci di requirements.txt.** Alasan: hasil di laptop
  reviewer sama dengan punya saya. Pemicu: peringatan deprecation `fitz` (B3)
  menunjukkan perilaku library bisa berubah antar versi.
- **D9 LLM: `qwen3:8b` via Ollama, ID `500a1f067a9f`.** Alasan: gratis, lokal,
  muat 100% di GPU 8 GB (O10). `think: false` untuk mematikan output "thinking".
  Catatan jujur: belum dibandingkan langsung dengan kandidat lain (`qwen3.5:4b`).
- **D10 Embedding disimpan sebagai matriks NumPy, tanpa vector database (Chroma).**
  Alasan: 627 chunk; pencarian eksak = satu perkalian matriks, cepat, tanpa error
  aproksimasi, dan setiap langkah terlihat. Vector DB baru relevan di skala jauh lebih besar.
- **D11 torch build CUDA 12.6 dari PyTorch index; torchvision dihapus.** Alasan:
  `+cu126` tidak ada di PyPI biasa (pakai `--extra-index-url`); torchvision tidak dipakai.
- **D12 Batas References/Appendix diisi manual di papers.yaml (override).**
  Alasan: heuristik gagal di 8 dari 10 paper (B5). Diisi dari `find_appendix.py`
  + membuka PDF. Nilai: lihat O6.
- **D13 References tidak di-index; main + appendix di-index** (`index_sections`).
  Alasan: daftar pustaka cocok secara kata kunci dengan banyak pertanyaan tapi
  tidak pernah berisi jawaban. Menjawab Q1.
- **D14 Chunk 400 token, overlap 80 (tokenizer bge-m3).** Baseline, BELUM di-tune.
  Overlap supaya kalimat yang terpotong di batas tetap utuh di salah satu chunk.
- **D15 Embedding: `BAAI/bge-m3`, fp16 di GPU.** Alasan: multilingual (pertanyaan
  bahasa Indonesia ke paper bahasa Inggris), fp16 menghemat VRAM.
- **D16 Sitasi lewat label [C1]..[Cn].** LLM hanya melihat passage bernomor;
  setiap [Cn] di jawaban dipetakan balik ke chunk yang benar-benar diambil,
  sehingga sitasi pasti sesuai bukti. Nomor yang tidak pernah diberikan ditandai invalid.
- **D17 Kirim Senin 19 Okt, bukan Kamis.** Alasan: Selasa–Rabu padat. Scope dipotong:
  set evaluasi 10–12 soal, tanpa slide/rekaman demo.

## Masalah & perbaikan
- **B1 Download berhenti total di error pertama.** Gejala: P1–P2 ada, P3–P10
  MISSING. Penyebab: raise_for_status() tanpa try/except. Perbaikan: tangkap
  RequestException per paper, lanjut ke paper berikutnya.
- **B2 P3 404.** Penyebab: URL ditebak dari pola PMLR lama; volume 267 menaruh
  PDF di GitHub. Perbaikan: URL diambil dari halaman resmi PMLR P3.
- **B3 Peringatan deprecation `fitz`.** Perbaikan: `import pymupdf`.
- **B4 Deteksi header/footer ikut menghapus baris isi.** Ditemukan saat testing
  dengan PDF tiruan: baris isi yang mirip antar halaman (beda angka saja) dianggap
  header. Perbaikan: hanya cek 3 baris teratas/terbawah tiap halaman.
- **B5 Appendix salah dilabeli references.** Gejala (dari tabel `ingest`):
  refs = 330 dari 781 chunk (42%); P4 = 121 chunk refs, 0 appendix. Penyebab:
  heuristik hanya mengenali judul "Appendix", padahal kebanyakan paper memakai
  "A Evaluation Prompts" / "A. Experiment Details". Dampak: ~250 chunk appendix
  akan terbuang dari index tanpa error. Perbaikan: override per paper (D12).
  Hasil: refs 330 → 158, appendix 79 → 332, yang di-index ~451 → 627.
- **B6 Chunk appendix berlabel section "References".** Penyebab: pelacak judul
  section membeku setelah "References" dan hanya bangun lagi kalau melihat kata
  "Appendix"; override tidak dipakai di situ. Perbaikan: pelacak mengikuti label
  efektif (override) dan mengenali judul berhuruf ("A.", "B.1").
- **B7 Error 404 dari Ollama tidak informatif.** "404 Not Found for url /api/chat"
  terdengar seperti alamat salah, padahal model belum di-pull. Perbaikan: tampilkan
  pesan asli Ollama + perintah `ollama pull`; pesan terpisah kalau Ollama tidak jalan.
- **B8 (environment, bukan kode) Folder model Ollama rusak.** `pull` sukses tapi
  `list` kosong; log server: "bad manifest ... cannot find the path specified" untuk
  qwen3 dan llama3 lama. Solusi: [TODO: tulis yang benar-benar saya lakukan,
  mis. OLLAMA_MODELS → D:\ollama-models lalu pull ulang].

## Observasi data
- **O1 Ukuran koleksi:** 10 paper, 284 halaman, ~915 ribu karakter.
- **O2 Tidak seimbang:** P3+P4+P5 = 148/284 halaman (52%), ~47% karakter.
  Penyebab: appendix panjang (terkonfirmasi di O12).
- **O3 Header noise:** baris pertama P2 & P4 = "Proceedings of ...";
  judul P5 huruf kapital semua.
- **O4 Halaman teks sedikit (<500 char):** P3 hal.16, P7 hal.23/27/37, P8 hal.19.
  Isinya: [TODO]
- **O5 Urutan kolom (P1 hal.3):** [TODO: utuh / tercampur? contoh kalimatnya]
- **O6 Batas References/Appendix (override di papers.yaml):**
  | Paper | references | appendix | catatan |
  |---|---|---|---|
  | P1 | 9 | 12 | |
  | P2 | 9 | 12 | |
  | P3 | 10 | 15 | |
  | P4 | 10 | 14 | |
  | P5 | 43 | 9 | appendix SEBELUM references |
  | P6 | 9 | 14 | |
  | P7 | 7 | 8 | format Nature, lihat L1 |
  | P8 | 13 | 16 | [TODO: cek baris "D.1." di hal. 11] |
  | P9 | 7 | – | hal. 3–6 "A./B." = subjudul IEEE, bukan appendix |
  | P10 | 9 | – | |
- **O7 Satu pertanyaan ("What is the MMVP benchmark?"), dense vs BM25:**
  dense menaruh chunk definisi (P1-m004) di #2; BM25 tidak memasukkannya ke top 5
  dan malah menaruh teks contoh soal dari gambar di #1–#2 (kata "MMVP benchmark"
  berulang di keterangan gambar). Hybrid menurunkan chunk definisi ke #5.
  BARU SATU PERTANYAAN, bukan kesimpulan.
- **O8 Faithfulness jawaban MMVP:** jawaban hanya mengutip [C2], tapi menyebut
  "embedding DINOv2 berbeda". [TODO: apakah "DINOv2" ada di teks C2? Cek di JSON
  outputs/. Kalau tidak: citation error atau parametric leakage (H2).]
- **O9 Latensi:** panggilan pertama 89,7 s, panggilan kedua 5,6 s. Selisihnya =
  waktu memuat model 5,2 GB ke VRAM; Ollama mengeluarkan model setelah beberapa
  menit idle.
- **O10 VRAM:** `ollama ps` = 6,2 GB, 100% GPU, context 8192 (RTX 4060 8 GB),
  bersamaan dengan bge-m3 di proses Python.
- **O11 Output tidak sepenuhnya deterministik** walau temperature 0: dua run
  pertanyaan sama memberi kalimat sedikit berbeda. Implikasi: perbedaan kecil
  antar-run belum tentu berarti sistem berubah.
- **O12 Komposisi index:** 627 chunk = 295 main + 332 appendix. Appendix = 53%
  index; P4 sendiri = 134 chunk (~21%).

## Batasan yang diketahui
- **L1 Override hanya 1 blok references + 1 blok appendix.** P7 (format Nature):
  main → refs → Methods → refs → supplementary. Akibat: Methods (hal. 8–11)
  berlabel appendix; daftar pustaka Methods di hal. 12 kemungkinan ikut ter-index
  (~1 chunk). Perbaikan mungkin: daftar rentang halaman per section.
- **L2 Satu chunk bisa melintasi dua section,** label hanya menyebut yang pertama
  (contoh: P1-a002 berisi akhir section A + awal "B. MMVP Benchmark").
- **L3 Halaman gambar berlabel judul section terakhir** (P1-a010 = Figure 10,
  berlabel "E.2 Scaling up...").
- **L4 Label override per halaman:** sisa daftar pustaka di halaman pertama
  appendix ikut berlabel appendix.
- **L5 Tanpa override, heuristik tidak mengenali appendix bergaya "A."** —
  dokumen baru butuh pengecekan manual.
- **L6 Penyambungan kata ber-tanda-hubung** juga menyambung kata majemuk yang
  kebetulan terpotong di akhir baris ("vision-\nlanguage" → "visionlanguage").
- **L7 Cold start ~90 s** untuk pertanyaan pertama setelah model idle (O9).

## Hipotesis & pertanyaan terbuka
- **H1** Chunk appendix (terutama P4: ~21% index, isinya banyak contoh soal)
  akan muncul di top-k untuk pertanyaan tentang paper lain. Dukungan awal: O7
  (noise dari gambar contoh soal naik di BM25). Cara uji: proporsi chunk appendix
  di top-k pada set evaluasi. BELUM DIUJI.
- **H2** LLM bisa menjawab benar dari pengetahuan pretraining, bukan dari chunk.
  Cara uji: closed-book baseline (`ask --closed-book`) pada set evaluasi.
  Kandidat contoh: O8. BELUM DIUJI.
- **Q1** ~~Apakah References & Appendix ikut di-index?~~ Dijawab di D13.

## Log penggunaan AI
- **Claude membantu:** memilih opsi & desain awal; menulis draft
  download_papers.py, inspect_extraction.py, find_appendix.py, dan seluruh paket
  rag/ (ingest, index, retrieve, generate, cli, config); menguji kode dengan PDF
  tiruan; menemukan URL P3 yang benar; mendiagnosis B5–B8 dari output yang saya
  kirim dan menulis perbaikan B4, B6, B7; membantu merapikan NOTES ini.
- **Saya sendiri:** menjalankan semua script di laptop saya; menemukan B1 dari
  output; menerapkan perbaikan B1/B3; memeriksa PDF secara manual untuk batas
  references/appendix P2, P6, P7, P8, P9 dan struktur P7; mengunci versi library;
  memperbaiki environment Ollama (B8); [TODO: O4, O5, O8, pertanyaan evaluasi].
