# NOTES — Kalbe RAG Assessment

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
- **D8 Versi library di kunci di requirements.txt** Alasan: Agar dapat di reproduksi

## Masalah & perbaikan
- **B1 Download berhenti total di error pertama.** Gejala: P1–P2 ada, P3–P10
  MISSING. Penyebab: raise_for_status() tanpa try/except. Perbaikan: tangkap
  RequestException per paper, lanjut ke paper berikutnya.
- **B2 P3 404.** Penyebab: URL ditebak dari pola PMLR lama; volume 267 menaruh
  PDF di GitHub. Perbaikan: URL diambil dari halaman resmi PMLR P3.
- **B3 Peringatan deprecation `fitz`.** Perbaikan: `import pymupdf`.

## Observasi data
- **O1 Ukuran koleksi:** 10 paper, 284 halaman, ~915 ribu karakter.
- **O2 Tidak seimbang:** P3+P4+P5 = 148/284 halaman (52%), ~47% karakter.
  Penyebab dugaan: appendix panjang.
- **O3 Header noise:** baris pertama P2 & P4 = "Proceedings of ...";
  judul P5 huruf kapital semua.
- **O4 Halaman teks sedikit (<500 char):** P3 hal.16, P7 hal.23/27/37, P8 hal.19.
  Isinya: [...]
- **O5 Urutan kolom (P1 hal.3):** [utuh / tercampur? contoh kalimatnya]
- **O6 Batas References/Appendix:** P3 [hal...], P4 [hal...], P5 [hal...]

## Hipotesis & pertanyaan terbuka
- **H1** Chunk appendix P3/P4/P5 akan mendominasi top-k untuk pertanyaan tentang
  paper lain. Cara uji: hitung proporsi chunk appendix di top-k pada set
  evaluasi (Minggu). BELUM DIUJI.
- **Q1** Apakah References & Appendix ikut di-index? Tunggu O6.

## Log penggunaan AI
- Claude membantu: memilih opsi & desain awal, menulis draft download_papers.py
  dan inspect_extraction.py, menemukan URL P3 yang benar.
- Saya sendiri: menjalankan script, menemukan B1 dari output, menerapkan
  perbaikan B1/B3, inspeksi manual O4–O6, [...]