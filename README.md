# Projek Moondream - Pipeline Utama

Program `main.py` menjalankan pipeline lengkap saat Anda memberi perintah `capture`:

1) Ambil gambar dari kamera (webcam)
2) Segmentasi objek dekat (FastSAM)
3) Kirim hasil segmentasi ke Moondream (Ollama) untuk deskripsi Bahasa Inggris
4) Rapikan dan terjemahkan ke Bahasa Indonesia (Argos Translate, offline)
5) Bangkitkan audio TTS Bahasa Indonesia (Piper)

Output teks (.txt), audio (.wav), dan gambar hasil segmentasi disimpan di folder `Output/` dan `runs/fastsam_near/`.

## Persiapan

- Pastikan model berikut ada di folder `models/` (sudah disertakan):
  - `FastSAM-x.pt`
  - `id_ID-news_tts-medium.onnx` dan `id_ID-news_tts-medium.onnx.json`
  - `translate-en_id-1_9.argosmodel`
- Pastikan Ollama sudah berjalan dan model `moondream:latest` sudah siap.
  - Jalankan Ollama server secara lokal (default: `http://localhost:11434`).
  - Pastikan model sudah di-pull: `ollama pull moondream:latest`.

## Instalasi (opsional, disarankan pakai virtualenv)

```powershell
# Buat virtualenv (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependensi
pip install --upgrade pip
pip install -r requirements.txt

# (Opsional) Jika ingin CUDA untuk PyTorch, ikuti petunjuk di https://pytorch.org
```

## Menjalankan

```powershell
# Aktifkan virtualenv jika ada
.\.venv\Scripts\Activate.ps1

# Jalankan loop
python main.py
```

Di terminal, ketik:
- `capture` atau `c` untuk mengambil gambar dan memulai proses
- `q` untuk keluar

## Catatan

- Kamera yang digunakan: index `0`. Ubah di `capture_frame_from_camera(device_index=0)` bila perlu.
- File keluaran:
  - `runs/fastsam_near/segmented.png` dan `runs/fastsam_near/bbox_near.png`
  - `Output/<nama>.txt` dan `Output/<nama>.wav`
- Pemutaran audio otomatis menggunakan `winsound` (Windows). Jika gagal, file WAV tetap tersimpan.
