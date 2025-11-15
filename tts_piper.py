# tts_piper.py
# Modul untuk membuat audio Bahasa Indonesia menggunakan Piper TTS (tanpa piper.exe).
# Menggunakan model id_ID-news_tts-medium.onnx dan konfigurasi .onnx.json secara langsung.

import os
import io
import wave
import tempfile
from typing import Optional
from piper import SynthesisConfig, PiperVoice

# ===== KONFIGURASI =====
PIPER_MODEL_PATH  = r"models\id_ID-news_tts-medium.onnx"
PIPER_CONFIG_PATH = r"models\id_ID-news_tts-medium.onnx.json"


# ===== FUNGSI UTAMA =====
def load_tts_model():
    """
    Muat model Piper TTS (voice Bahasa Indonesia).
    Fungsi ini mengembalikan objek PiperVoice dan konfigurasi sintesis.
    """
    print("=== Memuat model Piper TTS ===")
    if not os.path.exists(PIPER_MODEL_PATH):
        raise FileNotFoundError(f"Model tidak ditemukan: {PIPER_MODEL_PATH}")
    if not os.path.exists(PIPER_CONFIG_PATH):
        raise FileNotFoundError(f"Config tidak ditemukan: {PIPER_CONFIG_PATH}")

    # Load model suara
    tts = PiperVoice.load(PIPER_MODEL_PATH, PIPER_CONFIG_PATH)

    # Konfigurasi sintesis (atur volume, panjang, dan noise)
    cfg = SynthesisConfig(
        volume=0.8,
        length_scale=1.0,
        noise_scale=0.667,
        noise_w_scale=0.8,
        normalize_audio=True,
    )
    print("✓ Model Piper berhasil dimuat.")
    return tts, cfg


def tts_piper_to_wav(
    text_id: str,
    output_wav_path: Optional[str] = None,
    sample_rate: int = 22050,
) -> Optional[str]:
    """
    text_id         : Teks bahasa Indonesia yang akan dibacakan.
    output_wav_path : Path file .wav tujuan. Jika None -> dibuat file temp.
    sample_rate     : Sample rate output (default 22050).

    Return:
        path file WAV (string) jika sukses, None jika gagal.
    """
    if not text_id.strip():
        print("[PiperTTS] Warning: teks kosong, tidak ada audio dibuat.")
        return None

    # Load TTS model
    try:
        tts, cfg = load_tts_model()
    except Exception as e:
        print(f"[PiperTTS] Gagal memuat model: {e}")
        return None

    # Siapkan path output
    if output_wav_path is None:
        fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="tts_", text=False)
        os.close(fd)
        output_wav_path = tmp_path

    # Synthesize ke bytes buffer
    try:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            tts.synthesize_wav(text_id, wav, syn_config=cfg)
        wav_bytes = buf.getvalue()

        # Simpan ke file .wav
        with open(output_wav_path, "wb") as f:
            f.write(wav_bytes)

        print(f"[PiperTTS] Audio berhasil dibuat: {output_wav_path}")
        return output_wav_path

    except Exception as e:
        print(f"[PiperTTS] Error saat membuat audio: {e}")
        return None


def speak_id(text_id: str):
    """
    Buat audio dari teks Indonesia dan coba putar (Windows). Mengembalikan path WAV atau None.
    """
    wav_path = tts_piper_to_wav(text_id)
    if wav_path and os.name == "nt":
        try:
            import winsound
            winsound.PlaySound(wav_path, winsound.SND_FILENAME)
        except Exception as e:
            print(f"[PiperTTS] Tidak bisa memutar otomatis: {e}. Path file: {wav_path}")
    return wav_path


# ===== TEST MANUAL =====
if __name__ == "__main__":
    contoh_teks = (
        "Zebra cross ada di depan. "
        "Seseorang berdiri di sisi kiri. "
        "Sebuah mobil terparkir di sisi kanan jalan."
    )

    print("=== Tes Piper TTS (Bahasa Indonesia) ===")
    out_path = tts_piper_to_wav(contoh_teks)
    if out_path:
        print("Output WAV:", out_path)
        try:
            import winsound
            winsound.PlaySound(out_path, winsound.SND_FILENAME)
        except Exception as e:
            print(f"⚠ Tidak dapat memutar suara otomatis: {e}")
