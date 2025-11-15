# translator_argos.py
# Pipeline Moondream EN -> Bahasa Indonesia lisan -> siap dibacakan Piper TTS

import os
_argos_ready = False

ARGOS_MODEL_PATH = r"models\translate-en_id-1_9.argosmodel"


def _ensure_argos_loaded():
    """
    Pastikan model Argos en->id ter-install sekali di runtime.
    """
    global _argos_ready
    if _argos_ready:
        return
    try:
        import argostranslate.package
        if os.path.exists(ARGOS_MODEL_PATH):
            argostranslate.package.install_from_path(ARGOS_MODEL_PATH)
        else:
            print(f"[ArgosTranslate] Warning: model file not found at {ARGOS_MODEL_PATH}")
        _argos_ready = True
    except Exception as e:
        print(f"[ArgosTranslate] Warning: failed to load Argos model: {e}")
        _argos_ready = False


def normalize_en_for_translate(text_en: str) -> str:
    """
    Sederhanakan Inggris supaya:
    - pendek
    - langsung sebut lokasi (left/right/ahead)
    - hindari frasa panjang seperti 'in the foreground'
    """
    t = text_en.strip()

    replacements = [
        ("in the foreground", "in front of you"),
        ("in the foreground,", "in front of you,"),
        ("in the foreground.", "in front of you."),
        ("is located ahead of the intersection", "is ahead at the intersection"),
        ("is located ahead", "is ahead"),
        ("is parked at the curb", "is parked on the right side"),
        ("indicating that it's safe for pedestrians to cross",
         "the light says it is safe to cross"),
    ]

    for src, dst in replacements:
        t = t.replace(src, dst)

    return t


def _argos_translate_en_id(text_en_simple: str) -> str:
    """
    Terjemahkan Inggris -> Indonesia via Argos.
    Kalau Argos error, fallback ke teks Inggris biar gak crash.
    """
    _ensure_argos_loaded()
    try:
        from argostranslate import translate as argos_translate
        text_id = argos_translate.translate(text_en_simple, "en", "id")
        return text_id.strip()
    except Exception as e:
        print(f"[ArgosTranslate] Warning: translation failed: {e}")
        return text_en_simple.strip()


def _polish_indonesian_for_tts(text_id: str) -> str:
    """
    Rapikan hasil Argos supaya lebih natural dan tidak berulang.
    Fokus agar kalimat ringkas, mudah diucapkan, dan sesuai konteks navigasi.
    """
    t = text_id.strip().lower()

    replacements = [
        # Pola umum
        ("jalan kota dengan pejalan kaki menyeberang di depan anda", "di depan ada area penyeberangan dengan garis putih"),
        ("jalan kota dengan pejalan kaki menyeberang di depan kamu", "di depan ada area penyeberangan dengan garis putih"),
        ("jalan kota dengan penyeberangan pejalan kaki di depan anda", "di depan ada area penyeberangan dengan garis putih"),
        ("jalan kota dengan penyeberangan pejalan kaki di depan kamu", "di depan ada area penyeberangan dengan garis putih"),

        # Hilangkan pengulangan frasa
        ("menampilkan garis putih dan tanda penyeberangan", "serta tanda penyeberangan"),
        ("menampilkan garis putih", "dengan garis putih"),
        ("garis putih, menampilkan garis putih", "garis putih"),
        ("mobil diparkir di sisi kanan di sisi kanan", "mobil diparkir di sisi kanan"),
        ("lampu merah di depan persimpangan, lampu merah menunjukkan aman untuk menyeberang",
         "lampu merah di depan persimpangan menunjukkan aman untuk menyeberang"),

        # Umumkan bentuk kalimat agar natural
        ("terletak di", "ada di"),
        ("sementara di sisi kiri", "di sisi kiri"),
        ("sementara di sisi kanan", "di sisi kanan"),
        ("sementara di sebelah kiri", "di sisi kiri"),
        ("sementara di sebelah kanan", "di sisi kanan"),
        ("lampu lalu lintas", "lampu merah"),
        ("lampu mengatakan aman", "lampu merah menunjukkan aman"),
        ("lampu menyatakan aman", "lampu merah menunjukkan aman"),
        ("menunjukkan bahwa aman", "menunjukkan aman"),
        ("tanda \"wideway\"", "tanda penyeberangan"),
        ("tanda “wideway”", "tanda penyeberangan"),
        ("wideway", "penyeberangan"),
    ]

    for src, dst in replacements:
        if src in t:
            t = t.replace(src, dst)

    # Pisahkan kalimat jadi rapi: koma -> titik di tempat tertentu
    t = t.replace(", di sisi", ". Di sisi")
    t = t.replace(", lampu merah", ". Lampu merah")

    # Rapikan spasi
    while "  " in t:
        t = t.replace("  ", " ")

    # Kapitalisasi awal dan tanda titik
    t = t.strip()
    if t:
        t = t[0].upper() + t[1:]
        if not t.endswith("."):
            t += "."

    return t



def describe_scene_for_tts(text_en: str) -> str:
    """
    Fungsi utama yang dipakai pipeline kamu:
    1. bersihin English
    2. translate offline Argos
    3. poles biar natural untuk dibacakan
    Output: Bahasa Indonesia final (pendek, jelas, navigasi)
    """
    en_simple = normalize_en_for_translate(text_en)
    id_raw = _argos_translate_en_id(en_simple)
    id_clean = _polish_indonesian_for_tts(id_raw)
    return id_clean


# Alias sederhana sesuai API yang diinginkan main
def translate_id(text_en: str) -> str:
    """Terjemahkan EN -> ID yang siap TTS (alias describe_scene_for_tts)."""
    return describe_scene_for_tts(text_en)


if __name__ == "__main__":
    # Contoh kalimat dari Moondream (bahasa Inggris mentah)
    test_text = (
        "A city street with a pedestrian crossing in the foreground, "
        "featuring white stripes and a 'Wideway' sign. "
        "On the left side, there is a person standing on the sidewalk, "
        "while on the right side, a car is parked at the curb. "
        "The traffic lights are located ahead of the intersection, "
        "indicating that it's safe for pedestrians to cross."
    )

    final_id = describe_scene_for_tts(test_text)
    print("ID (final TTS-ready):", final_id)
