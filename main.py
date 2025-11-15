import os
import cv2
import time
from datetime import datetime

# === Import ONLY the functions from your existing modules ===
from segmentation import segment_objects  # returns dict with 'segmented_image_path', 'objects', etc.
from test import (
    encode_image_base64, 
    build_prompt, 
    build_segments_info,  # TAMBAHKAN ini untuk format objects info
    query_ollama_vision, 
    clean_output_for_tts, 
    MODEL_NAME
)
from translator_argos import translate_id  # EN -> ID (polished for TTS)
from tts_piper import speak_id  # ID text -> WAV

OUTPUT_DIR = "Output"
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

# Camera settings (USB default index 0). If you use CSI, switch to GStreamer pipeline.
CAMERA_INDEX = 0
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
FPS          = 30


def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    if not cap.isOpened():
        raise RuntimeError("Tidak bisa membuka kamera. Cek USB/driver.")
    return cap


def save_frame(frame):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(FRAMES_DIR, f"frame_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path


def process_image(image_path: str):
    """Run the full pipeline USING YOUR EXISTING MODULES ONLY.
    Steps: Segment -> pick segmented image -> Moondream -> clean EN -> Translate ID -> TTS.
    Returns a dict of outputs.
    """
    # 1) Segmentasi (pakai modul kamu)
    seg = segment_objects(image_path)
    seg_img = seg.get("segmented_image_path") or image_path  # fallback ke gambar asli kalau tidak ada segmented
    objects = seg.get("objects", [])  # Ambil objects dari segmentasi

    # 2) Build segments info untuk prompt (PERBAIKAN BARU)
    segments_info = ""
    if objects:
        segments_info = build_segments_info(objects)  # Format objects untuk prompt
        print(f"[INFO] {len(objects)} objects detected from segmentation")

    # 3) Moondream (pakai modul kamu dengan segments_info)
    img_b64 = encode_image_base64(seg_img)
    prompt  = build_prompt(segments_info)  # PASS segments_info ke build_prompt
    en_raw  = query_ollama_vision(MODEL_NAME, prompt, img_b64)

    # 4) Rapikan EN untuk TTS (pakai modul kamu)
    en_tts = clean_output_for_tts(en_raw)

    # 5) Translate ke Indonesia (pakai modul kamu)
    id_tts = translate_id(en_tts)

    # 6) Suarakan (pakai modul kamu)
    wav_path = speak_id(id_tts)

    # 7) Simpan teks
    base = os.path.splitext(os.path.basename(image_path))[0]
    with open(os.path.join(OUTPUT_DIR, f"{base}_en.txt"), "w", encoding="utf-8") as f:
        f.write(en_tts)
    with open(os.path.join(OUTPUT_DIR, f"{base}_id.txt"), "w", encoding="utf-8") as f:
        f.write(id_tts)

    return {
        "input_image": image_path,
        "segmented_image": seg_img,
        "objects": objects,
        "segments_info": segments_info,  # Tambahkan untuk debugging
        "raw_en": en_raw,
        "clean_en": en_tts,
        "final_id": id_tts,
        "audio_wav": wav_path,
    }


def main():
    print("=== Vision Assist — Manual Trigger Mode ===")
    print("Preview window: press SPACE to process current frame, Q to quit.")
    cap = open_camera()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERR] Gagal membaca frame kamera.")
                time.sleep(0.1)
                continue

            # show preview
            preview = cv2.resize(frame, (960, 540))
            cv2.imshow("Preview — SPACE: process, Q: quit", preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("[INFO] Keluar.")
                break

            if key == ord(' '):  # SPACE triggers one full pipeline run
                img_path = save_frame(frame)
                print(f"[RUN] Proses frame: {img_path}")
                try:
                    res = process_image(img_path)
                    print("\n=== HASIL ===")
                    if res["objects"]:
                        print(f"[Objects]: {len(res['objects'])} detected")
                        print("[Segments]:", res["segments_info"])  # Show formatted info
                    print("[EN RAW ]:", res["raw_en"]) 
                    print("[EN TTS ]:", res["clean_en"]) 
                    print("[ID TTS ]:", res["final_id"]) 
                    print("[Audio ]:", res["audio_wav"]) 
                    print("[SegImg]:", res["segmented_image"]) 
                    print("============\n")
                except Exception as e:
                    print("[ERR] Pipeline gagal:", e)
                    import traceback
                    traceback.print_exc()

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()