import os
import cv2
import time
from datetime import datetime
import Jetson.GPIO as GPIO

from segmentation import segment_objects
from test import (
    encode_image_base64, 
    build_prompt, 
    build_segments_info,
    query_ollama_vision, 
    clean_output_for_tts, 
    MODEL_NAME
)
from translator_argos import translate_id
from tts_piper import speak_id

# === KONFIGURASI ===
OUTPUT_DIR = "Output"
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

CAMERA_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT, FPS = 1280, 720, 30

BUTTON_PIN = 37
DEBOUNCE_SEC = 0.15

# === STATE GLOBAL ===
last_press_time = 0.0
trigger_requested = False
is_processing = False
cap = None


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


def save_outputs(base_name, en_text, id_text, latency):
    """Simpan file output teks dan latency"""
    with open(os.path.join(OUTPUT_DIR, f"{base_name}_en.txt"), "w", encoding="utf-8") as f:
        f.write(en_text)
    with open(os.path.join(OUTPUT_DIR, f"{base_name}_id.txt"), "w", encoding="utf-8") as f:
        f.write(id_text)
    
    with open(os.path.join(OUTPUT_DIR, f"{base_name}_latency.txt"), "w", encoding="utf-8") as f:
        f.write("=== LATENCY REPORT ===\n\n")
        for step, duration in latency.items():
            if step != "total_pipeline":
                percentage = (duration / latency["total_pipeline"]) * 100
                f.write(f"{step:25s}: {duration:6.3f}s ({percentage:5.1f}%)\n")
        f.write(f"\n{'TOTAL PIPELINE':25s}: {latency['total_pipeline']:6.3f}s (100.0%)\n")


def run_pipeline():
    """Pipeline lengkap: capture -> segment -> VLM -> translate -> TTS"""
    global cap
    
    print("\n================= PIPELINE DIMULAI =================")
    wall_start = time.time()
    latency = {}
    
    # Capture frame
    ok, frame = cap.read()
    if not ok:
        print("[ERR] Gagal membaca frame kamera.")
        return
    
    img_path = save_frame(frame)
    print(f"[1/7] Frame disimpan: {img_path}")
    
    # Segmentasi
    t0 = time.time()
    seg = segment_objects(img_path)
    latency["segmentation"] = time.time() - t0
    
    seg_img = seg.get("segmented_image_path") or img_path
    objects = seg.get("objects", [])
    print(f"[2/7] Segmentasi selesai: {len(objects)} objek ({latency['segmentation']:.3f}s)")
    
    # Build segments info
    t0 = time.time()
    segments_info = build_segments_info(objects) if objects else ""
    latency["build_segments"] = time.time() - t0
    
    # Encode & build prompt
    t0 = time.time()
    img_b64 = encode_image_base64(seg_img)
    latency["encode_image"] = time.time() - t0
    
    t0 = time.time()
    prompt = build_prompt(segments_info)
    latency["build_prompt"] = time.time() - t0
    print(f"[3/7] Encoding & prompt selesai ({latency['encode_image']:.3f}s)")
    
    # Moondream inference
    t0 = time.time()
    en_raw = query_ollama_vision(MODEL_NAME, prompt, img_b64)
    latency["moondream_inference"] = time.time() - t0
    print(f"[4/7] Moondream: {en_raw[:50]}... ({latency['moondream_inference']:.3f}s)")
    
    # Clean output
    t0 = time.time()
    en_tts = clean_output_for_tts(en_raw)
    latency["clean_output"] = time.time() - t0
    
    # Translate
    t0 = time.time()
    id_tts = translate_id(en_tts)
    latency["translation"] = time.time() - t0
    print(f"[5/7] Terjemahan: {id_tts[:50]}... ({latency['translation']:.3f}s)")
    
    # TTS
    t0 = time.time()
    wav_path = speak_id(id_tts)
    latency["tts"] = time.time() - t0
    print(f"[6/7] TTS selesai: {wav_path} ({latency['tts']:.3f}s)")
    
    # Total & save
    latency["total_pipeline"] = sum(latency.values())
    wall_time = time.time() - wall_start
    
    base = os.path.splitext(os.path.basename(img_path))[0]
    save_outputs(base, en_tts, id_tts, latency)
    
    print(f"[7/7] Pipeline selesai: {latency['total_pipeline']:.3f}s (wall: {wall_time:.3f}s)")
    print("================= PIPELINE SELESAI =================\n")


def on_button_pressed():
    global trigger_requested, is_processing
    if is_processing:
        print("[INFO] Pipeline masih berjalan. Abaikan.")
        return
    if not trigger_requested:
        trigger_requested = True
        print("[EVENT] Tombol ditekan! Pipeline akan dijalankan...")


def button_callback(channel):
    global last_press_time
    now = time.time()
    if now - last_press_time < DEBOUNCE_SEC:
        return
    last_press_time = now
    on_button_pressed()


def main():
    global trigger_requested, is_processing, cap
    
    # Setup GPIO
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING, callback=button_callback, bouncetime=1)
    
    # Buka kamera
    cap = open_camera()
    print("=== Vision Assist — Button Trigger Mode ===")
    print(f"Tombol pada pin fisik {BUTTON_PIN}. Tekan untuk proses.")
    print("Tekan Ctrl+C untuk keluar.\n")
    
    try:
        while True:
            if trigger_requested and not is_processing:
                trigger_requested = False
                is_processing = True
                try:
                    run_pipeline()
                except Exception as e:
                    print(f"[ERR] Pipeline gagal: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    is_processing = False
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n[MAIN] Dihentikan. Keluar...")
    finally:
        if cap:
            cap.release()
        GPIO.cleanup()


if __name__ == "__main__":
    main()
