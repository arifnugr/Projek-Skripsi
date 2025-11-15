import os
import json
import base64
import requests
from PIL import Image

# --- CONFIG ---
pic_path = r"runs\fastsam_near\segmented.png"
json_path = r"runs\fastsam_near\objects_info.json"
MODEL_NAME = "moondream:latest"
OLLAMA_URL = "http://localhost:11434/api/generate"
output_dir = "Output"


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def load_objects_info(json_path: str) -> dict:
    """Load FastSAM objects info dari JSON"""
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found!")
        return {"objects": []}
    
    with open(json_path, "r") as f:
        return json.load(f)

def build_segments_info(objects: list) -> str:
    """Format info objek dari FastSAM untuk prompt"""
    if not objects:
        return ""
    
    info_lines = []
    for obj in objects:
        obj_id = obj["id"]
        h_pos = obj["h_position"]  # left/center/right
        v_pos = obj["v_position"]  # far/medium/near
        area = obj["area"]
        
        # Format readable position
        position = f"{v_pos} {h_pos}"  # e.g., "far left", "medium center"
        
        info_lines.append(f"Object {obj_id}: {position} (area: {area} pixels)")
    
    return "\n".join(info_lines)


def build_prompt(segments_info: str = "") -> str:
    """Build prompt dengan info posisi dari FastSAM"""
    if segments_info:
        # Prompt dengan info posisi
        num_objects = segments_info.count("Object")
        prompt = f"""Analyze this segmentation image with these detected objects:

{segments_info}

Your task: Identify and describe ALL {num_objects} objects above. 
What are these objects? Give brief navigation advice."""
        
    return prompt


def query_ollama_vision(model_name: str, prompt_text: str, image_b64: str) -> str:
    payload = {
        "model": model_name,
        "prompt": prompt_text,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": 0.4,  # Sedikit lebih tinggi untuk variasi
            "num_predict": 150,  # NAIKKAN ke 150 tokens
            "top_k": 20,
            "top_p": 0.92,
            "stop": ["Image", "In the image", "\n\n\n"]  # Stop sequences
        }
    }
    
    print(f"\nQuerying {model_name}...")
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    
    answer_text = data.get("response", "")
    return answer_text.strip()


def clean_output_for_tts(answer: str) -> str:
    """Cleaning dan tambahkan navigasi"""
    txt = answer.replace("\n", " ").strip()
    
    # Remove leading spaces
    txt = txt.lstrip()
    
    # === HAPUS "Object 1:", "Object 2:", dll ===
    import re
    txt = re.sub(r'^Object\s+\d+:\s*', '', txt, flags=re.IGNORECASE)
    txt = re.sub(r'\.\s+Object\s+\d+:\s*', '. ', txt, flags=re.IGNORECASE)
    
    # Fix huruf terpotong di awal
    if txt.startswith("xtremely"):
        txt = "E" + txt
    
    # Capitalize first letter
    if txt:
        txt = txt[0].upper() + txt[1:]
    
    # Cleanup multiple spaces
    txt = " ".join(txt.split())
    
    # Potong jika kalimat terpotong (ending tidak wajar)
    if txt and not txt[-1] in ['.', '!', '?']:
        # Cari titik terakhir yang valid
        last_period = txt.rfind('.')
        if last_period > 0:
            txt = txt[:last_period + 1]
        else:
            # Tidak ada titik, tambahkan di akhir
            txt += "."
    
    # Jika terlalu pendek
    if len(txt) < 10 or len(txt.split()) < 3:
        return "Obstacles detected ahead. Please proceed with caution."
    
    # Tambahkan navigasi jika belum ada kata kunci navigasi
    navigation_keywords = ["avoid", "careful", "stop", "climb", "move", "proceed", "use", "hold", "turn", "go", "keep", "distance", "safe"]
    has_navigation = any(word in txt.lower() for word in navigation_keywords)
    
    if not has_navigation:
        # Deteksi jenis objek untuk navigasi spesifik
        if any(word in txt.lower() for word in ["stair", "step"]):
            txt += " Use handrail carefully."
        elif any(word in txt.lower() for word in ["hole", "pothole", "crack"]):
            txt += " Avoid or stop."
        elif any(word in txt.lower() for word in ["vehicle", "motorcycle", "truck", "car"]):
            txt += " Keep safe distance."
        elif any(word in txt.lower() for word in ["chair", "table", "obstacle", "object"]):
            txt += " Avoid or move around."
        else:
            txt += " Proceed with caution."
    
    # Pastikan ada titik di akhir
    if not txt.endswith("."):
        txt += "."
    
    return txt

def main():
    
    # 1. Load FastSAM data
    print("\n[1] Loading FastSAM data...")
    fastsam_data = load_objects_info(json_path)
    objects = fastsam_data.get("objects", [])
    
    if objects:
        print(f"✓ Found {len(objects)} objects from FastSAM\n")
    else:
        print("✗ No FastSAM data found, using visual-only mode\n")
    
    # 2. Build segments info untuk prompt
    segments_info = build_segments_info(objects)
    if segments_info:
        print("[2] DETECTED OBJECTS:")
        print(segments_info)
    
    # 3. Load dan encode gambar segmentasi
    print(f"\n[3] Loading image: {pic_path}")
    if not os.path.exists(pic_path):
        print(f"Error: Image not found!")
        return
    
    img = Image.open(pic_path).convert("RGB")
    img_b64 = encode_image_base64(pic_path)
    print(f"✓ Image loaded: {img.size[0]}x{img.size[1]}")

    # 4. Build prompt dengan posisi info
    prompt = build_prompt(segments_info)
    print("\n[4] PROMPT TO MODEL:")
    print("-" * 60)
    print(prompt)
    print("-" * 60)
    
    # 5. Query moondream
    en_raw = query_ollama_vision(MODEL_NAME, prompt, img_b64)

    # 6. Clean untuk TTS
    en_clean = clean_output_for_tts(en_raw)

    # 7. Simpan hasil
    os.makedirs(output_dir, exist_ok=True)
    name_noext = os.path.splitext(os.path.basename(pic_path))[0]
    txt_out = os.path.join(output_dir, f"{name_noext}.txt")
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write(en_clean)

    # 8. Display results
    print("\n[5] RESULTS:")
    print("=" * 60)
    print("RAW OUTPUT:")
    print(en_raw)
    print(f"\n(Output length: {len(en_raw)} chars, {len(en_raw.split())} words)")
    print("\n" + "-" * 60)
    print("CLEANED FOR TTS:")
    print(en_clean)
    print("=" * 60)
    
    print(f"\n✓ Text saved to: {txt_out}")
    
    # 9. Simpan full output JSON
    full_output = {
        "image_path": pic_path,
        "fastsam_objects": objects,
        "prompt_used": prompt,
        "moondream_raw": en_raw,
        "tts_output": en_clean,
        "stats": {
            "raw_length": len(en_raw),
            "raw_words": len(en_raw.split()),
            "clean_length": len(en_clean),
            "clean_words": len(en_clean.split())
        }
    }
    json_out = os.path.join(output_dir, f"{name_noext}_full.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"✓ Full output saved to: {json_out}")


if __name__ == "__main__":
    main()