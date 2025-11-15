from ultralytics import FastSAM
import os, cv2, numpy as np
import json

# ====== KONFIG ======
WEIGHTS     = "models/FastSAM-x.pt"
AREA_THRESH = 5000
TOP_K       = 10
MAX_AREA_RATIO = 0.50
MAX_COVER_WH   = 0.90
AR_MAX         = 8.0
MIN_THIN_PX    = 40
SOLID_MIN      = 0.55
TOP_BORDER_PAD = 20
NMS_IOU        = 0.5

SAVE_DIR = os.path.join(os.getcwd(), "runs", "fastsam_near")
CROP_DIR = os.path.join(SAVE_DIR, "crops")


def preprocess_image(image_path: str) -> str:
    """Simple pre-processing: denoise + sharpen + auto-brightness"""
    img = cv2.imread(image_path)
    
    # 1. Denoise
    img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    
    # 2. Auto brightness/contrast (CLAHE)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    
    # 3. Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    
    # Save
    processed_path = os.path.join(SAVE_DIR, "preprocessed.png")
    cv2.imwrite(processed_path, img)
    print(f"[Pre-processing] Done: {processed_path}")
    
    return processed_path


def solidity(mask):
    """Hitung solidity untuk filter objek tipis/tidak solid"""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return 0.0
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    return float(area / hull_area) if hull_area > 1e-6 else 0.0

def iou(a, b):
    """Intersection over Union untuk NMS"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/ua if ua > 0 else 0.0


def analyze_position(x1, y1, x2, y2, img_width, img_height):
    """Analisis posisi objek: left/center/right dan near/medium/far"""
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Horizontal position
    if center_x < img_width * 0.30:
        h_pos = "left"
    elif center_x < img_width * 0.70:
        h_pos = "center"
    else:
        h_pos = "right"
    
    # Vertical position
    if center_y > img_height * 0.65:
        v_pos = "near"
    elif center_y > img_height * 0.35:
        v_pos = "medium"
    else:
        v_pos = "far"
    
    return h_pos, v_pos

def add_position_overlay(image_path: str, segments_data: list) -> str:
    """
    segments_data: list of dict with keys: 'bbox', 'label', 'position'
    Example: [{'bbox': (x, y, w, h), 'label': 'obstacle', 'position': 'center-left'}]
    """
    img = cv2.imread(image_path)
    
    for seg in segments_data:
        x, y, w, h = seg['bbox']
        label = seg.get('label', 'object')
        position = seg.get('position', '')
        
        # Draw bounding box
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add text label with position
        text = f"{label} ({position})"
        cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 2)
    
    # Save temporary image
    overlay_path = "runs/fastsam_near/segmented_with_positions.png"
    cv2.imwrite(overlay_path, img)
    return overlay_path

def segment_objects(image_path: str, model=None, use_preprocess=True):
    """
    Melakukan segmentasi objek dekat.
    
    Args:
        image_path: Path ke gambar
        model: FastSAM model (optional)
        use_preprocess: True untuk pre-process gambar dulu
    
    Returns:
        dict: Info objek terdeteksi
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(CROP_DIR, exist_ok=True)
    
    # Pre-process jika diminta
    if use_preprocess:
        input_path = preprocess_image(image_path)
    else:
        input_path = image_path
    
    # Load model
    if model is None:
        model = FastSAM(WEIGHTS)
    
    results = model.predict(
        source=input_path,
        imgsz=640,
        conf=0.4,
        iou=0.7,
        retina_masks=True,
        device=0 if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu',
        save=False
    )
    
    objects_info = []
    segmented_path = None
    bbox_path = None
    
    for r in results:
        img = r.orig_img.copy()
        if r.masks is None:
            print("Tidak ada mask.")
            break

        H, W = img.shape[:2]
        frame_area = H * W

        masks  = r.masks.data.cpu().numpy().astype(np.uint8)
        boxes  = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()

        # Filter objek
        cand = []
        for i, m in enumerate(masks):
            area = int(m.sum())
            if area < AREA_THRESH:
                continue

            x1, y1, x2, y2 = boxes[i].astype(int)
            bw, bh = max(1, x2-x1), max(1, y2-y1)

            area_ratio = area / frame_area
            cover_w = bw / W
            cover_h = bh / H
            if area_ratio > MAX_AREA_RATIO or (cover_w > MAX_COVER_WH and cover_h > MAX_COVER_WH):
                continue

            ar = max(bw, bh) / max(1, min(bw, bh))
            if ar > AR_MAX and (bw < MIN_THIN_PX or bh < MIN_THIN_PX):
                continue

            if y1 <= TOP_BORDER_PAD and cover_w > 0.8:
                continue

            if solidity(m) < SOLID_MIN:
                continue

            cand.append((float(scores[i]), area, (x1, y1, x2, y2), i))

        if not cand:
            print("Tidak ada objek valid setelah filter bentuk.")
            break

        # NMS
        cand.sort(key=lambda x: (x[0], x[1]), reverse=True)
        kept = []
        for c in cand:
            _, _, box_c, _ = c
            if all(iou(box_c, box_k) < NMS_IOU for _, _, box_k, _ in kept):
                kept.append(c)
        kept = kept[:TOP_K]

        # Analisis posisi
        for idx, (score, area, (x1, y1, x2, y2), mi) in enumerate(kept, 1):
            h_pos, v_pos = analyze_position(x1, y1, x2, y2, W, H)
            objects_info.append({
                'id': idx,
                'area': area,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'h_position': h_pos,
                'v_position': v_pos,
                'score': float(score)
            })

        # Visual bbox
        vis = img.copy()
        for obj in objects_info:
            x1, y1, x2, y2 = obj['bbox']
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label = f"#{obj['id']} {obj['h_position']}-{obj['v_position']}"
            cv2.putText(vis, label, (x1, max(20, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
        bbox_path = os.path.join(SAVE_DIR, "bbox_near.png")
        cv2.imwrite(bbox_path, vis)

        # Segmented image
        union_mask = np.zeros((H, W), dtype=np.uint8)
        for _, _, _, mi in kept:
            union_mask = cv2.bitwise_or(union_mask, (masks[mi] * 255))

        fg = cv2.bitwise_and(img, img, mask=union_mask)
        bg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
        bg = (bg * 0.25).astype(np.uint8)
        inv_mask = cv2.bitwise_not(union_mask)
        bg_masked = cv2.bitwise_and(bg, bg, mask=inv_mask)
        segmented = cv2.add(fg, bg_masked)

        segmented_path = os.path.join(SAVE_DIR, "segmented.png")
        cv2.imwrite(segmented_path, segmented)
        
        print(f"[OK] Ditemukan {len(objects_info)} objek")
        print(f"[OK] Segmented: {segmented_path}")
        print(f"[OK] Bbox: {bbox_path}")
        break
    
    result = {
        'segmented_image_path': segmented_path,
        'bbox_image_path': bbox_path,
        'objects': objects_info
    }
    
    # Save JSON
    json_path = os.path.join(SAVE_DIR, "objects_info.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[OK] JSON: {json_path}")
    
    return result


if __name__ == "__main__":
    image_path = r"E:\Skripsi\FastSAM\jalan_berlubang.jpg"
    
    # Dengan pre-processing
    result = segment_objects(image_path, use_preprocess=True)
    
    # Atau tanpa pre-processing
    # result = segment_objects(image_path, use_preprocess=False)
    
    print("\n=== OBJECTS DETECTED ===")
    for obj in result['objects']:
        print(f"Object #{obj['id']}: {obj['h_position']}-{obj['v_position']} (area={obj['area']})")