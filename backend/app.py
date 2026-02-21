import math
import uuid
from pathlib import Path
from typing import List, Tuple

import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ----------------------------
# Config
# ----------------------------
SERVICE_EXPORT_URL = "https://maps.nashville.gov/arcgis/rest/services/Imagery/2023Imagery_WGS84/MapServer/export"
EXPORT_ROOT = Path("exports")  # images saved here
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

BBoxLL = Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat) EPSG:4326

app = Flask(__name__)
CORS(app)  # allow React dev server


# ----------------------------
# Web Mercator helpers
# ----------------------------
R = 6378137.0

def lonlat_to_webmerc(lon: float, lat: float) -> Tuple[float, float]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    x = R * math.radians(lon)
    y = R * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
    return x, y

def webmerc_to_lonlat(x: float, y: float) -> Tuple[float, float]:
    lon = math.degrees(x / R)
    lat = math.degrees(2.0 * math.atan(math.exp(y / R)) - math.pi / 2.0)
    return lon, lat

def bbox_3857_to_bbox_ll(bbox_3857: Tuple[float, float, float, float]) -> BBoxLL:
    minx, miny, maxx, maxy = bbox_3857
    lon0, lat0 = webmerc_to_lonlat(minx, miny)
    lon1, lat1 = webmerc_to_lonlat(maxx, maxy)
    return (min(lon0, lon1), min(lat0, lat1), max(lon0, lon1), max(lat0, lat1))

def bbox_from_point_radius_ll(lon: float, lat: float, radius_m: float) -> BBoxLL:
    cx, cy = lonlat_to_webmerc(lon, lat)
    bbox_3857 = (cx - radius_m, cy - radius_m, cx + radius_m, cy + radius_m)
    return bbox_3857_to_bbox_ll(bbox_3857)


import os
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# --- load once at startup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_extractor = SegformerImageProcessor(do_resize=True, size=(256, 256))

model = SegformerForSemanticSegmentation.from_pretrained("C:\\Users\\jayjo\\Downloads\\GT_Hackathon\\segformer-parkseg12k").to(device)
model.eval()




import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# set this once (IMPORTANT)
PARKING_CLASS_ID = 1  # change if your label mapping uses a different id

def run_segformer_argmax_mask(pil_img: Image.Image, size_px=None):
    """
    Returns:
      pred_class: (H,W) uint8 class ids
      parking_mask: (H,W) uint8 0/255 for PARKING_CLASS_ID
    """
    pil_img = pil_img.convert("RGB")
    orig_w, orig_h = pil_img.size

    encoded = feature_extractor(pil_img, return_tensors="pt")
    pixel_values = encoded["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits  # [1, C, h, w]

    # upsample logits back to target resolution
    if size_px is None:
        target_h, target_w = orig_h, orig_w
    else:
        target_w, target_h = size_px  # (W,H)

    logits_up = F.interpolate(
        logits, size=(target_h, target_w),
        mode="bilinear", align_corners=False
    )

    pred_class = torch.argmax(logits_up, dim=1)[0].cpu().numpy().astype(np.uint8)  # (H,W)
    parking_mask = (pred_class == PARKING_CLASS_ID).astype(np.uint8) * 255

    return pred_class, parking_mask


def save_mask_and_overlay_argmax(img_path: str, mask_path: str, overlay_path: str, alpha=0.4):
    img = Image.open(img_path).convert("RGB")
    mask = run_segformer_sliding_window(img, patch_size=256, stride=128)
    coverage = float((mask > 0).mean())
    print(f"{img_path} parking_coverage={coverage:.4f}")

    Image.fromarray(mask).save(mask_path)

    # overlay red where mask is 255
    mask_l = Image.fromarray(mask).convert("L")
    red = Image.new("RGB", img.size, (255, 0, 0))
    overlay = Image.composite(red, img, mask_l)
    blended = Image.blend(img, overlay, alpha=alpha)
    blended.save(overlay_path)

# ----------------------------
# Grid / tiling
# ----------------------------
def bbox_grid_ll(bbox: BBoxLL, step_deg: float) -> List[BBoxLL]:
    minx, miny, maxx, maxy = bbox
    if minx >= maxx or miny >= maxy:
        raise ValueError("Invalid bbox order. Must be (min_lon, min_lat, max_lon, max_lat).")

    xs = []
    x = minx
    while x < maxx:
        xs.append(x)
        x += step_deg

    ys = []
    y = miny
    while y < maxy:
        ys.append(y)
        y += step_deg

    cells: List[BBoxLL] = []
    for x0 in xs:
        for y0 in ys:
            x1 = min(x0 + step_deg, maxx)
            y1 = min(y0 + step_deg, maxy)
            cells.append((x0, y0, x1, y1))
    return cells

def bbox_ll_to_bbox_3857(bbox: BBoxLL) -> Tuple[float, float, float, float]:
    min_lon, min_lat, max_lon, max_lat = bbox
    x0, y0 = lonlat_to_webmerc(min_lon, min_lat)
    x1, y1 = lonlat_to_webmerc(max_lon, max_lat)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

def run_segformer_sliding_window(pil_img, patch_size=256, stride=256):
    img = np.array(pil_img.convert("RGB"))
    H, W, _ = img.shape
    out_mask = np.zeros((H, W), dtype=np.uint8)

    ys = list(range(0, max(H - patch_size + 1, 1), stride))
    xs = list(range(0, max(W - patch_size + 1, 1), stride))
    if ys[-1] != H - patch_size: ys.append(H - patch_size)
    if xs[-1] != W - patch_size: xs.append(W - patch_size)

    for y0 in ys:
        for x0 in xs:
            patch = img[y0:y0+patch_size, x0:x0+patch_size]
            patch_pil = Image.fromarray(patch)

            encoded = feature_extractor(patch_pil, return_tensors="pt")
            pixel_values = encoded["pixel_values"].to(device)

            with torch.no_grad():
                logits = model(pixel_values=pixel_values).logits

            logits_up = F.interpolate(logits, size=(patch_size, patch_size),
                                      mode="bilinear", align_corners=False)
            pred_class = torch.argmax(logits_up, dim=1)[0].cpu().numpy().astype(np.uint8)
            pred = (pred_class == PARKING_CLASS_ID).astype(np.uint8) * 255

            out_mask[y0:y0+patch_size, x0:x0+patch_size] = pred

    return out_mask

# ----------------------------
# ArcGIS export downloader
# ----------------------------
def export_cell_image(
    bbox_ll: BBoxLL,
    out_file: Path,
    size_px: Tuple[int, int] = (1024, 1024),
    fmt: str = "png",
    timeout: int = 120,
) -> bool:
    bbox_3857 = bbox_ll_to_bbox_3857(bbox_ll)
    bbox_str = ",".join(f"{v:.3f}" for v in bbox_3857)
    w, h = size_px

    params = {
        "f": "image",
        "bbox": bbox_str,
        "bboxSR": "3857",
        "imageSR": "3857",
        "size": f"{w},{h}",
        "format": fmt,
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(SERVICE_EXPORT_URL, params=params, stream=True, timeout=timeout)

    ctype = r.headers.get("Content-Type", "")
    if "image" not in ctype.lower():
        try:
            txt = r.text[:500]
        except Exception:
            txt = "<no text>"
        print(f"[WARN] Non-image response: {ctype}\n{txt}\n")
        return False

    with open(out_file, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
    return True


# ----------------------------
# API + static serving
# ----------------------------
@app.route("/api/imagery", methods=["POST"])
def api_imagery():
    """
    Body JSON:
      { "lat": 36.1627, "lon": -86.7816, "radius_m": 1200,
        "step_deg": 0.005, "size_px": 1024, "max_tiles": 25 }
    """
    data = request.get_json(force=True)

    lat = float(data["lat"])
    lon = float(data["lon"])
    radius_m = float(data.get("radius_m", 1000))

    # controls
    step_deg = float(data.get("step_deg", 0.005))
    size_px = int(data.get("size_px", 1024))
    max_tiles = int(data.get("max_tiles", 25))  # IMPORTANT: keep runtime reasonable

    bbox = bbox_from_point_radius_ll(lon, lat, radius_m)
    cells = bbox_grid_ll(bbox, step_deg)

    # Limit tiles to avoid huge downloads in a demo
    cells = cells[:max_tiles]

    job_id = uuid.uuid4().hex[:10]
    job_dir = EXPORT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    saved_urls = []
    for i, cell in enumerate(cells):
        out_path = job_dir / f"cell_{i:05d}.png"
        ok = export_cell_image(cell, out_path, size_px=(size_px, size_px), fmt="png")
        if ok:
            saved_urls.append(f"/exports/{job_id}/{out_path.name}")

    return jsonify({
        "job_id": job_id,
        "bbox": bbox,
        "tiles": saved_urls,
        "tile_count": len(saved_urls),
        "note": "Tiles are square exports clipped by bbox grid."
    })

@app.route("/exports/<job_id>/<filename>")
def serve_export(job_id, filename):
    return send_from_directory(EXPORT_ROOT / job_id, filename)

from flask import jsonify, request




@app.route("/api/segment", methods=["POST"])
def api_segment():
    data = request.get_json(silent=True) or {}

    job_id = data.get("job_id")
    if not job_id:
        return jsonify({
            "error": "Missing job_id. Call /api/imagery first, then POST {job_id} to /api/segment."
        }), 400

    return _segment_job(job_id)


@app.route("/api/segment/<job_id>", methods=["GET"])
def api_segment_get(job_id):
    # easier testing in browser: GET /api/segment/<job_id>
    return _segment_job(job_id)


def _segment_job(job_id: str):
    job_dir = EXPORT_ROOT / job_id
    if not job_dir.exists():
        return jsonify({"error": f"job_id not found: {job_id}"}), 404

    tiles = sorted(job_dir.glob("cell_*.png"))
    tiles = [p for p in tiles if not p.name.endswith("_overlay.png") and not p.name.endswith("_mask.png")]

    results = []
    for p in tiles:
        stem = p.stem
        mask_file = job_dir / f"{stem}_mask.png"
        overlay_file = job_dir / f"{stem}_overlay.png"

        if not (mask_file.exists() and overlay_file.exists()):
            save_mask_and_overlay_argmax(
                img_path=str(p),
                mask_path=str(mask_file),
                overlay_path=str(overlay_file),
                alpha=0.4
            )

        results.append({
            "image_url": f"/exports/{job_id}/{p.name}",
            "mask_url": f"/exports/{job_id}/{mask_file.name}",
            "overlay_url": f"/exports/{job_id}/{overlay_file.name}",
        })

    return jsonify({"job_id": job_id, "count": len(results), "results": results})
if __name__ == "__main__":
    # Run backend: python app.py
    app.run(host="0.0.0.0", port=5000, debug=True)