import ast
import csv
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ----------------------------
# Config
# ----------------------------
SERVICE_EXPORT_URL = "https://maps.nashville.gov/arcgis/rest/services/Imagery/2023Imagery_WGS84/MapServer/export"
EXPORT_ROOT = Path("exports")
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

BBoxLL = Tuple[float, float, float, float]  

CSV_NASH = "inference_parking_masks_nash.csv"
CSV_COOK = "inference_parking_masks_cook.csv"

# ----------------------------
# Flask
# ----------------------------
app = Flask(__name__)
CORS(
    app,
    resources={
        r"/api/*": {"origins": "*"},
        r"/exports/*": {"origins": "*"},
    },
)

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


# ----------------------------
# SegFormer load 
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = SegformerImageProcessor(do_resize=True, size=(256, 256))

# If you truly never want to run the model, you can remove this block.
# Keeping it as-is from your file:
model = SegformerForSemanticSegmentation.from_pretrained(
    r"C:\Users\jayjo\Downloads\GT_Hackathon\segformer-parkseg12k_2"
).to(device)
model.eval()

PARKING_CLASS_ID = 0  


# ----------------------------
# CSV loading
# ----------------------------
def load_mask_csv(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns dict: image_name(stem) -> row_dict
    expects columns:
      image_name, bbox_ll_epsg4326, width_px, height_px, parking_mask_flat
    """
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV not found: {csv_path}", flush=True)
        return {}

    df = pd.read_csv(csv_path)
    df["image_name"] = df["image_name"].astype(str).str.replace(".png", "", regex=False)
    return {row["image_name"]: row.to_dict() for _, row in df.iterrows()}


MASKS_NASH = load_mask_csv(CSV_NASH)
MASKS_COOK = load_mask_csv(CSV_COOK)


def parse_bbox_ll(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (list, tuple)):
        return [float(x) for x in val]
    s = str(val).strip()
    try:
        out = ast.literal_eval(s)
        if isinstance(out, (list, tuple)) and len(out) == 4:
            return [float(x) for x in out]
    except Exception:
        pass
    return None


def parse_mask_flat(val, width: int, height: int):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None

    if isinstance(val, str):
        try:
            arr = ast.literal_eval(val)
        except Exception:
            arr = [int(x) for x in val.replace("[", "").replace("]", "").split(",") if x.strip() != ""]
    else:
        arr = val

    arr = np.array(arr, dtype=np.uint8)
    if arr.size != width * height:
        raise ValueError(f"mask_flat size {arr.size} != width*height {width*height}")

    mask = arr.reshape((height, width))
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    return mask


def save_mask_and_overlay_from_csv(img_path: Path, mask_path: Path, overlay_path: Path, city_key: str, alpha=0.4):
    stem = img_path.stem  # "cell_00001"

    lookup = MASKS_NASH if city_key == "nashville" else MASKS_COOK
    row = lookup.get(stem)
    if not row:
        # fallback try both
        row = MASKS_NASH.get(stem) or MASKS_COOK.get(stem)
    if not row:
        raise FileNotFoundError(f"No CSV row found for {stem} in city {city_key}")

    bbox_ll = parse_bbox_ll(row.get("bbox_ll_epsg4326"))
    width = int(row.get("width_px", 1024))
    height = int(row.get("height_px", 1024))
    mask = parse_mask_flat(row.get("parking_mask_flat"), width=width, height=height)

    # Save mask
    Image.fromarray(mask).save(mask_path)

    # Make overlay
    img = Image.open(img_path).convert("RGB")
    if img.size != (width, height):
        mask_img = Image.fromarray(mask).resize(img.size, resample=Image.NEAREST)
    else:
        mask_img = Image.fromarray(mask)

    mask_l = mask_img.convert("L")
    red = Image.new("RGB", img.size, (255, 0, 0))
    overlay = Image.composite(red, img, mask_l)
    blended = Image.blend(img, overlay, alpha=alpha)
    blended.save(overlay_path)

    coverage = float((np.array(mask_img) > 0).mean())
    return bbox_ll, coverage


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
        print(f"[WARN] Non-image response: {ctype}\n{txt}\n", flush=True)
        return False

    with open(out_file, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
    return True


# ----------------------------
# Static serving
# ----------------------------
@app.route("/exports/<job_id>/<filename>")
def serve_export(job_id, filename):
    return send_from_directory(EXPORT_ROOT / job_id, filename)


# ----------------------------
# API: imagery 
# ----------------------------
@app.route("/api/imagery", methods=["POST"])
def api_imagery():
    data = request.get_json(force=True)

    lat = float(data["lat"])
    lon = float(data["lon"])
    radius_m = float(data.get("radius_m", 1000))

    step_deg = float(data.get("step_deg", 0.005))
    size_px = int(data.get("size_px", 1024))
    max_tiles = int(data.get("max_tiles", 25))

    bbox = bbox_from_point_radius_ll(lon, lat, radius_m)
    cells = bbox_grid_ll(bbox, step_deg)
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

    return jsonify(
        {
            "job_id": job_id,
            "bbox": bbox,
            "tiles": saved_urls,
            "tile_count": len(saved_urls),
            "note": "Tiles are square exports clipped by bbox grid.",
        }
    )


# ----------------------------
# API: segment
# ----------------------------
@app.route("/api/segment", methods=["POST"])
def api_segment():
    print("HIT /api/segment", flush=True)
    data = request.get_json(silent=True) or {}
    job_id = data.get("job_id")
    city_key = (data.get("city_key") or "").lower()

    if not job_id:
        return jsonify({"error": "Missing job_id"}), 400
    if city_key not in ("nashville", "cookeville"):
        return jsonify({"error": "Missing/invalid city_key (nashville/cookeville)"}), 400

    return _segment_job(job_id, city_key)


def _segment_job(job_id: str, city_key: str):
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
            bbox_ll, _coverage = save_mask_and_overlay_from_csv(
                img_path=p,
                mask_path=mask_file,
                overlay_path=overlay_file,
                city_key=city_key,
                alpha=0.4,
            )
        else:
            row = (MASKS_NASH if city_key == "nashville" else MASKS_COOK).get(stem)
            bbox_ll = parse_bbox_ll(row.get("bbox_ll_epsg4326")) if row else None

        results.append(
            {
                "image_url": f"/exports/{job_id}/{p.name}",
                "mask_url": f"/exports/{job_id}/{mask_file.name}",
                "overlay_url": f"/exports/{job_id}/{overlay_file.name}",
                "bbox_ll": bbox_ll,
            }
        )

    return jsonify({"job_id": job_id, "count": len(results), "results": results})


# ----------------------------
# Overlay APIs
# ----------------------------
@app.route("/api/overlays", methods=["GET"])
def api_overlays():
    city_key = (request.args.get("city_key") or "").lower().strip()
    if city_key not in ("nashville", "cookeville"):
        return jsonify({"error": "city_key must be nashville or cookeville"}), 400

    lookup = MASKS_NASH if city_key == "nashville" else MASKS_COOK

    # Find every job dir under exports
    results = []

    for job_dir in sorted(EXPORT_ROOT.glob("*")):
        if not job_dir.is_dir():
            continue

        for tile_path in sorted(job_dir.glob("cell_*.png")):
            if tile_path.name.endswith("_overlay.png") or tile_path.name.endswith("_mask.png"):
                continue

            stem = tile_path.stem  # cell_00012
            row = lookup.get(stem)
            if not row:
                continue

            bbox_ll = parse_bbox_ll(row.get("bbox_ll_epsg4326"))
            if not bbox_ll:
                continue

            overlay_file = job_dir / f"{stem}_overlay.png"
            mask_file = job_dir / f"{stem}_mask.png"

            if not overlay_file.exists() or not mask_file.exists():
                try:
                    save_mask_and_overlay_from_csv(
                        img_path=tile_path,
                        mask_path=mask_file,
                        overlay_path=overlay_file,
                        city_key=city_key,
                        alpha=0.4,
                    )
                except Exception as e:
                    print(f"[WARN] failed to create overlay for {tile_path}: {e}", flush=True)
                    continue

            results.append(
                {
                    "job_id": job_dir.name,
                    "overlay_url": f"/exports/{job_dir.name}/{overlay_file.name}",
                    "bbox_ll": bbox_ll,
                    "tile_name": tile_path.name,
                }
            )

    return jsonify({"city_key": city_key, "count": len(results), "results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)