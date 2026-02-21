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


if __name__ == "__main__":
    # Run backend: python app.py
    app.run(host="0.0.0.0", port=5000, debug=True)