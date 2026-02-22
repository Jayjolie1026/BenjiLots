
import math
import json
from pathlib import Path
from typing import List, Tuple

import requests

# --- TDOT statewide imagery (Web Mercator) ---
SERVICE_EXPORT_URL = "https://tnmap.tn.gov/arcgis/rest/services/BASEMAPS/IMAGERY_WEB_MERCATOR/MapServer/export"

OUT_DIR = Path("downtown_cookeville_tdot")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Downtown Cookeville (center-ish)
LAT = 36.165325
LON = -85.505630

RADIUS_M = 3400        # ~0.75 miles around downtown
STEP_DEG = 0.003         # try 0.0015 for more zoom (more tiles)
SIZE_PX = (1024, 1024)
FMT = "png"
OVERWRITE = True

R = 6378137.0
BBoxLL = Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat) EPSG:4326


# ----------------------------
# Projection helpers
# ----------------------------
def lonlat_to_webmerc(lon: float, lat: float):
    lat = max(min(lat, 85.05112878), -85.05112878)
    x = R * math.radians(lon)
    y = R * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
    return x, y

def webmerc_to_lonlat(x: float, y: float):
    lon = math.degrees(x / R)
    lat = math.degrees(2.0 * math.atan(math.exp(y / R)) - math.pi / 2.0)
    return lon, lat

def bbox_3857_to_bbox_ll(bbox_3857):
    minx, miny, maxx, maxy = bbox_3857
    lon0, lat0 = webmerc_to_lonlat(minx, miny)
    lon1, lat1 = webmerc_to_lonlat(maxx, maxy)
    return (min(lon0, lon1), min(lat0, lat1), max(lon0, lon1), max(lat0, lat1))

def bbox_from_point_radius_ll(lon: float, lat: float, radius_m: float) -> BBoxLL:
    cx, cy = lonlat_to_webmerc(lon, lat)
    bbox_3857 = (cx - radius_m, cy - radius_m, cx + radius_m, cy + radius_m)
    return bbox_3857_to_bbox_ll(bbox_3857)

def bbox_ll_to_bbox_3857(bbox: BBoxLL):
    min_lon, min_lat, max_lon, max_lat = bbox
    x0, y0 = lonlat_to_webmerc(min_lon, min_lat)
    x1, y1 = lonlat_to_webmerc(max_lon, max_lat)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


# ----------------------------
# Tiling
# ----------------------------
def bbox_grid_ll(bbox: BBoxLL, step_deg: float) -> List[BBoxLL]:
    minx, miny, maxx, maxy = bbox
    xs, ys = [], []
    x = minx
    while x < maxx:
        xs.append(x); x += step_deg
    y = miny
    while y < maxy:
        ys.append(y); y += step_deg

    cells: List[BBoxLL] = []
    for x0 in xs:
        for y0 in ys:
            x1 = min(x0 + step_deg, maxx)
            y1 = min(y0 + step_deg, maxy)
            cells.append((x0, y0, x1, y1))
    return cells


# ----------------------------
# Geo sidecars (worldfile + metadata)
# ----------------------------
def write_worldfile_for_png(out_png: Path, bbox_3857, width_px: int, height_px: int):
    minx, miny, maxx, maxy = bbox_3857
    A = (maxx - minx) / width_px
    E = -(maxy - miny) / height_px
    C = minx + A / 2.0
    F = maxy + E / 2.0
    out_png.with_suffix(".pgw").write_text(f"{A}\n0.0\n0.0\n{E}\n{C}\n{F}\n")

def write_prj_3857(out_png: Path):
    prj = """PROJCS["WGS_1984_Web_Mercator_Auxiliary_Sphere",
GEOGCS["GCS_WGS_1984",
DATUM["D_WGS_1984",
SPHEROID["WGS_1984",6378137.0,298.257223563]],
PRIMEM["Greenwich",0.0],
UNIT["Degree",0.0174532925199433]],
PROJECTION["Mercator_Auxiliary_Sphere"],
PARAMETER["False_Easting",0.0],
PARAMETER["False_Northing",0.0],
PARAMETER["Central_Meridian",0.0],
PARAMETER["Standard_Parallel_1",0.0],
PARAMETER["Auxiliary_Sphere_Type",0.0],
UNIT["Meter",1.0]]"""
    out_png.with_suffix(".prj").write_text(prj)

def write_tile_metadata(out_png: Path, bbox_ll, bbox_3857, size_px):
    meta = {
        "image": out_png.name,
        "crs": "EPSG:3857",
        "bbox_ll_epsg4326": bbox_ll,
        "bbox_3857_m": bbox_3857,
        "width_px": int(size_px[0]),
        "height_px": int(size_px[1]),
        "pixel_size_m": [
            (bbox_3857[2] - bbox_3857[0]) / size_px[0],
            (bbox_3857[3] - bbox_3857[1]) / size_px[1],
        ],
    }
    out_png.with_suffix(".json").write_text(json.dumps(meta, indent=2))


# ----------------------------
# Download
# ----------------------------
def export_cell_image(bbox_ll: BBoxLL, out_file: Path, size_px=(1024,1024), fmt="png", timeout=120) -> bool:
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

    r = requests.get(SERVICE_EXPORT_URL, params=params, stream=True, timeout=timeout)
    ctype = r.headers.get("Content-Type", "")
    if "image" not in ctype.lower():
        print("[WARN] Non-image response:", ctype, r.text[:200])
        return False

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)

    # ✅ Write coordinate sidecars AFTER file is complete
    write_worldfile_for_png(out_file, bbox_3857, w, h)
    write_prj_3857(out_file)
    write_tile_metadata(out_file, bbox_ll, bbox_3857, (w, h))

    return True


def main():
    bbox = bbox_from_point_radius_ll(LON, LAT, RADIUS_M)
    cells = bbox_grid_ll(bbox, STEP_DEG)
    print(f"Downtown Cookeville cells: {len(cells)} step_deg={STEP_DEG} size_px={SIZE_PX}")

    ok_count = 0
    for i, cell in enumerate(cells):
        out_img = OUT_DIR / f"cell_{i:05d}.png"
        if out_img.exists() and not OVERWRITE:
            ok_count += 1
            continue

        ok = export_cell_image(cell, out_img, size_px=SIZE_PX, fmt=FMT)
        if ok:
            ok_count += 1
            (OUT_DIR / f"cell_{i:05d}.bbox.txt").write_text(f"{cell}\n")

    print(f"Done. Downloaded {ok_count}/{len(cells)} tiles to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()