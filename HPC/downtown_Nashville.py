import math
import json
from pathlib import Path
from typing import List, Tuple

import requests

# ----------------------------
# CONFIG (Nashville 2023 Ortho)
# ----------------------------
SERVICE_EXPORT_URL = "https://maps.nashville.gov/arcgis/rest/services/Imagery/2023Imagery_WGS84/MapServer/export"

OUT_DIR = Path("downtown_nashville_2023")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Downtown-ish Nashville bbox (EPSG:4326 lon/lat)
# Covers: Broadway, Gulch, Midtown edge, East bank edge (adjust as needed)
DOWNTOWN_BBOX_LL = (-86.820, 36.145, -86.760, 36.180)

# Zoom controls
STEP_DEG = 0.0015          # smaller = more zoom (more tiles)
SIZE_PX = (1024, 1024)     # pixels per tile
FMT = "png"
OVERWRITE = True
TIMEOUT = 120

R = 6378137.0
BBoxLL = Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat) EPSG:4326


# ----------------------------
# Web Mercator helpers (EPSG:4326 -> EPSG:3857)
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


# ----------------------------
# Geo sidecars (location attached to PNG)
# ----------------------------
def write_worldfile_for_png(out_png: Path, bbox_3857, width_px: int, height_px: int):
    """
    Writes .pgw (world file) so the PNG can be georeferenced in GIS tools.
    bbox_3857 = (minx, miny, maxx, maxy) meters in EPSG:3857
    """
    minx, miny, maxx, maxy = bbox_3857
    A = (maxx - minx) / width_px          # pixel size x
    E = -(maxy - miny) / height_px        # pixel size y (negative for north-up)
    C = minx + A / 2.0                    # x center of top-left pixel
    F = maxy + E / 2.0                    # y center of top-left pixel

    out_png.with_suffix(".pgw").write_text(f"{A}\n0.0\n0.0\n{E}\n{C}\n{F}\n")

def write_prj_3857(out_png: Path):
    """
    Writes .prj (projection) alongside the PNG.
    """
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

def write_tile_metadata(out_png: Path, bbox_ll: BBoxLL, bbox_3857, size_px: Tuple[int, int]):
    """
    Writes .json metadata alongside the PNG, including bbox and center location.
    """
    min_lon, min_lat, max_lon, max_lat = bbox_ll
    center_lon = (min_lon + max_lon) / 2.0
    center_lat = (min_lat + max_lat) / 2.0

    w, h = int(size_px[0]), int(size_px[1])
    meta = {
        "image": out_png.name,
        "crs": "EPSG:3857",
        "bbox_ll_epsg4326": [min_lon, min_lat, max_lon, max_lat],
        "center_ll_epsg4326": [center_lon, center_lat],
        "bbox_3857_m": [bbox_3857[0], bbox_3857[1], bbox_3857[2], bbox_3857[3]],
        "width_px": w,
        "height_px": h,
        "pixel_size_m": [
            (bbox_3857[2] - bbox_3857[0]) / w,
            (bbox_3857[3] - bbox_3857[1]) / h,
        ],
    }
    out_png.with_suffix(".json").write_text(json.dumps(meta, indent=2))


# ----------------------------
# ArcGIS export download
# ----------------------------
def export_cell_image(bbox_ll: BBoxLL, out_file: Path, size_px=(1024, 1024), fmt="png", timeout=120) -> bool:
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
        # ArcGIS sometimes returns JSON errors with status 200
        try:
            txt = r.text[:300]
        except Exception:
            txt = "<no text>"
        print(f"[WARN] Non-image response for {out_file.name}: {ctype} | {txt}")
        return False

    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Save PNG
    with open(out_file, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)

    # ✅ Attach location sidecars
    write_worldfile_for_png(out_file, bbox_3857, w, h)                 # .pgw
    write_prj_3857(out_file)                                           # .prj
    write_tile_metadata(out_file, bbox_ll, bbox_3857, (w, h))           # .json
    out_file.with_suffix(".bbox.txt").write_text(f"{bbox_ll}\n")        # .bbox.txt (easy human read)

    return True


# ----------------------------
# Main
# ----------------------------
def main():
    cells = bbox_grid_ll(DOWNTOWN_BBOX_LL, STEP_DEG)
    print(f"Downtown Nashville cells: {len(cells)}  step_deg={STEP_DEG}  size_px={SIZE_PX}")

    ok_count = 0
    for i, cell in enumerate(cells):
        out_img = OUT_DIR / f"cell_{i:05d}.png"

        # If PNG exists and overwrite is False, make sure sidecars exist anyway
        if out_img.exists() and not OVERWRITE:
            bbox_3857 = bbox_ll_to_bbox_3857(cell)
            w, h = SIZE_PX

            if not out_img.with_suffix(".pgw").exists():
                write_worldfile_for_png(out_img, bbox_3857, w, h)
            if not out_img.with_suffix(".prj").exists():
                write_prj_3857(out_img)
            if not out_img.with_suffix(".json").exists():
                write_tile_metadata(out_img, cell, bbox_3857, (w, h))
            if not out_img.with_suffix(".bbox.txt").exists():
                out_img.with_suffix(".bbox.txt").write_text(f"{cell}\n")

            ok_count += 1
            continue

        ok = export_cell_image(cell, out_img, size_px=SIZE_PX, fmt=FMT, timeout=TIMEOUT)
        if ok:
            ok_count += 1

    print(f"Done. Tiles with location sidecars: {ok_count}/{len(cells)}")
    print("Output folder:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()