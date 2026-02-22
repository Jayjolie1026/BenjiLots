import React, { useEffect, useMemo, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import * as turf from "@turf/turf";
import "maplibre-gl/dist/maplibre-gl.css";
import BenjiLotsCar from "../assets/Benji Lots Car.png";

const PRIMARY = "#00D4FF";
const PRIMARY_DIM = "rgba(0, 212, 255, 0.18)";
const ACCENT = "#FF6B35";

const CITIES = {
  nashville: {
    label: "Nashville",
    abbr: "BNA",
    center: [-86.7816, 36.1627],
    zoom: 12,
  },
  cookeville: {
    label: "Cookeville",
    abbr: "CVL",
    center: [-85.5016, 36.1628],
    zoom: 13,
  },
};

const SOURCE_ID = "radius-source";
const FILL_LAYER_ID = "radius-fill";
const LINE_LAYER_ID = "radius-outline";
const CENTER_SOURCE_ID = "center-source";
const CENTER_LAYER_ID = "center-point";
const CROSSHAIR_SOURCE_ID = "crosshair-source";
const CROSSHAIR_LAYER_H = "crosshair-h";
const CROSSHAIR_LAYER_V = "crosshair-v";

function makeCircleGeoJSON(centerLngLat, radiusMeters) {
  const [lng, lat] = centerLngLat;
  return turf.circle([lng, lat], radiusMeters / 1000, {
    steps: 128,
    units: "kilometers",
  });
}

function makeCrosshair(centerLngLat, radiusMeters) {
  const [lng, lat] = centerLngLat;
  const offset = (radiusMeters / 1000 / 111.32) * 0.35;
  return {
    h: turf.lineString([[lng - offset, lat], [lng + offset, lat]]),
    v: turf.lineString([[lng, lat - offset], [lng, lat + offset]]),
  };
}

export default function MapUI() {
  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);

  const [selectedCityKey, setSelectedCityKey] = useState("cookeville");
  const [radiusMeters, setRadiusMeters] = useState(500);
  const [centerLngLat, setCenterLngLat] = useState(null);
  const [mapLoaded, setMapLoaded] = useState(false);
  const [hoverLngLat, setHoverLngLat] = useState(null);
  const [flyingTo, setFlyingTo] = useState(false);

  const selectedCity = CITIES[selectedCityKey];

  const radiusLabel = useMemo(() => {
    if (radiusMeters < 1000) return `${radiusMeters}m`;
    const km = radiusMeters / 1000;
    return `${km.toFixed(km < 10 ? 2 : 1)}km`;
  }, [radiusMeters]);

  const areaLabel = useMemo(() => {
    const r = radiusMeters;
    const area = Math.PI * r * r;
    if (area < 1_000_000) return `${(area / 1000).toFixed(1)}k m²`;
    return `${(area / 1_000_000).toFixed(3)} km²`;
  }, [radiusMeters]);

  useEffect(() => {
    if (!mapContainerRef.current) return;

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: {
        version: 8,
        sources: {
          "satellite": {
            type: "raster",
            tiles: [
              "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            ],
            tileSize: 256,
          },
        },
        layers: [
          {
            id: "base-raster",
            type: "raster",
            source: "satellite",
            paint: { "raster-saturation": -0.3, "raster-brightness-min": 0.05 },
          },
        ],
      },
      center: selectedCity.center,
      zoom: selectedCity.zoom,
    });

    mapRef.current = map;

    map.addControl(new maplibregl.NavigationControl({ visualizePitch: false }), "top-right");

    map.on("load", () => {
      // Radius fill
      map.addSource(SOURCE_ID, { type: "geojson", data: turf.featureCollection([]) });
      map.addLayer({
        id: FILL_LAYER_ID,
        type: "fill",
        source: SOURCE_ID,
        paint: { "fill-color": PRIMARY, "fill-opacity": 0.1 },
      });
      map.addLayer({
        id: LINE_LAYER_ID,
        type: "line",
        source: SOURCE_ID,
        paint: {
          "line-color": PRIMARY,
          "line-width": 1.5,
          "line-opacity": 1,
          "line-dasharray": [6, 3],
        },
      });

      // Crosshair lines
      map.addSource(CROSSHAIR_SOURCE_ID, { type: "geojson", data: turf.featureCollection([]) });
      map.addLayer({
        id: CROSSHAIR_LAYER_H,
        type: "line",
        source: CROSSHAIR_SOURCE_ID,
        filter: ["==", "$type", "LineString"],
        paint: { "line-color": PRIMARY, "line-width": 1, "line-opacity": 0.5, "line-dasharray": [3, 2] },
      });

      // Center point
      map.addSource(CENTER_SOURCE_ID, { type: "geojson", data: turf.featureCollection([]) });
      map.addLayer({
        id: CENTER_LAYER_ID,
        type: "circle",
        source: CENTER_SOURCE_ID,
        paint: {
          "circle-radius": 5,
          "circle-color": ACCENT,
          "circle-stroke-color": "#fff",
          "circle-stroke-width": 1.5,
          "circle-opacity": 1,
        },
      });

      map.on("click", (e) => {
        const lngLat = [e.lngLat.lng, e.lngLat.lat];
        setCenterLngLat(lngLat);
      });

      map.on("mousemove", (e) => {
        setHoverLngLat([e.lngLat.lng, e.lngLat.lat]);
      });
      map.on("mouseleave", () => setHoverLngLat(null));

      map.getCanvas().style.cursor = "crosshair";
      setMapLoaded(true);
    });

    return () => { map.remove(); mapRef.current = null; };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    setFlyingTo(true);
    map.flyTo({
      center: selectedCity.center,
      zoom: selectedCity.zoom,
      speed: 1.4,
      curve: 1.4,
      essential: true,
    });
    setCenterLngLat(null);
    const timer = setTimeout(() => setFlyingTo(false), 1800);
    return () => clearTimeout(timer);
  }, [selectedCityKey]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded) return;

    const radiusSource = map.getSource(SOURCE_ID);
    const centerSource = map.getSource(CENTER_SOURCE_ID);
    const crosshairSource = map.getSource(CROSSHAIR_SOURCE_ID);
    if (!radiusSource || !centerSource || !crosshairSource) return;

    if (!centerLngLat) {
      radiusSource.setData(turf.featureCollection([]));
      centerSource.setData(turf.featureCollection([]));
      crosshairSource.setData(turf.featureCollection([]));
      return;
    }

    radiusSource.setData(makeCircleGeoJSON(centerLngLat, radiusMeters));
    centerSource.setData(turf.point(centerLngLat));
    const ch = makeCrosshair(centerLngLat, radiusMeters);
    crosshairSource.setData(turf.featureCollection([ch.h, ch.v]));
  }, [centerLngLat, radiusMeters, mapLoaded]);

  return (
    <div style={s.root}>
      <style>{css}</style>

      {/* Top bar */}
      <header style={s.header}>
        <div style={s.headerLeft}>
          <div style={s.logo}>
            <img
              src={BenjiLotsCar}
              alt="BenjiLots"
              style={{ width: 130, height: 80, display: "block" }}
            />
          </div>
          <div>
            <div style={s.appTitle}>BENJI LOTS</div>
            <div style={s.appSub}>Parking Lot Locater</div>
          </div>
        </div>

        <div style={s.cityToggle}>
          {Object.entries(CITIES).map(([key, city]) => (
            <button
              key={key}
              style={{
                ...s.cityBtn,
                ...(selectedCityKey === key ? s.cityBtnActive : {}),
              }}
              onClick={() => setSelectedCityKey(key)}
              className="city-btn"
            >
              <span style={s.cityAbbr}>{city.abbr}</span>
              <span style={s.cityName}>{city.label}</span>
            </button>
          ))}
        </div>

        <div style={s.statusBar}>
          {flyingTo && <span style={s.flyingBadge}>NAVIGATING</span>}
          {hoverLngLat && (
            <span style={s.coordReadout}>
              {hoverLngLat[1].toFixed(5)}°N · {Math.abs(hoverLngLat[0]).toFixed(5)}°W
            </span>
          )}
        </div>
      </header>

      {/* Map */}
      <div style={s.mapWrap}>
        <div ref={mapContainerRef} style={s.map} />

        {/* Scanline overlay */}
        <div style={s.scanlines} />

        {/* Corner reticles */}
        <div style={{...s.reticle, top: 12, left: 12, borderTop: `2px solid ${PRIMARY}`, borderLeft: `2px solid ${PRIMARY}`}} />
        <div style={{...s.reticle, top: 12, right: 12, borderTop: `2px solid ${PRIMARY}`, borderRight: `2px solid ${PRIMARY}`}} />
        <div style={{...s.reticle, bottom: 12, left: 12, borderBottom: `2px solid ${PRIMARY}`, borderLeft: `2px solid ${PRIMARY}`}} />
        <div style={{...s.reticle, bottom: 12, right: 12, borderBottom: `2px solid ${PRIMARY}`, borderRight: `2px solid ${PRIMARY}`}} />

        {/* Side panel */}
        <div style={s.panel}>
          <div style={s.panelSection}>
            <div style={s.panelLabel}>RADIUS</div>
            <div style={s.radiusDisplay}>
              <span style={s.radiusBig}>{radiusLabel}</span>
            </div>

            <input
              type="range"
              min={50}
              max={5000}
              step={25}
              value={radiusMeters}
              onChange={(e) => setRadiusMeters(Number(e.target.value))}
              style={s.slider}
              className="radius-slider"
            />
            <div style={s.sliderTicks}>
              <span>50m</span>
              <span>1km</span>
              <span>2.5km</span>
              <span>5km</span>
            </div>
          </div>

          <div style={s.divider} />

          <div style={s.panelSection}>
            <div style={s.panelLabel}>COVERAGE AREA</div>
            <div style={s.statRow}>
              <span style={s.statVal}>{areaLabel}</span>
            </div>
          </div>

          <div style={s.divider} />

          <div style={s.panelSection}>
            <div style={s.panelLabel}>CENTER POINT</div>
            {centerLngLat ? (
              <div style={s.coordBlock}>
                <div style={s.coordRow}>
                  <span style={s.coordKey}>LAT</span>
                  <span style={s.coordVal}>{centerLngLat[1].toFixed(6)}°</span>
                </div>
                <div style={s.coordRow}>
                  <span style={s.coordKey}>LNG</span>
                  <span style={s.coordVal}>{centerLngLat[0].toFixed(6)}°</span>
                </div>
                <button style={s.clearBtn} onClick={() => setCenterLngLat(null)}>
                  CLEAR SELECTION
                </button>
              </div>
            ) : (
              <div style={s.hint}>
                <div style={s.hintIcon}>+</div>
                <div style={s.hintText}>Click anywhere on the map to drop a pin and draw a radius.</div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

const s = {
  root: {
    height: "100vh",
    display: "flex",
    flexDirection: "column",
    background: "#060A0D",
    color: "#C8DDE8",
    fontFamily: "'Courier New', 'Lucida Console', monospace",
    overflow: "hidden",
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "0 20px",
    height: 56,
    background: "rgba(4, 8, 11, 0.98)",
    borderBottom: "1px solid rgba(0, 212, 255, 0.2)",
    gap: 16,
    flexShrink: 0,
    zIndex: 10,
  },
  headerLeft: {
    display: "flex",
    alignItems: "center",
    gap: 12,
  },
  logo: {
    opacity: 0.9,
  },
  appTitle: {
    fontSize: 15,
    fontWeight: 700,
    letterSpacing: 5,
    color: PRIMARY,
  },
  appSub: {
    fontSize: 9,
    letterSpacing: 2,
    opacity: 0.5,
    marginTop: 1,
    textTransform: "uppercase",
  },
  cityToggle: {
    display: "flex",
    gap: 4,
    background: "rgba(0,0,0,0.5)",
    border: "1px solid rgba(0, 212, 255, 0.2)",
    borderRadius: 4,
    padding: 3,
  },
  cityBtn: {
    background: "transparent",
    border: "1px solid transparent",
    color: "rgba(200, 221, 232, 0.5)",
    cursor: "pointer",
    padding: "6px 14px",
    borderRadius: 3,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 1,
    transition: "all 0.15s",
  },
  cityBtnActive: {
    background: PRIMARY_DIM,
    border: `1px solid rgba(0, 212, 255, 0.4)`,
    color: PRIMARY,
  },
  cityAbbr: {
    fontSize: 12,
    fontWeight: 700,
    letterSpacing: 2,
  },
  cityName: {
    fontSize: 8,
    letterSpacing: 1,
    opacity: 0.7,
  },
  statusBar: {
    display: "flex",
    alignItems: "center",
    gap: 12,
    minWidth: 280,
    justifyContent: "flex-end",
  },
  flyingBadge: {
    fontSize: 9,
    letterSpacing: 3,
    color: ACCENT,
    border: `1px solid ${ACCENT}`,
    padding: "2px 8px",
    borderRadius: 2,
    animation: "pulse 0.8s ease-in-out infinite",
  },
  coordReadout: {
    fontSize: 10,
    letterSpacing: 1,
    color: "rgba(200, 221, 232, 0.45)",
    fontVariantNumeric: "tabular-nums",
  },
  mapWrap: {
    flex: 1,
    position: "relative",
    overflow: "hidden",
  },
  map: {
    position: "absolute",
    inset: 0,
  },
  scanlines: {
    position: "absolute",
    inset: 0,
    background: "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.03) 2px, rgba(0,0,0,0.03) 4px)",
    pointerEvents: "none",
    zIndex: 1,
  },
  reticle: {
    position: "absolute",
    width: 18,
    height: 18,
    zIndex: 2,
    opacity: 0.6,
  },
  panel: {
    position: "absolute",
    top: 16,
    left: 16,
    width: 220,
    background: "rgba(4, 8, 12, 0.92)",
    border: "1px solid rgba(0, 212, 255, 0.25)",
    borderRadius: 4,
    zIndex: 5,
    backdropFilter: "blur(12px)",
    overflow: "hidden",
  },
  panelSection: {
    padding: "14px 16px",
  },
  panelLabel: {
    fontSize: 8,
    letterSpacing: 3,
    color: "rgba(0, 212, 255, 0.5)",
    marginBottom: 10,
    textTransform: "uppercase",
  },
  radiusDisplay: {
    marginBottom: 12,
  },
  radiusBig: {
    fontSize: 32,
    fontWeight: 700,
    color: PRIMARY,
    letterSpacing: -1,
    fontVariantNumeric: "tabular-nums",
  },
  slider: {
    width: "100%",
    accentColor: PRIMARY,
    cursor: "pointer",
  },
  sliderTicks: {
    display: "flex",
    justifyContent: "space-between",
    fontSize: 8,
    letterSpacing: 0.5,
    opacity: 0.4,
    marginTop: 4,
  },
  divider: {
    height: 1,
    background: "rgba(0, 212, 255, 0.12)",
    margin: "0 16px",
  },
  statRow: {
    display: "flex",
    alignItems: "baseline",
    gap: 6,
  },
  statVal: {
    fontSize: 18,
    fontWeight: 700,
    color: "rgba(200, 221, 232, 0.85)",
    fontVariantNumeric: "tabular-nums",
  },
  coordBlock: {
    display: "flex",
    flexDirection: "column",
    gap: 6,
  },
  coordRow: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  coordKey: {
    fontSize: 9,
    letterSpacing: 2,
    color: "rgba(0, 212, 255, 0.5)",
  },
  coordVal: {
    fontSize: 11,
    fontVariantNumeric: "tabular-nums",
    color: "rgba(200, 221, 232, 0.9)",
  },
  clearBtn: {
    marginTop: 10,
    width: "100%",
    padding: "6px 0",
    background: "transparent",
    border: `1px solid rgba(255, 107, 53, 0.4)`,
    color: ACCENT,
    fontSize: 9,
    letterSpacing: 2,
    cursor: "pointer",
    borderRadius: 2,
    transition: "all 0.15s",
  },
  hint: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 8,
    padding: "8px 0",
    opacity: 0.6,
  },
  hintIcon: {
    fontSize: 28,
    color: PRIMARY,
    lineHeight: 1,
    opacity: 0.4,
  },
  hintText: {
    fontSize: 10,
    textAlign: "center",
    lineHeight: 1.5,
    letterSpacing: 0.3,
    opacity: 0.7,
  },
  attribution: {
    position: "absolute",
    bottom: 8,
    right: 8,
    fontSize: 9,
    opacity: 0.35,
    zIndex: 5,
    letterSpacing: 0.5,
  },
};

const css = `
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }
  .city-btn:hover {
    background: rgba(0, 212, 255, 0.08) !important;
    color: rgba(0, 212, 255, 0.8) !important;
  }
  .radius-slider::-webkit-slider-thumb {
    width: 14px;
    height: 14px;
  }
  .maplibregl-ctrl-top-right {
    top: 16px !important;
    right: 16px !important;
  }
  .maplibregl-ctrl button {
    background-color: rgba(4, 8, 12, 0.92) !important;
    border-color: rgba(0, 212, 255, 0.25) !important;
  }
  .maplibregl-ctrl button span {
    filter: invert(1) !important;
  }
`;
