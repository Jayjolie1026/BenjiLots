import { useState } from "react";
import "./App.css";

export default function App() {
  const [lat, setLat] = useState("36.1627");
  const [lon, setLon] = useState("-86.7816");
  const [radius, setRadius] = useState("1200");
  const [stepDeg, setStepDeg] = useState("0.005");
  const [sizePx, setSizePx] = useState("1024");
  const [maxTiles, setMaxTiles] = useState("25");

  const [loading, setLoading] = useState(false);
  const [tiles, setTiles] = useState([]);
  const [jobId, setJobId] = useState(null);
  const [error, setError] = useState("");

  const backendBase = "http://localhost:5000";

  async function fetchImagery() {
    setLoading(true);
    setError("");
    setTiles([]);
    setJobId(null);

    try {
      const res = await fetch(`${backendBase}/api/imagery`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          lat: Number(lat),
          lon: Number(lon),
          radius_m: Number(radius),
          step_deg: Number(stepDeg),
          size_px: Number(sizePx),
          max_tiles: Number(maxTiles),
        }),
      });

      if (!res.ok) throw new Error(`Backend error: ${res.status}`);
      const json = await res.json();

      setJobId(json.job_id);
      setTiles(json.tiles.map((t) => `${backendBase}${t}`));
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="hero">
        <h1>ParkSight Imagery Fetch</h1>
        <p>Enter a coordinate + radius to pull Nashville 2023 ortho tiles.</p>
      </header>

      <section className="panel">
        <div className="grid">
          <label>
            Latitude
            <input value={lat} onChange={(e) => setLat(e.target.value)} />
          </label>

          <label>
            Longitude
            <input value={lon} onChange={(e) => setLon(e.target.value)} />
          </label>

          <label>
            Radius (meters)
            <input value={radius} onChange={(e) => setRadius(e.target.value)} />
          </label>

          <label>
            step_deg (tile geo size)
            <input value={stepDeg} onChange={(e) => setStepDeg(e.target.value)} />
          </label>

          <label>
            size_px (tile pixels)
            <input value={sizePx} onChange={(e) => setSizePx(e.target.value)} />
          </label>

          <label>
            max_tiles
            <input value={maxTiles} onChange={(e) => setMaxTiles(e.target.value)} />
          </label>
        </div>

        <button className="btn" onClick={fetchImagery} disabled={loading}>
          {loading ? "Fetching..." : "Fetch imagery"}
        </button>

        {jobId && <div className="meta">Job: <b>{jobId}</b> — Tiles: <b>{tiles.length}</b></div>}
        {error && <div className="error">{error}</div>}
      </section>

      <section className="tiles">
        {tiles.map((src) => (
          <a key={src} href={src} target="_blank" rel="noreferrer">
            <img src={src} alt="tile" />
          </a>
        ))}
      </section>
    </div>
  );
}
