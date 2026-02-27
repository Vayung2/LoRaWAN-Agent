from __future__ import annotations
import json, os, math
from typing import Dict, Tuple
from math import radians, cos

from .constants import SENSORS, GATEWAYS

R_EARTH = 6_371_000.0  # meters

def ll_to_xy(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    lat, lon, lat0, lon0 = map(radians, [lat, lon, lat0, lon0])
    x = (lon - lon0) * cos(lat0) * R_EARTH
    y = (lat - lat0) * R_EARTH
    return float(x), float(y)

def main(out_path: str = "models/metadata.json") -> None:
    all_lats = [v["lat"] for v in SENSORS.values()] + [v["lat"] for v in GATEWAYS.values()]
    all_lons = [v["lon"] for v in SENSORS.values()] + [v["lon"] for v in GATEWAYS.values()]
    lat0 = sum(all_lats) / len(all_lats)
    lon0 = sum(all_lons) / len(all_lons)

    GW_XY = {g: ll_to_xy(meta["lat"], meta["lon"], lat0, lon0) for g, meta in GATEWAYS.items()}
    S_XY_TRUE = {s: ll_to_xy(meta["lat"], meta["lon"], lat0, lon0) for s, meta in SENSORS.items()}

    out = {
        "SENSORS_LATLON": SENSORS,
        "GATEWAYS_LATLON": GATEWAYS,
        "GW_XY": GW_XY,
        "S_XY_TRUE": S_XY_TRUE,
        "ref": {"lat0": lat0, "lon0": lon0},
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()

