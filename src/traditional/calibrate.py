from __future__ import annotations
import os, json, glob
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from .pathloss import TraditionalParams, model_rssi

from src.helpers import load_pair_parquet, drop_rssi_outliers, haversine_m, frequency_filter

def build_refined_table(
    data_dir: str,
    sensors_latlon: Dict[str, Dict[str, float]],
    gateways_latlon: Dict[str, Dict[str, float]],
    outlier_db: float = 20.0,
    min_pkts: int = 10,
    target_freq_mhz: float = 915.0
) -> pd.DataFrame:
    """
    Returns per-packet trimmed data aggregated per (sensor,gateway) with linear-avg RSSI:
      ['sensor','gateway','n_pkts','rssi_refined_dbm','snr_mean_db','true_distance_m','sensor_env','freq_mhz']
    """
    paths = sorted(glob.glob(os.path.join(data_dir, "sensor*_gateway*.parquet")))
    if not paths:
        raise FileNotFoundError(f"No pair parquet files found under {data_dir}")

    rows = []
    for p in paths:
        df = load_pair_parquet(p)
        if df.empty: 
            continue
        df = frequency_filter(df, target_freq_mhz)
        if df.empty:
            continue

        for (sensor, gateway), g in df.groupby(['sensor','gateway'], as_index=False):
            if g.empty:
                continue

            g_f = drop_rssi_outliers(g, outlier_db)
            if g_f.shape[0] < min_pkts:
                continue

            # linear-domain RSSI averaging
            p_mw = (10.0 ** (g_f['rssi_dbm'] / 10.0)).mean()
            rssi_refined_dbm = 10.0 * np.log10(p_mw)
            n_pkts = g_f.shape[0]
            snr_mean_db = g_f['snr_db'].mean() if 'snr_db' in g_f.columns else np.nan

            if sensor not in sensors_latlon or gateway not in gateways_latlon:
                continue
            true_d = haversine_m(
                sensors_latlon[sensor]['lat'], sensors_latlon[sensor]['lon'],
                gateways_latlon[gateway]['lat'], gateways_latlon[gateway]['lon']
            )

            rows.append({
                'sensor': sensor,
                'gateway': gateway,
                'n_pkts': n_pkts,
                'rssi_refined_dbm': rssi_refined_dbm,
                'snr_mean_db': snr_mean_db,
                'true_distance_m': true_d,
                'sensor_env': sensors_latlon[sensor].get('env', None),
                'freq_mhz': target_freq_mhz
            })

    refined = pd.DataFrame(rows).sort_values(['sensor','gateway']).reset_index(drop=True)
    refined = refined.dropna(subset=['true_distance_m','rssi_refined_dbm'])
    refined = refined[refined['true_distance_m'] > 0].copy()
    return refined


def fit_gamma_and_Kg(
    refined: pd.DataFrame,
    f_hz: float,
    Pt_dbm: float,
    d0_m: float = 1.0,
    gamma_subset: str = "outdoor"  # fit gamma using outdoor sensors (recommended)
) -> TraditionalParams:
    """
    Fit path-loss exponent gamma (with K fixed to Friis) and per-gateway bias K_g.
    """
    c = 3e8
    lambda_m = c / f_hz
    C_dB = 20.0 * np.log10(lambda_m / (4.0 * np.pi * d0_m))
    K_dbm = Pt_dbm - C_dB  # Friis term

    # Fit gamma on a clean subset (e.g., outdoor sensors & certain gateways)
    if gamma_subset and 'sensor_env' in refined.columns:
        cal = refined[refined['sensor_env'] == gamma_subset].copy()
    else:
        cal = refined.copy()

    X = np.log10((cal['true_distance_m'].values / d0_m).clip(min=1e-3)).reshape(-1, 1)
    Y = (K_dbm - cal['rssi_refined_dbm'].values).reshape(-1, 1)
    # y = 10*gamma*x  => gamma = lstsq(y / 10, x)
    gamma_est, *_ = np.linalg.lstsq(10.0 * X, Y, rcond=None)
    gamma = float(gamma_est[0, 0])

    # Compute residuals on all pairs to get per-gateway ΔK_g
    refined = refined.copy()
    refined['rssi_model_base'] = model_rssi(refined['true_distance_m'].values, K_dbm, gamma, d0_m)
    refined['residual'] = refined['rssi_refined_dbm'] - refined['rssi_model_base']
    deltaK = refined.groupby('gateway')['residual'].mean().to_dict()
    K_g = {gw: float(K_dbm + deltaK.get(gw, 0.0)) for gw in refined['gateway'].unique()}

    return TraditionalParams(gamma=gamma, K_g=K_g, f_hz=float(f_hz), Pt_dbm=float(Pt_dbm), d0_m=float(d0_m))


def _load_metadata_latlon(meta_path: str) -> Tuple[Dict, Dict]:
    """Expect models/metadata.json with {'GW_XY':..., 'S_XY_TRUE':..., 'ref': {...}} AND lat/lon inlined OR provide separate files.
    If your metadata.json only has XY, load SENSORS/GATEWAYS from a python module instead.
    """
    with open(meta_path, "r") as f:
        meta = json.load(f)
    # In your earlier step you likely stored XY only; if lat/lon are not present here,
    # import them from a constants module. For now we try to read 'SENSORS_LATLON'/'GATEWAYS_LATLON'.
    sensors = meta.get("SENSORS_LATLON")
    gateways = meta.get("GATEWAYS_LATLON")
    if sensors is None or gateways is None:
        raise ValueError(
            "metadata.json lacks lat/lon. Add SENSORS_LATLON and GATEWAYS_LATLON or pass dicts directly to build_refined_table()."
        )
    return sensors, gateways


def main():
    import argparse, sys
    ap = argparse.ArgumentParser(description="Calibrate traditional model (gamma & per-gateway K_g).")
    ap.add_argument("--data_dir", default="dataset/lorawan_metadata")
    ap.add_argument("--meta", default="models/metadata.json",
                    help="metadata.json containing SENSORS_LATLON and GATEWAYS_LATLON (or pass --sensors_json/--gateways_json).")
    ap.add_argument("--out", default="models/traditional_params.json")
    ap.add_argument("--freq_mhz", type=float, default=915.0)
    ap.add_argument("--ptx_dbm", type=float, default=14.0)
    ap.add_argument("--min_pkts", type=int, default=10)
    ap.add_argument("--outlier_db", type=float, default=20.0)
    ap.add_argument("--gamma_subset", default="outdoor", choices=["outdoor","indoor","",None],
                    help="subset used to fit gamma; '' disables subsetting.")
    args = ap.parse_args()

    # Load lat/lon for true distance calc
    try:
        sensors_ll, gateways_ll = _load_metadata_latlon(args.meta)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    refined = build_refined_table(
        data_dir=args.data_dir,
        sensors_latlon=sensors_ll,
        gateways_latlon=gateways_ll,
        outlier_db=args.outlier_db,
        min_pkts=args.min_pkts,
        target_freq_mhz=args.freq_mhz,
    )

    params = fit_gamma_and_Kg(
        refined=refined,
        f_hz=args.freq_mhz * 1e6,
        Pt_dbm=args.ptx_dbm,
        d0_m=1.0,
        gamma_subset=(args.gamma_subset or None)
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    params.save(args.out)

    print(f"Fitted gamma = {params.gamma:.3f}")
    for g in sorted(params.K_g):
        print(f"K_{g} = {params.K_g[g]:.2f} dB")
    print(f"Saved params → {args.out}")

if __name__ == "__main__":
    main()

