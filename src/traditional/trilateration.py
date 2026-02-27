from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

def multilaterate_ls(gw_names: List[str], dists_est: np.ndarray, GW_XY: Dict[str, Tuple[float,float]]) -> Tuple[float,float]:
    """
    Least-squares multilateration in local XY.
    gw_names: list of gateways to use
    dists_est: array of distances (meters) same order as gw_names
    GW_XY: {gateway: (x,y)} map in meters
    Returns (x_est, y_est)
    """
    xi = np.array([GW_XY[g][0] for g in gw_names], dtype=float)
    yi = np.array([GW_XY[g][1] for g in gw_names], dtype=float)
    di = np.asarray(dists_est, dtype=float)

    A = np.column_stack([np.ones_like(xi), -2.0 * xi, -2.0 * yi])
    b = (di ** 2 - xi ** 2 - yi ** 2).reshape(-1, 1)

    m, *_ = np.linalg.lstsq(A, b, rcond=None)
    x_est = float(m[1, 0])
    y_est = float(m[2, 0])
    return x_est, y_est


def trilaterate_all(
    pair_pred: pd.DataFrame,
    GW_XY: Dict[str, Tuple[float,float]],
    S_XY_TRUE: Dict[str, Tuple[float,float]] | None = None,
    min_gateways: int = 3,
) -> pd.DataFrame:
    """
    pair_pred: per-(sensor,gateway) distances: ['sensor','gateway','d_est_m']
    Returns a DataFrame with per-sensor estimates (and error if S_XY_TRUE provided):
      ['sensor','n_gateways_used','x_est','y_est','x_true','y_true','error_m']
    """
    rows = []
    for s, gdf in pair_pred.groupby('sensor'):
        gdf = gdf.sort_values('gateway')
        if gdf.shape[0] < min_gateways:
            continue
        gw_list = gdf['gateway'].tolist()
        d_est   = gdf['d_est_m'].values
        x_est, y_est = multilaterate_ls(gw_list, d_est, GW_XY)
        x_true = y_true = np.nan
        err_m = np.nan
        if S_XY_TRUE and s in S_XY_TRUE:
            x_true, y_true = S_XY_TRUE[s]
            err_m = float(np.hypot(x_est - x_true, y_est - y_true))
        rows.append({
            'sensor': s,
            'n_gateways_used': len(gw_list),
            'x_est': x_est, 'y_est': y_est,
            'x_true': x_true, 'y_true': y_true,
            'error_m': err_m
        })
    return pd.DataFrame(rows).sort_values('sensor').reset_index(drop=True)

