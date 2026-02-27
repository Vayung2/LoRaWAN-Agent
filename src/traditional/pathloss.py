from __future__ import annotations
import json
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

D0_M = 1.0  # reference distance (m)

@dataclass
class TraditionalParams:
    """Calibrated parameters for the traditional RSSI→distance model."""
    gamma: float                 # path-loss exponent
    K_g: dict                    # per-gateway K (dB), e.g. {"gatewayA": 74.1, ...}
    f_hz: float                  # carrier frequency (Hz), e.g. 915e6
    Pt_dbm: float                # TX power (dBm)
    d0_m: float = D0_M           # reference distance (m)

    @staticmethod
    def load(path: str) -> "TraditionalParams":
        with open(path, "r") as f:
            obj = json.load(f)
        return TraditionalParams(**obj)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


def model_rssi(d_m: np.ndarray | float, KdB: float, gamma: float, d0_m: float = D0_M) -> np.ndarray:
    """RSSI(d) = K - 10*gamma*log10(d/d0).  Returns dBm."""
    d = np.asarray(d_m, dtype=float)
    return KdB - 10.0 * gamma * np.log10(d / d0_m)


def rssi_to_distance(rssi_dbm: np.ndarray | float, KdB: float, gamma: float, d0_m: float = D0_M) -> np.ndarray:
    """Invert model: d = d0 * 10^((K - RSSI)/(10*gamma)).  Returns meters."""
    r = np.asarray(rssi_dbm, dtype=float)
    return d0_m * (10.0 ** ((KdB - r) / (10.0 * gamma)))


def estimate_pair_distances_traditional(
    pairs_df: pd.DataFrame,
    params: TraditionalParams,
    use_linear_avg: bool = True,
) -> pd.DataFrame:
    """
    Input: packet-level rows with columns:
      ['sensor','gateway','rssi_dbm','snr_db','freq_mhz'] (others ignored)
      (outliers already trimmed)
    Output: per-(sensor,gateway) median distance estimate:
      columns ['sensor','gateway','d_est_m']
    """
    df = pairs_df.copy()
    if df.empty:
        return pd.DataFrame(columns=["sensor", "gateway", "d_est_m"])

    # Per-paper, linear-domain averaging of RSSI is preferred
    if use_linear_avg:
        # linear mean per (sensor,gateway), then back to dBm
        rssi_refined = (
            df.groupby(["sensor", "gateway"])["rssi_dbm"]
              .apply(lambda s: 10.0 * np.log10((10.0 ** (s / 10.0)).mean()))
              .reset_index(name="rssi_refined_dbm")
        )
    else:
        # fallback: median in dB domain
        rssi_refined = (
            df.groupby(["sensor", "gateway"])["rssi_dbm"]
              .median()
              .reset_index(name="rssi_refined_dbm")
        )

    def row_to_d(r):
        Kg = params.K_g.get(r['gateway'])
        if Kg is None:
            raise KeyError(f"Missing K_g for gateway {r['gateway']}")
        return float(rssi_to_distance(r['rssi_refined_dbm'], Kg, params.gamma, params.d0_m))

    rssi_refined['d_est_m'] = rssi_refined.apply(row_to_d, axis=1)
    return rssi_refined[['sensor','gateway','d_est_m']]

