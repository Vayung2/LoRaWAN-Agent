from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any

import numpy as np
import pandas as pd

from src.helpers import load_and_prepare_packets, summarize_pairs_to_csv
from src.traditional.pathloss import TraditionalParams, estimate_pair_distances_traditional
from src.traditional.trilateration import trilaterate_all
from src.regression_model.predict import predict_pair_distances


MethodName = Literal["traditional", "regression"]


@dataclass
class LoRaWANEnvironmentConfig:
    data_dir: str = "dataset/lorawan_metadata"
    metadata_path: str = "models/metadata.json"
    traditional_params_path: str = "models/traditional_params.json"
    target_freq_mhz: float = 915.0
    outlier_db: float = 20.0
    min_pkts: int = 10
    # ---- Attack controls 
    attack_scope: AttackScope = "global"
    attack_sensor: Optional[str] = None
    attack_gateway: Optional[str] = None
    # RSSI perturbations
    rssi_shift_db: float = 0.0              # foil / bias
    rssi_noise_sigma_db: float = 0.0        # random noise
    # packet drop / jamming
    drop_prob: float = 0.0                  # probability to drop targeted packets
    # reproducibility for noise/drop
    seed: int = 0
    rssi_shift_db: float = 0.0
    pairs_csv_cache: Optional[str] = None

class LoRaWANEnvironment:
    """
    Unified environment that owns the dataset view and can execute:
      - traditional path-loss pipeline
      - regression (ML) pipeline

    Later you can plug in an agentic policy that chooses between them using this same API.
    """

    def __init__(self, config: Optional[LoRaWANEnvironmentConfig] = None):
        self.config = config or LoRaWANEnvironmentConfig()
        self._pairs_df: Optional[pd.DataFrame] = None
        self._meta: Optional[Dict[str, Any]] = None

    # --------- core data access ---------
    @property
    def pairs(self) -> pd.DataFrame:
        if self._pairs_df is None:
            c = self.config
            self._pairs_df = load_and_prepare_packets(
                            data_dir=c.data_dir,
                            target_freq_mhz=c.target_freq_mhz,
                            outlier_db=c.outlier_db,
                            min_pkts=c.min_pkts,
                            attack_scope=c.attack_scope,
                            attack_sensor=c.attack_sensor,
                            attack_gateway=c.attack_gateway,
                            rssi_shift_db=c.rssi_shift_db,
                            rssi_noise_sigma_db=c.rssi_noise_sigma_db,
                            drop_prob=c.drop_prob,
                            seed=c.seed,
                        )
            if c.pairs_csv_cache:
                summarize_pairs_to_csv(self._pairs_df, c.pairs_csv_cache)
        return self._pairs_df

    @property
    def meta(self) -> Dict[str, Any]:
        if self._meta is None:
            with open(self.config.metadata_path, "r") as f:
                self._meta = json.load(f)
        return self._meta

    @property
    def GW_XY(self) -> Dict[str, tuple]:
        return {k: tuple(v) for k, v in self.meta["GW_XY"].items()}

    @property
    def S_XY_TRUE(self) -> Optional[Dict[str, tuple]]:
        if "S_XY_TRUE" not in self.meta:
            return None
        return {k: tuple(v) for k, v in self.meta["S_XY_TRUE"].items()}

    # --------- execution paths ---------
    def run_traditional(self) -> pd.DataFrame:
        """Run the traditional path-loss + trilateration pipeline."""
        params = TraditionalParams.load(self.config.traditional_params_path)
        pair_pred = estimate_pair_distances_traditional(self.pairs, params)
        loc_df = trilaterate_all(pair_pred, GW_XY=self.GW_XY, S_XY_TRUE=self.S_XY_TRUE, min_gateways=3)
        return loc_df

    def run_regression(self) -> pd.DataFrame:
        """Run the ML regression + trilateration pipeline."""
        pair_pred = predict_pair_distances(self.pairs)
        loc_df = trilaterate_all(pair_pred, GW_XY=self.GW_XY, S_XY_TRUE=self.S_XY_TRUE, min_gateways=3)
        return loc_df

    def run(self, method: MethodName) -> pd.DataFrame:
        if method == "traditional":
            return self.run_traditional()
        elif method == "regression":
            return self.run_regression()
        else:
            raise ValueError(f"Unknown method: {method}")

    # --------- simple scalar diagnostics ---------
    @staticmethod
    def median_error(loc_df: pd.DataFrame) -> float:
        if "error_m" not in loc_df or loc_df["error_m"].isna().all():
            return float("nan")
        return float(np.median(loc_df["error_m"]))

