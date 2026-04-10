from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.helpers import load_and_prepare_packets
from src.traditional.pathloss import TraditionalParams, estimate_pair_distances_traditional
from src.traditional.trilateration import trilaterate_all

from .types import AttackScenario


@dataclass
class ServerConfig:
    data_dir: str = "dataset/lorawan_metadata"
    metadata_path: str = "models/metadata.json"
    traditional_params_path: str = "models/traditional_params.json"
    target_freq_mhz: float = 915.0
    outlier_db: float = 20.0
    min_pkts: int = 10


class GatewayCoordinatorServer:
    """
    Server-side coordinator that exposes read-only tools over gateway observations.

    The ReAct agent uses these tools to investigate whether a synthetic attack
    was injected into the dataset-backed network snapshot.
    """

    def __init__(self, config: ServerConfig | None = None):
        self.config = config or ServerConfig()
        self.params = TraditionalParams.load(self.config.traditional_params_path)
        with open(self.config.metadata_path, "r") as handle:
            self.metadata = json.load(handle)

        self.gateway_xy = {name: tuple(xy) for name, xy in self.metadata["GW_XY"].items()}
        self.sensor_xy_true = {
            name: tuple(xy) for name, xy in self.metadata.get("S_XY_TRUE", {}).items()
        }
        self._baseline_packets = self._load_packets(AttackScenario())
        self._baseline_summary = self._summarize_pairs(self._baseline_packets)

    def _load_packets(self, scenario: AttackScenario) -> pd.DataFrame:
        return load_and_prepare_packets(
            data_dir=self.config.data_dir,
            target_freq_mhz=self.config.target_freq_mhz,
            outlier_db=self.config.outlier_db,
            min_pkts=self.config.min_pkts,
            **scenario.load_kwargs(),
        )

    @staticmethod
    def _summarize_pairs(packets: pd.DataFrame) -> pd.DataFrame:
        if packets.empty:
            return pd.DataFrame(
                columns=[
                    "sensor",
                    "gateway",
                    "packet_count",
                    "rssi_median_dbm",
                    "rssi_mean_dbm",
                    "rssi_std_db",
                    "snr_median_db",
                ]
            )

        summary = (
            packets.groupby(["sensor", "gateway"], as_index=False)
            .agg(
                packet_count=("rssi_dbm", "size"),
                rssi_median_dbm=("rssi_dbm", "median"),
                rssi_mean_dbm=("rssi_dbm", "mean"),
                rssi_std_db=("rssi_dbm", "std"),
                snr_median_db=("snr_db", "median"),
            )
            .fillna({"rssi_std_db": 0.0})
        )
        return summary.sort_values(["sensor", "gateway"]).reset_index(drop=True)

    @staticmethod
    def _pairwise_delta(
        baseline_summary: pd.DataFrame, scenario_summary: pd.DataFrame
    ) -> pd.DataFrame:
        merged = baseline_summary.merge(
            scenario_summary,
            on=["sensor", "gateway"],
            how="left",
            suffixes=("_baseline", "_scenario"),
        )
        fill_from_baseline = [
            "rssi_median_dbm",
            "rssi_mean_dbm",
            "rssi_std_db",
            "snr_median_db",
        ]
        for column in fill_from_baseline:
            merged[f"{column}_scenario"] = merged[f"{column}_scenario"].fillna(
                merged[f"{column}_baseline"]
            )
        merged["packet_count_scenario"] = merged["packet_count_scenario"].fillna(0)
        merged["rssi_shift_db"] = (
            merged["rssi_median_dbm_scenario"] - merged["rssi_median_dbm_baseline"]
        )
        merged["packet_ratio"] = (
            merged["packet_count_scenario"] / merged["packet_count_baseline"].clip(lower=1)
        )
        merged["std_delta_db"] = merged["rssi_std_db_scenario"] - merged["rssi_std_db_baseline"]
        merged["snr_shift_db"] = (
            merged["snr_median_db_scenario"] - merged["snr_median_db_baseline"]
        )
        return merged

    def get_network_snapshot(self, scenario: AttackScenario) -> dict[str, Any]:
        packets = self._load_packets(scenario)
        summary = self._summarize_pairs(packets)
        delta = self._pairwise_delta(self._baseline_summary, summary)
        return {"packets": packets, "summary": summary, "delta": delta}

    def rank_sensors(self, delta: pd.DataFrame) -> pd.DataFrame:
        if delta.empty:
            return pd.DataFrame(columns=["sensor", "anomaly_score"])

        sensor_rank = (
            delta.groupby("sensor", as_index=False)
            .agg(
                mean_abs_rssi_shift=("rssi_shift_db", lambda s: float(np.mean(np.abs(s)))),
                worst_packet_ratio=("packet_ratio", "min"),
                mean_std_delta=("std_delta_db", "mean"),
            )
        )
        sensor_rank["anomaly_score"] = (
            sensor_rank["mean_abs_rssi_shift"] * 1.4
            + (1.0 - sensor_rank["worst_packet_ratio"].clip(lower=0.0, upper=1.0)) * 10.0
            + sensor_rank["mean_std_delta"].clip(lower=0.0) * 1.2
        )
        return sensor_rank.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    def rank_gateways(self, delta: pd.DataFrame) -> pd.DataFrame:
        if delta.empty:
            return pd.DataFrame(columns=["gateway", "anomaly_score"])

        gateway_rank = (
            delta.groupby("gateway", as_index=False)
            .agg(
                mean_abs_rssi_shift=("rssi_shift_db", lambda s: float(np.mean(np.abs(s)))),
                worst_packet_ratio=("packet_ratio", "min"),
                mean_std_delta=("std_delta_db", "mean"),
            )
        )
        gateway_rank["anomaly_score"] = (
            gateway_rank["mean_abs_rssi_shift"] * 1.5
            + (1.0 - gateway_rank["worst_packet_ratio"].clip(lower=0.0, upper=1.0)) * 8.0
            + gateway_rank["mean_std_delta"].clip(lower=0.0)
        )
        return gateway_rank.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    @staticmethod
    def sensor_view(delta: pd.DataFrame, sensor: str) -> pd.DataFrame:
        cols = [
            "sensor",
            "gateway",
            "rssi_shift_db",
            "packet_ratio",
            "std_delta_db",
            "snr_shift_db",
            "packet_count_baseline",
            "packet_count_scenario",
        ]
        return delta.loc[delta["sensor"] == sensor, cols].sort_values("gateway").reset_index(drop=True)

    @staticmethod
    def gateway_view(delta: pd.DataFrame, gateway: str) -> pd.DataFrame:
        cols = [
            "sensor",
            "gateway",
            "rssi_shift_db",
            "packet_ratio",
            "std_delta_db",
            "snr_shift_db",
        ]
        return delta.loc[delta["gateway"] == gateway, cols].sort_values("sensor").reset_index(drop=True)

    def trilateration_view(self, packets: pd.DataFrame, sensor: str) -> dict[str, Any]:
        sensor_packets = packets.loc[packets["sensor"] == sensor].copy()
        if sensor_packets.empty:
            return {
                "sensor": sensor,
                "available": False,
                "reason": "No packets were available for this sensor in the scenario snapshot.",
            }

        pair_pred = estimate_pair_distances_traditional(sensor_packets, self.params)
        loc_df = trilaterate_all(
            pair_pred,
            GW_XY=self.gateway_xy,
            S_XY_TRUE=self.sensor_xy_true,
            min_gateways=3,
        )
        if loc_df.empty:
            return {
                "sensor": sensor,
                "available": False,
                "reason": "Fewer than three gateways survived preprocessing for this sensor.",
            }

        row = loc_df.iloc[0].to_dict()
        x_est = float(row["x_est"])
        y_est = float(row["y_est"])
        residuals = []
        for _, pair in pair_pred.iterrows():
            gx, gy = self.gateway_xy[pair["gateway"]]
            modeled = float(np.hypot(x_est - gx, y_est - gy))
            residuals.append(modeled - float(pair["d_est_m"]))

        row["available"] = True
        row["gateway_distance_estimates_m"] = {
            pair["gateway"]: float(pair["d_est_m"]) for _, pair in pair_pred.iterrows()
        }
        row["residual_rmse_m"] = float(np.sqrt(np.mean(np.square(residuals))))
        row["pair_count"] = int(pair_pred.shape[0])
        return row

    def baseline_trilateration_view(self, sensor: str) -> dict[str, Any]:
        return self.trilateration_view(self._baseline_packets, sensor)

    def ensure_metadata(self) -> None:
        for path in [self.config.metadata_path, self.config.traditional_params_path]:
            if not Path(path).exists():
                raise FileNotFoundError(f"Required model metadata is missing: {path}")
