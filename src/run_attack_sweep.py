from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd

from src.environment import LoRaWANEnvironment, LoRaWANEnvironmentConfig


def parse_shifts(arg: str) -> List[float]:
    return [float(x.strip()) for x in arg.split(",") if x.strip()]


def parse_methods(arg: str) -> List[str]:
    vals = [x.strip() for x in arg.split(",") if x.strip()]
    out: List[str] = []
    for v in vals:
        if v not in {"traditional", "regression"}:
            raise ValueError(f"Unsupported method in --methods: {v}")
        out.append(v)
    return out


def main():
    ap = argparse.ArgumentParser(description="Run RSSI-shift attack sweep for traditional and regression methods.")
    ap.add_argument("--data_dir", default="dataset/lorawan_metadata")
    ap.add_argument("--metadata", default="models/metadata.json")
    ap.add_argument("--traditional_params", default="models/traditional_params.json")
    ap.add_argument("--target_freq_mhz", type=float, default=915.0)
    ap.add_argument("--outlier_db", type=float, default=20.0)
    ap.add_argument("--min_pkts", type=int, default=10)
    ap.add_argument(
        "--shifts",
        default="-2,-10,-20,-30,-40",
        help="Comma-separated RSSI shifts in dB (e.g. '-2,-10,-20,-30,-40').",
    )
    ap.add_argument(
        "--methods",
        default="traditional,regression",
        help="Comma-separated list of methods to run: traditional,regression.",
    )
    ap.add_argument(
        "--out_dir",
        default="reports/attacks/foil_shift",
        help="Base directory where per-shift outputs and summary.csv are written.",
    )
    args = ap.parse_args()

    shifts = parse_shifts(args.shifts)
    methods = parse_methods(args.methods)

    os.makedirs(args.out_dir, exist_ok=True)
    summary_rows = []

    for shift in shifts:
        shift_dir = os.path.join(args.out_dir, f"shift_{int(shift)}")
        os.makedirs(shift_dir, exist_ok=True)

        pairs_csv = os.path.join(shift_dir, "pairs_summary.csv")
        cfg = LoRaWANEnvironmentConfig(
            data_dir=args.data_dir,
            metadata_path=args.metadata,
            traditional_params_path=args.traditional_params,
            target_freq_mhz=args.target_freq_mhz,
            outlier_db=args.outlier_db,
            min_pkts=args.min_pkts,
            rssi_shift_db=shift,
            pairs_csv_cache=pairs_csv,
        )
        env = LoRaWANEnvironment(cfg)

        for method in methods:
            loc_csv = os.path.join(shift_dir, f"loc_{method}.csv")
            loc_df = env.run(method)  # uses attacked packets via rssi_shift_db
            loc_df.to_csv(loc_csv, index=False)

            med = env.median_error(loc_df)
            summary_rows.append(
                dict(
                    shift_db=float(shift),
                    method=method,
                    sensors=int(loc_df.shape[0]),
                    median_error_m=float(med),
                )
            )
            print(
                f"[attack_sweep] shift_db={shift:.1f} method={method} "
                f"sensors={loc_df.shape[0]} median_error_m={med:.2f} → {loc_csv}"
            )

    if summary_rows:
        summary = pd.DataFrame(summary_rows).sort_values(["shift_db", "method"]).reset_index(drop=True)
        summary_path = os.path.join(args.out_dir, "summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"[attack_sweep] wrote summary → {summary_path}")


if __name__ == "__main__":
    main()

