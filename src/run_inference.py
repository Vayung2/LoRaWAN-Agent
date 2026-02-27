from __future__ import annotations

import argparse
import json
import os
import glob
import numpy as np
import pandas as pd

from src.helpers import (
    load_and_prepare_packets,
    summarize_pairs_to_csv,
)
from src.traditional.pathloss import TraditionalParams, estimate_pair_distances_traditional
from src.traditional.trilateration import trilaterate_all
from src.regression_model.predict import predict_pair_distances

TARGET_FREQ_MHZ_DEFAULT = 915
OUTLIER_DB_DEFAULT = 20.0
MIN_PKTS_DEFAULT = 10


def main():
    ap = argparse.ArgumentParser(description="Run localization inference pipeline (traditional vs ML).")
    ap.add_argument("--data_dir", default="dataset/lorawan_metadata")
    ap.add_argument("--method", choices=["traditional", "regression"], default="traditional")
    ap.add_argument("--metadata", default="models/metadata.json", help="Contains GW_XY and S_XY_TRUE.")
    ap.add_argument("--traditional_params", default="models/traditional_params.json")
    ap.add_argument("--out_csv", default="reports/loc_report.csv")
    ap.add_argument("--target_freq_mhz", type=float, default=TARGET_FREQ_MHZ_DEFAULT)
    ap.add_argument("--outlier_db", type=float, default=OUTLIER_DB_DEFAULT)
    ap.add_argument("--min_pkts", type=int, default=MIN_PKTS_DEFAULT)
    ap.add_argument("--pairs_csv_out", default=None, help="Optional per-(sensor,gateway) summary CSV.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    pairs = load_and_prepare_packets(
        data_dir=args.data_dir,
        target_freq_mhz=args.target_freq_mhz,
        outlier_db=args.outlier_db,
        min_pkts=args.min_pkts,
    )
    if args.pairs_csv_out:
        summarize_pairs_to_csv(pairs, args.pairs_csv_out)
        print(f"[run_inference] wrote pair summary → {args.pairs_csv_out}")

    if args.method == "traditional":
        params = TraditionalParams.load(args.traditional_params)
        pair_pred = estimate_pair_distances_traditional(pairs, params)
    else:
        pair_pred = predict_pair_distances(pairs)

    meta = json.load(open(args.metadata, "r"))
    GW_XY = {k: tuple(v) for k, v in meta["GW_XY"].items()}
    S_XY_TRUE = {k: tuple(v) for k, v in meta["S_XY_TRUE"].items()} if "S_XY_TRUE" in meta else None

    loc_df = trilaterate_all(pair_pred, GW_XY=GW_XY, S_XY_TRUE=S_XY_TRUE, min_gateways=3)
    loc_df.to_csv(args.out_csv, index=False)

    med = float(np.median(loc_df["error_m"])) if "error_m" in loc_df and not loc_df["error_m"].isna().all() else np.nan
    print(f"[run_inference] method={args.method} sensors={loc_df.shape[0]} median_error_m={med:.2f}")
    print(f"[run_inference] wrote → {args.out_csv}")


if __name__ == "__main__":
    main()

