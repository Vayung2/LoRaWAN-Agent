from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
from joblib import load

from src.helpers import load_and_prepare_packets

def predict_pair_distances(pairs_df: pd.DataFrame, model_path: str = "models/regression_model.joblib") -> pd.DataFrame:
    """
    Input:
      pairs_df: packet-level rows with ['sensor','gateway','rssi_dbm','snr_db','freq_mhz']
    Output:
      per-(sensor,gateway) median distance estimate:
      ['sensor','gateway','d_est_m']
    """
    pipe = load(model_path)
    X = pairs_df[[*pipe.named_steps['pre'].transformers_[0][2], *pipe.named_steps['pre'].transformers_[1][2]]]
    y_pred_log = pipe.predict(X)
    pairs_df = pairs_df.copy()
    pairs_df['d_est_m'] = 10 ** y_pred_log
    agg = (pairs_df.groupby(['sensor','gateway'], as_index=False)
                 .agg(d_est_m=('d_est_m','median')))
    return agg[['sensor','gateway','d_est_m']]

def main():
    ap = argparse.ArgumentParser(description="Predict per-(sensor,gateway) distances with saved regression model.")
    ap.add_argument("--data_dir", default="dataset/lorawan_metadata")
    ap.add_argument("--target_freq_mhz", type=float, default=915.0)
    ap.add_argument("--outlier_db", type=float, default=20.0)
    ap.add_argument("--min_pkts", type=int, default=10)
    ap.add_argument("--model_path", default="models/regression_model.joblib")
    ap.add_argument("--out_csv", default="reports/pair_pred_regression.csv")
    args = ap.parse_args()

    pairs = load_and_prepare_packets(args.data_dir, args.target_freq_mhz, args.outlier_db, args.min_pkts)
    pred = predict_pair_distances(pairs, model_path=args.model_path)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    pred.to_csv(args.out_csv, index=False)
    print(f"[predict] wrote → {args.out_csv}")

if __name__ == "__main__":
    main()

