from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
from joblib import dump

from src.constants import SENSORS, GATEWAYS
from src.helpers import haversine_m, load_and_prepare_packets


def _ohe():
    # sklearn changed OneHotEncoder arg name from sparse -> sparse_output
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _attach_true_distance(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pairs

    # compute per-(sensor,gateway) true distance using lat/lon constants
    key = pairs[["sensor", "gateway"]].drop_duplicates().copy()
    key["true_distance_m"] = np.nan

    d_true: List[float] = []
    for s, g in zip(key["sensor"].astype(str).values, key["gateway"].astype(str).values):
        if s in SENSORS and g in GATEWAYS:
            d_true.append(
                float(
                    haversine_m(
                        SENSORS[s]["lat"],
                        SENSORS[s]["lon"],
                        GATEWAYS[g]["lat"],
                        GATEWAYS[g]["lon"],
                    )
                )
            )
        else:
            d_true.append(np.nan)
    key["true_distance_m"] = d_true

    out = pairs.merge(key, on=["sensor", "gateway"], how="left")
    out = out.dropna(subset=["true_distance_m"]).copy()
    return out


def train_regression_model(
    data_dir: str = "dataset/lorawan_metadata",
    target_freq_mhz: float = 915.0,
    outlier_db: float = 20.0,
    min_pkts: int = 10,
    random_state: int = 42,
    cv_folds: int = 5,
) -> tuple[object, dict]:
    """
    Mirrors the approach in `2_ml_localization.ipynb`:
    - labels: y = log10(true_distance_m) using known sensor/gateway lat/lon
    - features: rssi_dbm, snr_db, freq_mhz (numeric) + gateway (categorical)
    - GroupKFold by sensor
    - model selection: Ridge vs GradientBoostingRegressor by median MAE in meters
    """
    pairs = load_and_prepare_packets(
        data_dir=data_dir,
        target_freq_mhz=target_freq_mhz,
        outlier_db=outlier_db,
        min_pkts=min_pkts,
    )
    pairs = _attach_true_distance(pairs)
    if pairs.empty:
        raise RuntimeError("No training rows after loading + filtering + attaching ground-truth distances.")

    pairs = pairs.copy()
    pairs["y_log10d"] = np.log10(pairs["true_distance_m"].clip(lower=1.0))

    feature_cols_num = ["rssi_dbm", "snr_db", "freq_mhz"]
    feature_cols_cat = ["gateway"]
    X_cols = feature_cols_num + feature_cols_cat

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import GroupKFold

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feature_cols_num),
            ("cat", _ohe(), feature_cols_cat),
        ],
        remainder="drop",
    )

    ridge = Pipeline([("pre", pre), ("reg", Ridge(alpha=1.0, random_state=random_state))])
    gbr = Pipeline([("pre", pre), ("reg", GradientBoostingRegressor(random_state=random_state))])

    y = pairs["y_log10d"].values
    groups = pairs["sensor"].astype(str).values
    n_groups = int(pd.Series(groups).nunique())

    metrics: dict = {"n_rows": int(pairs.shape[0]), "n_sensors": n_groups}

    def cv_mae(model):
        n_splits = min(int(cv_folds), n_groups)
        if n_splits < 2:
            return float("nan"), float("nan")

        cv = GroupKFold(n_splits=n_splits)
        maes = []
        for train_idx, test_idx in cv.split(pairs, y, groups):
            X_train = pairs.iloc[train_idx][X_cols]
            X_test = pairs.iloc[test_idx][X_cols]
            y_train = y[train_idx]
            y_test = y[test_idx]

            model.fit(X_train, y_train)
            y_pred_log = model.predict(X_test)
            d_pred = 10**y_pred_log
            d_true = 10**y_test
            maes.append(mean_absolute_error(d_true, d_pred))

        return float(np.median(maes)), float(np.mean(maes))

    ridge_med, ridge_mean = cv_mae(ridge)
    gbr_med, gbr_mean = cv_mae(gbr)

    metrics.update(
        {
            "ridge_mae_median_m": ridge_med,
            "ridge_mae_mean_m": ridge_mean,
            "gbr_mae_median_m": gbr_med,
            "gbr_mae_mean_m": gbr_mean,
        }
    )

    # same selection rule as notebook (prefer lower median MAE; tie => ridge)
    best = gbr if (np.isfinite(gbr_med) and np.isfinite(ridge_med) and gbr_med < ridge_med) else ridge
    metrics["selected_model"] = "GradientBoostingRegressor" if best is gbr else "Ridge"

    best.fit(pairs[X_cols], y)
    return best, metrics


def main():
    ap = argparse.ArgumentParser(description="Train regression model (ML) for LoRaWAN distance prediction.")
    ap.add_argument("--data_dir", default="dataset/lorawan_metadata")
    ap.add_argument("--target_freq_mhz", type=float, default=915.0)
    ap.add_argument("--outlier_db", type=float, default=20.0)
    ap.add_argument("--min_pkts", type=int, default=10)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--out_model", default="models/regression_model.joblib")
    args = ap.parse_args()

    model, metrics = train_regression_model(
        data_dir=args.data_dir,
        target_freq_mhz=args.target_freq_mhz,
        outlier_db=args.outlier_db,
        min_pkts=args.min_pkts,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
    )

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    dump(model, args.out_model)

    ridge_med = metrics.get("ridge_mae_median_m", float("nan"))
    gbr_med = metrics.get("gbr_mae_median_m", float("nan"))
    print(f"[train] rows={metrics['n_rows']} sensors={metrics['n_sensors']}")
    if np.isfinite(ridge_med) and np.isfinite(gbr_med):
        print(f"[CV] Ridge MAE (m): median={ridge_med:.2f}, mean={metrics['ridge_mae_mean_m']:.2f}")
        print(f"[CV] GBReg MAE (m): median={gbr_med:.2f}, mean={metrics['gbr_mae_mean_m']:.2f}")
    print(f"[train] selected={metrics['selected_model']}")
    print(f"[train] wrote → {args.out_model}")


if __name__ == "__main__":
    main()

