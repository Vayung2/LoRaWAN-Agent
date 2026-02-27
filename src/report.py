from __future__ import annotations
import os, json, math, argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def ecdf(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(arr, float)
    a = a[~np.isnan(a)]
    x = np.sort(a)
    n = len(x)
    F = np.arange(1, n + 1) / n if n else np.array([])
    return x, F

def savefig(path: str):
    ensure_dir(os.path.dirname(path))
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=140)
    plt.close()

def cond_number_A(gw_xy: Dict[str, Tuple[float,float]], gw_list: List[str]) -> float:
    """
    Triangulation design 'quality' proxy: condition number of A = [1, -2xi, -2yi].
    Lower is better. Very high => geometry nearly degenerate.
    """
    xi = np.array([gw_xy[g][0] for g in gw_list], float)
    yi = np.array([gw_xy[g][1] for g in gw_list], float)
    A = np.column_stack([np.ones_like(xi), -2.0*xi, -2.0*yi])
    try:
        u, s, v = np.linalg.svd(A, full_matrices=False)
        cn = (s[0] / s[-1]) if s[-1] != 0 else np.inf
        return float(cn)
    except Exception:
        return np.inf

def qbin(series: pd.Series, q: List[float], labels: Optional[List[str]] = None) -> pd.Series:
    """Quantile bin with stable labels."""
    res = pd.qcut(series, q=q, duplicates='drop')
    if labels and len(labels) == res.cat.categories.size:
        res.cat.rename_categories(labels, inplace=True)
    return res

@dataclass
class ReportInputs:
    loc_csv: str
    pairs_csv: Optional[str]
    meta_json: Optional[str]
    out_dir: str
    out_html: str
    title: str
    method_name: str
    compare_loc_csv: Optional[str] = None

def load_data(inputs: ReportInputs):
    loc = pd.read_csv(inputs.loc_csv)
    pairs = pd.read_csv(inputs.pairs_csv) if inputs.pairs_csv and os.path.exists(inputs.pairs_csv) else None
    meta = json.load(open(inputs.meta_json, "r")) if inputs.meta_json and os.path.exists(inputs.meta_json) else {}
    return loc, pairs, meta

def enrich_loc_with_pairs(loc: pd.DataFrame, pairs: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = loc.copy()
    if pairs is None:
        return df

    if "n_pkts" in pairs.columns:
        k = (pairs.groupby("sensor")["n_pkts"].sum().rename("sensor_total_pkts"))
        df = df.merge(k, on="sensor", how="left")

    if "snr_db_median" in pairs.columns:
        snr = pairs.drop_duplicates("sensor")[["sensor","snr_db_median"]].rename(
            columns={"snr_db_median":"snr_sensor_median_db"})
        df = df.merge(snr, on="sensor", how="left")

    if "sensor_env" in pairs.columns and "sensor_env" not in df.columns:
        env = pairs.drop_duplicates("sensor")[["sensor","sensor_env"]]
        df = df.merge(env, on="sensor", how="left")

    if "sensor_sf" in pairs.columns and "sensor_sf" not in df.columns:
        sf = pairs.drop_duplicates("sensor")[["sensor","sensor_sf"]]
        df = df.merge(sf, on="sensor", how="left")

    if "sensor_bw_hz" in pairs.columns and "sensor_bw_hz" not in df.columns:
        bw = pairs.drop_duplicates("sensor")[["sensor","sensor_bw_hz"]]
        df = df.merge(bw, on="sensor", how="left")

    return df

def plot_cdf(series, label, out_png, title="Error CDF", xlabel="Localization error (m)"):
    x, F = ecdf(np.asarray(series))
    plt.figure(figsize=(7,5))
    plt.plot(x, F, label=label)
    plt.xlabel(xlabel); plt.ylabel("Cumulative probability")
    plt.title(title); plt.grid(True, linestyle=":")
    if label: plt.legend()
    savefig(out_png)

def plot_cdf_multi(named_arrays: Dict[str, np.ndarray], out_png, title="Error CDFs", xlabel="Localization error (m)"):
    plt.figure(figsize=(7,5))
    for lab, arr in named_arrays.items():
        x, F = ecdf(np.asarray(arr))
        if len(x): plt.plot(x, F, label=lab)
    plt.xlabel(xlabel); plt.ylabel("Cumulative probability")
    plt.title(title); plt.grid(True, linestyle=":"); plt.legend()
    savefig(out_png)

def scatter_xy(df, xcol, ycol, out_png, title, xlabel, ylabel, annotate_quantiles=False):
    plt.figure(figsize=(6.8,5.2))
    plt.scatter(df[xcol], df[ycol], s=18, alpha=0.7)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.grid(True, linestyle=":")
    if annotate_quantiles:
        try:
            qx = np.percentile(df[xcol].dropna(), [25,50,75])
            qy = np.percentile(df[ycol].dropna(), [25,50,75])
            for val in qx: plt.axvline(val, color='k', lw=0.8, ls='--', alpha=0.4)
            for val in qy: plt.axhline(val, color='k', lw=0.8, ls='--', alpha=0.4)
        except Exception:
            pass
    savefig(out_png)

def barh_series(s: pd.Series, out_png, title, xlabel):
    s = s.sort_values()
    plt.figure(figsize=(7, max(3, 0.25*len(s))))
    plt.barh(s.index.astype(str), s.values)
    plt.title(title); plt.xlabel(xlabel); plt.grid(True, axis='x', linestyle=":")
    savefig(out_png)

def cdf_by_spreading_factor(loc: pd.DataFrame, out_png: str):
    if "sensor_sf" not in loc.columns:
        return False
    sub = loc.dropna(subset=["sensor_sf", "error_m"]).copy()
    if sub.empty:
        return False
    try:
        sub["sensor_sf"] = sub["sensor_sf"].astype(int)
        order = sorted(sub["sensor_sf"].unique())
        groups = {f"SF={sf}": sub.loc[sub["sensor_sf"]==sf, "error_m"].values for sf in order}
    except Exception:
        order = sorted(sub["sensor_sf"].astype(str).unique())
        groups = {f"SF={sf}": sub.loc[sub["sensor_sf"].astype(str)==sf, "error_m"].values for sf in order}
    plot_cdf_multi(groups, out_png, title="Error CDF by Spreading Factor")
    return True

def cdf_by_bandwidth(loc: pd.DataFrame, out_png: str):
    if "sensor_bw_hz" not in loc.columns:
        return False
    sub = loc.dropna(subset=["sensor_bw_hz","error_m"]).copy()
    if sub.empty:
        return False
    labs = {}
    for bw in sorted(sub["sensor_bw_hz"].unique()):
        try:
            k = int(round(float(bw)/1000.0))
            labs[bw] = f"BW={k} kHz"
        except Exception:
            labs[bw] = f"BW={bw}"
    groups = {labs[bw]: sub.loc[sub["sensor_bw_hz"]==bw, "error_m"].values for bw in labs}
    plot_cdf_multi(groups, out_png, title="Error CDF by Bandwidth")
    return True

def boxplot_by_packet_buckets(loc: pd.DataFrame, out_png: str, n_buckets: int = 4):
    if "sensor_total_pkts" not in loc.columns:
        return False

    sub = loc.dropna(subset=["sensor_total_pkts", "error_m"]).copy()
    if sub.empty:
        return False

    try:
        sub["pkt_bin"] = pd.qcut(sub["sensor_total_pkts"], q=n_buckets, duplicates="drop")
    except ValueError:
        return False

    cats = list(sub["pkt_bin"].cat.categories)
    if not cats:
        return False

    data = []
    labels = []
    for c in cats:
        arr = sub.loc[sub["pkt_bin"] == c, "error_m"].astype(float).values
        if arr.size == 0:
            continue
        data.append(arr)
        labels.append(str(c))

    if len(data) == 0:
        return False

    plt.figure(figsize=(7.2, 5.2))

    positions = []
    for i, arr in enumerate(data, start=1):
        plt.boxplot(arr, positions=[i], widths=0.6, showfliers=True)
        positions.append(i)

    plt.xticks(positions, labels, rotation=0)
    plt.title("Localization error vs. packet volume (quantile bins)")
    plt.xlabel("Sensor total packets (quantile bins)")
    plt.ylabel("Localization error (m)")
    plt.grid(True, axis='y', linestyle=":")
    savefig(out_png)
    return True

def summarize_errors(vals: np.ndarray) -> Dict[str, float]:
    a = np.asarray(vals, float)
    a = a[~np.isnan(a)]
    if not len(a):
        return dict(count=0, median_m=np.nan, p80_m=np.nan, p90_m=np.nan, p95_m=np.nan)
    qs = np.percentile(a, [50,80,90,95])
    return dict(count=len(a), median_m=qs[0], p80_m=qs[1], p90_m=qs[2], p95_m=qs[3])

def compute_geometry_diagnostics(pair_pred: pd.DataFrame, GW_XY: Dict[str, Tuple[float,float]]) -> pd.DataFrame:
    rows = []
    for s, gdf in pair_pred.groupby("sensor"):
        gws = gdf["gateway"].tolist()
        cn = cond_number_A(GW_XY, gws) if GW_XY else np.nan
        rows.append(dict(sensor=s, cond_number=cn, n_gateways=len(gws)))
    return pd.DataFrame(rows)

def method_comparison_table(loc_a: pd.DataFrame, loc_b: pd.DataFrame, name_a: str, name_b: str) -> pd.DataFrame:
    j = loc_a[['sensor','error_m']].merge(loc_b[['sensor','error_m']], on='sensor', suffixes=(f'_{name_a}', f'_{name_b}'))
    j['delta_m'] = j[f'error_m_{name_b}'] - j[f'error_m_{name_a}']
    return j

def html_wrap(title: str, sections: List[str]) -> str:
    head = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; color: #111; }}
h1,h2,h3 {{ margin-top: 1.2em; }}
hr {{ border: none; border-top: 1px solid #ddd; margin: 24px 0; }}
.figure {{ margin: 12px 0 20px; }}
.caption {{ color:#444; font-size: 0.95em; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 0.95em; }}
th {{ background: #fafafa; text-align: left; }}
code {{ background: #f5f5f7; padding: 1px 4px; border-radius: 4px; }}
.small {{ font-size: 0.9em; color:#555; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="small">Generated by <code>src/report.py</code>.</p>
"""
    tail = "</body></html>"
    return head + "\n".join(sections) + tail

def img_block(src: str, caption: str) -> str:
    return f"""
<div class="figure">
  <img src="{src}" alt="{caption}" style="max-width: 900px; width: 100%; height: auto; border:1px solid #eee"/>
  <div class="caption">{caption}</div>
</div>
"""

def build_report(inputs: ReportInputs):
    ensure_dir(inputs.out_dir)
    loc, pairs, meta = load_data(inputs)
    loc = enrich_loc_with_pairs(loc, pairs)

    key = summarize_errors(loc['error_m'].values if 'error_m' in loc.columns else np.array([]))
    sec_intro = f"""
<h2>Overview</h2>
<ul>
  <li><b>Method:</b> {inputs.method_name}</li>
  <li><b># Sensors localized:</b> {loc.shape[0]}</li>
  <li><b>Median error (m):</b> {key['median_m']:.2f} &nbsp; | &nbsp; P80={key['p80_m']:.1f}, P90={key['p90_m']:.1f}, P95={key['p95_m']:.1f}</li>
</ul>
"""
    cdf_all_png = os.path.join(inputs.out_dir, "cdf_overall.png")
    plot_cdf(loc['error_m'], "All sensors", cdf_all_png, title="Localization Error CDF")
    sec_cdf = "<h2>Error CDFs</h2>" + img_block(os.path.relpath(cdf_all_png, os.path.dirname(inputs.out_html)), "Overall error CDF")

    if "sensor_env" in loc.columns:
        env_groups = {k: v["error_m"].values for k, v in loc.groupby("sensor_env")}
        env_cdf_png = os.path.join(inputs.out_dir, "cdf_by_env.png")
        plot_cdf_multi(env_groups, env_cdf_png, title="Error CDF by sensor environment")
        sec_cdf += img_block(os.path.relpath(env_cdf_png, os.path.dirname(inputs.out_html)), "CDF by sensor environment")

    if "n_gateways_used" in loc.columns:
        bins = {}
        if (loc["n_gateways_used"]==3).any(): bins["=3 gateways"] = loc.query("n_gateways_used==3")["error_m"].values
        if (loc["n_gateways_used"]>=4).any(): bins["≥4 gateways"] = loc.query("n_gateways_used>=4")["error_m"].values
        if len(bins) >= 1:
            gw_cdf_png = os.path.join(inputs.out_dir, "cdf_by_ngw.png")
            plot_cdf_multi(bins, gw_cdf_png, title="Error CDF by # of gateways used")
            sec_cdf += img_block(os.path.relpath(gw_cdf_png, os.path.dirname(inputs.out_html)), "CDF by # of gateways used")

    if "snr_sensor_median_db" in loc.columns:
        snr_med = float(loc["snr_sensor_median_db"].median())
        data = {
            f"Low SNR (≤{snr_med:.1f} dB)":  loc.query("snr_sensor_median_db <= @snr_med")["error_m"].values,
            f"High SNR (>{snr_med:.1f} dB)": loc.query("snr_sensor_median_db >  @snr_med")["error_m"].values,
        }
        snr_cdf_png = os.path.join(inputs.out_dir, "cdf_by_snr.png")
        plot_cdf_multi(data, snr_cdf_png, title="Error CDF by sensor SNR")
        sec_cdf += img_block(os.path.relpath(snr_cdf_png, os.path.dirname(inputs.out_html)), "CDF by SNR")
    
    sf_png = os.path.join(inputs.out_dir, "cdf_by_sf.png")
    if cdf_by_spreading_factor(loc, sf_png):
        sec_cdf += img_block(os.path.relpath(sf_png, os.path.dirname(inputs.out_html)),
                             "CDF by spreading factor (dominant SF per sensor)")

    bw_png = os.path.join(inputs.out_dir, "cdf_by_bw.png")
    if cdf_by_bandwidth(loc, bw_png):
        sec_cdf += img_block(
            os.path.relpath(bw_png, os.path.dirname(inputs.out_html)),
            "CDF by bandwidth (dominant BW per sensor)"
        )
        
    sections = [sec_intro, sec_cdf, "<h2>Diagnostics</h2>"]

    if "n_gateways_used" in loc.columns:
        scat_png = os.path.join(inputs.out_dir, "scatter_err_vs_ngw.png")
        scatter_xy(loc, "n_gateways_used", "error_m", scat_png,
                   "Error vs number of gateways", "# gateways used", "Localization error (m)", annotate_quantiles=True)
        sections.append(img_block(os.path.relpath(scat_png, os.path.dirname(inputs.out_html)), "Error vs # gateways"))

    if "sensor_total_pkts" in loc.columns:
        scat_png = os.path.join(inputs.out_dir, "scatter_err_vs_pkts.png")
        scatter_xy(loc, "sensor_total_pkts", "error_m", scat_png,
                   "Error vs total packets per sensor", "Total packets per sensor", "Localization error (m)", annotate_quantiles=True)
        sections.append(img_block(os.path.relpath(scat_png, os.path.dirname(inputs.out_html)), "Error vs packet volume"))

    if "snr_sensor_median_db" in loc.columns:
        scat_png = os.path.join(inputs.out_dir, "scatter_err_vs_snr.png")
        scatter_xy(loc, "snr_sensor_median_db", "error_m", scat_png,
                   "Error vs median SNR", "Median SNR (dB)", "Localization error (m)", annotate_quantiles=True)
        sections.append(img_block(os.path.relpath(scat_png, os.path.dirname(inputs.out_html)), "Error vs SNR"))

    pkt_box_png = os.path.join(inputs.out_dir, "box_err_by_packet_bins.png")
    if boxplot_by_packet_buckets(loc, pkt_box_png, n_buckets=4):
        sections.append(img_block(os.path.relpath(pkt_box_png, os.path.dirname(inputs.out_html)),
                                  "Error distribution by packet-volume quartiles"))

    geom_note = ""
    if inputs.pairs_csv and os.path.exists(inputs.pairs_csv) and "GW_XY" in meta:
        gws_per_sensor = (pd.read_csv(inputs.pairs_csv)
                            .groupby(["sensor","gateway"]).size().reset_index()[["sensor","gateway"]])
        gw_map = meta["GW_XY"]
        rows = []
        for s, gdf in gws_per_sensor.groupby("sensor"):
            gws = gdf["gateway"].tolist()
            if len(gws) >= 3:
                cn = cond_number_A(gw_map, gws)
                rows.append((s, cn, len(gws)))
        if rows:
            geom = pd.DataFrame(rows, columns=["sensor","cond_number","n_gateways_observed"])
            loc = loc.merge(geom, on="sensor", how="left")
            cn_png = os.path.join(inputs.out_dir, "scatter_err_vs_cond.png")
            scatter_xy(loc.dropna(subset=["cond_number"]), "cond_number", "error_m", cn_png,
                       "Error vs geometry condition number", "Condition number (A)", "Localization error (m)", annotate_quantiles=True)
            sections.append(img_block(os.path.relpath(cn_png, os.path.dirname(inputs.out_html)), "Geometry: condition number vs error"))
            geom_note = "<p class='small'>Lower condition number indicates better geometry.</p>"

    sections.append("<h2>Tabular summaries</h2>")
    worst = loc[['sensor','error_m']].sort_values('error_m', ascending=False).head(10)
    sections.append("<h3>Descending order of sensor errors</h3>" + worst.to_html(index=False))

    def block_summary(title, mask=None):
        vals = loc['error_m'][mask].values if mask is not None else loc['error_m'].values
        s = summarize_errors(vals)
        tbl = pd.DataFrame([s])
        return f"<h3>{title}</h3>" + tbl.to_html(index=False)

    sections.append(block_summary("Overall summary"))
    if "sensor_env" in loc.columns:
        for env, g in loc.groupby("sensor_env"):
            sections.append(block_summary(f"Summary: env = {env}", loc["sensor_env"]==env))
    if "n_gateways_used" in loc.columns:
        sections.append(block_summary("Summary: =3 gateways", loc["n_gateways_used"]==3))
        sections.append(block_summary("Summary: ≥4 gateways", loc["n_gateways_used"]>=4))
    if "sensor_total_pkts" in loc.columns:
        medk = int(loc["sensor_total_pkts"].median())
        sections.append(block_summary(f"Summary: Low packets (≤{medk})", loc["sensor_total_pkts"]<=medk))
        sections.append(block_summary(f"Summary: High packets (>{medk})", loc["sensor_total_pkts"]>medk))
    if "snr_sensor_median_db" in loc.columns:
        snr_med = float(loc["snr_sensor_median_db"].median())
        sections.append(block_summary(f"Summary: Low SNR (≤{snr_med:.1f})", loc["snr_sensor_median_db"]<=snr_med))
        sections.append(block_summary(f"Summary: High SNR (>{snr_med:.1f})", loc["snr_sensor_median_db"]>snr_med))

    if inputs.compare_loc_csv and os.path.exists(inputs.compare_loc_csv):
        sec_cmp = "<h2>Method comparison</h2>"
        other = pd.read_csv(inputs.compare_loc_csv)
        common = method_comparison_table(loc, other, inputs.method_name, "other")
        delta_summary = {
            "median_delta_m": float(np.nanmedian(common["delta_m"].values)),
            "p80_delta_m": float(np.nanpercentile(common["delta_m"].values, 80)),
        }
        sec_cmp += "<p>Δ = other − current.</p>"
        sec_cmp += pd.DataFrame([delta_summary]).to_html(index=False)
        d_png = os.path.join(inputs.out_dir, "cdf_method_delta.png")
        plot_cdf(common["delta_m"].values, "Δ error (other - current)", d_png,
                 title="CDF of error difference (other − current)", xlabel="Δ meters")
        sec_cmp += img_block(os.path.relpath(d_png, os.path.dirname(inputs.out_html)), "Method delta CDF")
        sec_cmp += "<h3>Per-sensor deltas (first 30)</h3>" + common.head(30).to_html(index=False)
        sections.append(sec_cmp)

    if geom_note:
        sections.append(geom_note)

    html = html_wrap(inputs.title, sections)
    ensure_dir(os.path.dirname(inputs.out_html))
    with open(inputs.out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[report] wrote → {inputs.out_html}")

def main():
    ap = argparse.ArgumentParser(description="Generate a rich HTML analysis report from localization outputs.")
    ap.add_argument("--loc_csv", required=True, help="Per-sensor localization CSV (from run_inference).")
    ap.add_argument("--pairs_csv", default=None, help="Optional per-(sensor,gateway) CSV with n_pkts, snr_db, sensor_env.")
    ap.add_argument("--meta_json", default="models/metadata.json", help="Optional metadata with GW_XY for geometry diagnostics.")
    ap.add_argument("--out_dir", default="reports/analysis_assets")
    ap.add_argument("--out_html", default="reports/analysis.html")
    ap.add_argument("--title", default="LoRaWAN Localization Analysis Report")
    ap.add_argument("--method_name", default="traditional")
    ap.add_argument("--compare_loc_csv", default=None, help="Optional other method loc CSV for comparison.")
    args = ap.parse_args()

    inputs = ReportInputs(
        loc_csv=args.loc_csv,
        pairs_csv=args.pairs_csv,
        meta_json=args.meta_json,
        out_dir=args.out_dir,
        out_html=args.out_html,
        title=args.title,
        method_name=args.method_name,
        compare_loc_csv=args.compare_loc_csv,
    )
    build_report(inputs)

if __name__ == "__main__":
    main()

