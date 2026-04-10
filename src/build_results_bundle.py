from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.react_agent import (
    AttackScenario,
    GatewayCoordinatorServer,
    OllamaAdjudicator,
    OllamaConfig,
    ReActGatewayAgent,
)


RESULTS_DIR = Path("outputs/results_bundle")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a presentation-ready results bundle with charts and case studies."
    )
    parser.add_argument("--llm-model", default="qwen2.5:7b")
    parser.add_argument(
        "--ollama-endpoint",
        default="http://127.0.0.1:11434/api/generate",
    )
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    return parser


def build_scenarios(server: GatewayCoordinatorServer) -> list[dict]:
    sensors = ["sensor01", "sensor05", "sensor08"]
    noise_sensors = ["sensor05", "sensor07", "sensor09"]
    gateways = sorted(server.metadata["GATEWAYS_LATLON"].keys())
    scenarios: list[dict] = [
        {
            "scenario_name": "clean_baseline",
            "attack_family": "clean",
            "severity": 0.0,
            "expected_label": "none",
            "scenario": AttackScenario(attack_type="none"),
        }
    ]
    for shift in [-2, -4, -6, -8, -10, -12]:
        for sensor in sensors:
            scenarios.append(
                {
                    "scenario_name": f"sensor_foil_{sensor}_{abs(shift)}db",
                    "attack_family": "sensor_foil",
                    "severity": abs(float(shift)),
                    "expected_label": "sensor_foil" if abs(shift) >= 6 else "sensor_foil",
                    "scenario": AttackScenario(
                        attack_type="sensor_foil",
                        sensor=sensor,
                        rssi_shift_db=float(shift),
                    ),
                }
            )
    for shift in [-2, -4, -6, -8, -10, -12]:
        for gateway in gateways:
            scenarios.append(
                {
                    "scenario_name": f"gateway_bias_{gateway}_{abs(shift)}db",
                    "attack_family": "gateway_bias",
                    "severity": abs(float(shift)),
                    "expected_label": "gateway_bias" if abs(shift) >= 6 else "gateway_bias",
                    "scenario": AttackScenario(
                        attack_type="gateway_bias",
                        gateway=gateway,
                        rssi_shift_db=float(shift),
                    ),
                }
            )
    for sigma in [0.5, 1, 1.5, 2, 3, 4, 5, 6]:
        for sensor in noise_sensors:
            scenarios.append(
                {
                    "scenario_name": f"random_noise_{sensor}_{sigma:.1f}",
                    "attack_family": "random_noise",
                    "severity": float(sigma),
                    "expected_label": "none" if sigma <= 1.5 else "random_noise",
                    "scenario": AttackScenario(
                        attack_type="random_noise",
                        sensor=sensor,
                        noise_sigma_db=float(sigma),
                        seed=7,
                    ),
                }
            )
    for drop_prob in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        for gateway in gateways:
            scenarios.append(
                {
                    "scenario_name": f"packet_drop_{gateway}_{drop_prob:.1f}",
                    "attack_family": "packet_drop",
                    "severity": float(drop_prob),
                    "expected_label": "none" if drop_prob <= 0.2 else "packet_drop",
                    "scenario": AttackScenario(
                        attack_type="packet_drop",
                        gateway=gateway,
                        drop_prob=float(drop_prob),
                        seed=7,
                    ),
                }
            )
    return scenarios


def evaluate_mode(
    *,
    server: GatewayCoordinatorServer,
    scenarios: list[dict],
    mode: str,
    llm_model: str,
    ollama_endpoint: str,
) -> pd.DataFrame:
    llm_client = None
    if mode != "off":
        llm_client = OllamaAdjudicator(
            OllamaConfig(model=llm_model, endpoint=ollama_endpoint, timeout_s=45.0)
        )
    agent = ReActGatewayAgent(server, llm_client=llm_client, llm_mode=mode)
    rows: list[dict] = []

    for item in scenarios:
        report = agent.investigate(item["scenario"]).to_dict()
        row = {
            "mode": mode,
            "scenario_name": item["scenario_name"],
            "attack_family": item["attack_family"],
            "severity": item["severity"],
            "expected_label": item["expected_label"],
            "predicted_label": report["predicted_attack_type"],
            "correct": report["predicted_attack_type"] == item["expected_label"],
            "attack_detected": report["attack_detected"],
            "expected_attack_detected": item["expected_label"] != "none",
            "heuristic_prediction": report["heuristic"]["predicted_attack_type"],
            "heuristic_confidence": report["heuristic"]["confidence"],
            "heuristic_triggered": report["heuristic"]["should_invoke_llm"],
            "heuristic_reasons": "|".join(report["heuristic"]["trigger_reasons"]),
            "llm_invoked": report["llm"]["invoked"],
            "llm_available": report["llm"]["available"],
            "llm_faithful": report["llm"]["faithful"],
            "llm_final_label": report["llm"]["final_label"],
            "llm_confidence": report["llm"]["confidence"],
            "llm_request_more_evidence": report["llm"]["request_more_evidence"],
            "llm_fallback_reason": report["llm"]["fallback_reason"],
            "final_confidence": report["confidence"],
            "sensor_mean_abs_rssi_shift_db": report["evidence"]["sensor_mean_abs_rssi_shift_db"],
            "sensor_worst_packet_ratio": report["evidence"]["sensor_worst_packet_ratio"],
            "sensor_mean_std_delta_db": report["evidence"]["sensor_mean_std_delta_db"],
            "gateway_mean_abs_rssi_shift_db": report["evidence"]["gateway_mean_abs_rssi_shift_db"],
            "gateway_worst_packet_ratio": report["evidence"]["gateway_worst_packet_ratio"],
            "gateway_impacted_sensor_count": report["evidence"]["gateway_impacted_sensor_count"],
            "localization_error_delta_m": report["evidence"]["localization_error_delta_m"],
            "trilateration_residual_delta_m": report["evidence"]["trilateration_residual_delta_m"],
            "top_sensor_score": report["evidence"]["top_sensor_score"],
            "top_gateway_score": report["evidence"]["top_gateway_score"],
            "suspicious_sensor": report["suspicious_sensor"],
            "suspicious_gateway": report["suspicious_gateway"],
        }
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for mode, df in rows.groupby("mode", sort=False):
        ambiguous = df[df["heuristic_triggered"]]
        cleanish = df[df["expected_label"] == "none"]
        summaries.append(
            {
                "mode": mode,
                "scenario_count": int(df.shape[0]),
                "overall_accuracy": float(df["correct"].mean()),
                "attack_detection_accuracy": float((df["attack_detected"] == df["expected_attack_detected"]).mean()),
                "ambiguous_case_accuracy": float(ambiguous["correct"].mean()) if not ambiguous.empty else 0.0,
                "false_positive_rate_on_none": float(cleanish["attack_detected"].mean()) if not cleanish.empty else 0.0,
                "heuristic_trigger_rate": float(df["heuristic_triggered"].mean()),
                "llm_invocation_rate": float(df["llm_invoked"].mean()),
                "llm_faithfulness_rate": float(df.loc[df["llm_invoked"], "llm_faithful"].mean()) if df["llm_invoked"].any() else 0.0,
            }
        )
    return pd.DataFrame(summaries)


def select_representative_ambiguous(off_rows: pd.DataFrame) -> set[str]:
    ambiguous = off_rows.loc[off_rows["heuristic_triggered"]].copy()
    if ambiguous.empty:
        return set()
    selected: list[str] = []
    for family, family_df in ambiguous.groupby("attack_family", sort=False):
        family_df = family_df.sort_values(["severity", "final_confidence"], ascending=[True, True])
        selected.extend(family_df.head(2)["scenario_name"].tolist())
    return set(selected)


def save_case_study_json(server: GatewayCoordinatorServer, out_dir: Path) -> None:
    cases = [
        (
            "sensor_foil_borderline",
            AttackScenario(attack_type="sensor_foil", sensor="sensor08", rssi_shift_db=-4),
        ),
        (
            "packet_drop_borderline",
            AttackScenario(attack_type="packet_drop", gateway="gatewayA", drop_prob=0.3, seed=7),
        ),
        (
            "random_noise_borderline",
            AttackScenario(attack_type="random_noise", sensor="sensor05", noise_sigma_db=5.0, seed=7),
        ),
    ]
    agent = ReActGatewayAgent(server, llm_mode="off")
    payload = {}
    for name, scenario in cases:
        report = agent.investigate(scenario).to_dict()
        payload[name] = report
    (out_dir / "case_studies.json").write_text(json.dumps(payload, indent=2))


def plot_mode_metrics(summary: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        "overall_accuracy",
        "ambiguous_case_accuracy",
        "false_positive_rate_on_none",
        "llm_invocation_rate",
    ]
    labels = ["Overall Acc.", "Ambiguous Acc.", "False Pos.", "LLM Invoke"]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    modes = summary["mode"].tolist()
    colors = {"off": "#355070", "adjudicate": "#6d597a", "explain": "#b56576"}
    for idx, mode in enumerate(modes):
        values = summary.loc[summary["mode"] == mode, metrics].iloc[0].to_numpy(dtype=float)
        ax.bar(x + (idx - (len(modes) - 1) / 2) * width, values, width=width, label=mode, color=colors.get(mode, "#457b9d"))
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Mode Comparison on Weak-to-Strong Attack Sweep")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "mode_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ambiguity_by_severity(rows: pd.DataFrame, out_dir: Path) -> None:
    families = ["sensor_foil", "gateway_bias", "random_noise", "packet_drop"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    axes = axes.flatten()
    color_map = {"off": "#355070", "adjudicate": "#b56576"}
    for ax, family in zip(axes, families):
        family_rows = rows[rows["attack_family"] == family]
        for mode, mode_df in family_rows.groupby("mode", sort=False):
            agg = (
                mode_df.groupby("severity", as_index=False)
                .agg(
                    heuristic_trigger_rate=("heuristic_triggered", "mean"),
                    accuracy=("correct", "mean"),
                )
                .sort_values("severity")
            )
            ax.plot(
                agg["severity"],
                agg["heuristic_trigger_rate"],
                marker="o",
                color=color_map.get(mode, "#457b9d"),
                label=f"{mode} trigger",
            )
            ax.plot(
                agg["severity"],
                agg["accuracy"],
                marker="s",
                linestyle="--",
                color=color_map.get(mode, "#457b9d"),
                alpha=0.55,
                label=f"{mode} accuracy",
            )
        ax.set_title(family.replace("_", " ").title())
        ax.set_xlabel("Severity")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:4], labels[:4], loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Where Ambiguity Appears Across Attack Severity", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "ambiguity_vs_severity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_case_heatmaps(server: GatewayCoordinatorServer, out_dir: Path) -> None:
    cases = [
        ("Borderline Sensor Foil", AttackScenario(attack_type="sensor_foil", sensor="sensor08", rssi_shift_db=-4)),
        ("Borderline Packet Drop", AttackScenario(attack_type="packet_drop", gateway="gatewayA", drop_prob=0.3, seed=7)),
        ("Borderline Random Noise", AttackScenario(attack_type="random_noise", sensor="sensor05", noise_sigma_db=5.0, seed=7)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (title, scenario) in zip(axes, cases):
        delta = server.get_network_snapshot(scenario)["delta"]
        pivot = (
            delta.pivot(index="sensor", columns="gateway", values="rssi_shift_db")
            .reindex(sorted(delta["sensor"].unique()))
            .fillna(0.0)
        )
        im = ax.imshow(pivot.to_numpy(), cmap="coolwarm", aspect="auto", vmin=-12, vmax=12)
        ax.set_title(title)
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.iloc[i, j]:.1f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=axes, shrink=0.8)
    cbar.set_label("Median RSSI Shift (dB)")
    fig.suptitle("Case Studies: Per-Sensor / Per-Gateway RSSI Delta Patterns", y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / "case_study_heatmaps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_localization_case(server: GatewayCoordinatorServer, out_dir: Path) -> None:
    cases = [
        ("Clean", AttackScenario(attack_type="none"), "sensor05", "#355070"),
        ("Noise σ=5", AttackScenario(attack_type="random_noise", sensor="sensor05", noise_sigma_db=5.0, seed=7), "sensor05", "#e56b6f"),
        ("Foil -4 dB", AttackScenario(attack_type="sensor_foil", sensor="sensor08", rssi_shift_db=-4), "sensor08", "#6d597a"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axes[0]
    for gateway, (gx, gy) in server.gateway_xy.items():
        ax.scatter(gx, gy, marker="^", s=120, color="#2a9d8f")
        ax.text(gx + 10, gy + 10, gateway, fontsize=8)
    plotted_truth = set()
    for label, scenario, sensor, color in cases:
        snap = server.get_network_snapshot(scenario)
        tri = server.trilateration_view(snap["packets"], sensor)
        x_true, y_true = server.sensor_xy_true[sensor]
        if sensor not in plotted_truth:
            ax.scatter(x_true, y_true, marker="x", s=80, color="black")
            ax.text(x_true + 8, y_true + 8, f"{sensor} true", fontsize=8)
            plotted_truth.add(sensor)
        if tri.get("available"):
            ax.scatter(tri["x_est"], tri["y_est"], s=90, color=color, label=f"{label} est.")
    ax.set_title("Localization Shifts in Representative Cases")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1]
    records = []
    for label, scenario, sensor, color in cases:
        snap = server.get_network_snapshot(scenario)
        tri = server.trilateration_view(snap["packets"], sensor)
        records.append(
            {
                "label": label,
                "error_m": float(tri.get("error_m", np.nan)),
                "residual_rmse_m": float(tri.get("residual_rmse_m", np.nan)),
                "color": color,
            }
        )
    df = pd.DataFrame(records)
    x = np.arange(df.shape[0])
    ax.bar(x - 0.18, df["error_m"], width=0.36, color=df["color"], alpha=0.9, label="Localization error")
    ax.bar(x + 0.18, df["residual_rmse_m"], width=0.36, color=df["color"], alpha=0.45, label="Residual RMSE")
    ax.set_xticks(x, df["label"], rotation=20)
    ax.set_ylabel("Meters")
    ax.set_title("Physics-Based Cross-Checks")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "localization_case_study.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary_markdown(summary: pd.DataFrame, rows: pd.DataFrame, out_dir: Path) -> None:
    off = summary.loc[summary["mode"] == "off"].iloc[0]
    adjudicate = summary.loc[summary["mode"] == "adjudicate"].iloc[0]
    ambiguous = rows[(rows["mode"] == "off") & (rows["heuristic_triggered"])]
    top_reasons = (
        ambiguous["heuristic_reasons"]
        .str.split("|")
        .explode()
        .replace("", np.nan)
        .dropna()
        .value_counts()
        .head(5)
    )
    lines = [
        "# Results Bundle",
        "",
        "## Headline Findings",
        f"- The weak-to-strong sweep contains {rows['scenario_name'].nunique()} scenarios and {ambiguous.shape[0]} heuristic-borderline cases.",
        f"- Heuristic-only overall accuracy: {off['overall_accuracy']:.3f}; ambiguous-case accuracy: {off['ambiguous_case_accuracy']:.3f}.",
        f"- LLM-adjudication overall accuracy: {adjudicate['overall_accuracy']:.3f}; ambiguous-case accuracy: {adjudicate['ambiguous_case_accuracy']:.3f}.",
        f"- LLM invocation rate in adjudication mode: {adjudicate['llm_invocation_rate']:.3f}.",
        "- For speed, the bundle only runs the LLM on a representative subset of the ambiguous scenarios; non-ambiguous scenarios inherit the heuristic result.",
        "",
        "## What To Show",
        "- `mode_metrics.png`: summary bar chart for advisor-facing discussion.",
        "- `ambiguity_vs_severity.png`: where borderline cases emerge by attack family.",
        "- `case_study_heatmaps.png`: interpretable per-sensor / per-gateway RSSI patterns.",
        "- `localization_case_study.png`: geometric cross-check that complements the RSSI plots.",
        "",
        "## Most Common Ambiguity Triggers",
    ]
    for reason, count in top_reasons.items():
        lines.append(f"- `{reason}`: {count} scenarios")
    lines.extend(
        [
            "",
            "## Suggested Story",
            "- Strong attacks are easy for heuristics; the interesting region is the weak-to-medium regime.",
            "- Ambiguity appears when anomaly scores rise but do not cleanly match one attack signature.",
            "- This is where the LLM is justified as a bounded adjudicator rather than a heuristic replacement.",
        ]
    )
    (out_dir / "RESULTS_SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("ggplot")
    server = GatewayCoordinatorServer()
    server.ensure_metadata()
    scenarios = build_scenarios(server)

    off_rows = evaluate_mode(
        server=server,
        scenarios=scenarios,
        mode="off",
        llm_model=args.llm_model,
        ollama_endpoint=args.ollama_endpoint,
    )
    ambiguous_names = select_representative_ambiguous(off_rows)
    ambiguous_scenarios = [item for item in scenarios if item["scenario_name"] in ambiguous_names]
    adjudicate_subset = evaluate_mode(
        server=server,
        scenarios=ambiguous_scenarios,
        mode="adjudicate",
        llm_model=args.llm_model,
        ollama_endpoint=args.ollama_endpoint,
    )
    adjudicate_rows = off_rows.copy()
    adjudicate_rows["mode"] = "adjudicate"
    adjudicate_rows["llm_invoked"] = False
    adjudicate_rows["llm_available"] = False
    adjudicate_rows["llm_faithful"] = False
    adjudicate_rows["llm_final_label"] = None
    adjudicate_rows["llm_confidence"] = None
    adjudicate_rows["llm_request_more_evidence"] = False
    adjudicate_rows["llm_fallback_reason"] = None
    adjudicate_rows = adjudicate_rows.set_index("scenario_name")
    adjudicate_subset = adjudicate_subset.set_index("scenario_name")
    for column in adjudicate_subset.columns:
        adjudicate_rows.loc[adjudicate_subset.index, column] = adjudicate_subset[column]
    adjudicate_rows = adjudicate_rows.reset_index()
    rows = pd.concat([off_rows, adjudicate_rows], ignore_index=True)
    summary = summarize_metrics(rows)

    rows.to_csv(out_dir / "scenario_rows.csv", index=False)
    summary.to_csv(out_dir / "summary_metrics.csv", index=False)
    save_case_study_json(server, out_dir)
    plot_mode_metrics(summary, out_dir)
    plot_ambiguity_by_severity(rows, out_dir)
    plot_case_heatmaps(server, out_dir)
    plot_localization_case(server, out_dir)
    write_summary_markdown(summary, rows, out_dir)
    print(f"Wrote results bundle to {out_dir}")


if __name__ == "__main__":
    main()
