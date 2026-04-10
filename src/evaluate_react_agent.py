from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.react_agent import AttackScenario, GatewayCoordinatorServer, OllamaAdjudicator, OllamaConfig, ReActGatewayAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate heuristic-only, explanation-only, and LLM-adjudicated LoRaWAN detection."
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["off", "explain", "adjudicate"],
        choices=["off", "explain", "adjudicate"],
        help="Agent modes to evaluate.",
    )
    parser.add_argument("--llm-model", default="qwen2.5:7b")
    parser.add_argument(
        "--ollama-endpoint",
        default="http://127.0.0.1:11434/api/generate",
    )
    parser.add_argument("--json-out", default=None, help="Optional JSON path for the evaluation summary.")
    return parser


def make_scenarios(server: GatewayCoordinatorServer) -> list[dict]:
    sensors = sorted(server.metadata["SENSORS_LATLON"].keys())
    gateways = sorted(server.metadata["GATEWAYS_LATLON"].keys())
    scenarios: list[dict] = [
        {
            "name": "baseline_clean",
            "expected_label": "none",
            "group": "clean",
            "scenario": AttackScenario(attack_type="none"),
        }
    ]
    for sensor in sensors[:4]:
        scenarios.append(
            {
                "name": f"sensor_foil_{sensor}",
                "expected_label": "sensor_foil",
                "group": "attack",
                "scenario": AttackScenario(
                    attack_type="sensor_foil",
                    sensor=sensor,
                    rssi_shift_db=-12.0,
                ),
            }
        )
    for gateway in gateways:
        scenarios.append(
            {
                "name": f"gateway_bias_{gateway}",
                "expected_label": "gateway_bias",
                "group": "attack",
                "scenario": AttackScenario(
                    attack_type="gateway_bias",
                    gateway=gateway,
                    rssi_shift_db=-10.0,
                ),
            }
        )
    for sensor in sensors[4:7]:
        scenarios.append(
            {
                "name": f"random_noise_{sensor}",
                "expected_label": "random_noise",
                "group": "attack",
                "scenario": AttackScenario(
                    attack_type="random_noise",
                    sensor=sensor,
                    noise_sigma_db=6.0,
                    seed=7,
                ),
            }
        )
    for gateway in gateways:
        scenarios.append(
            {
                "name": f"packet_drop_{gateway}",
                "expected_label": "packet_drop",
                "group": "attack",
                "scenario": AttackScenario(
                    attack_type="packet_drop",
                    gateway=gateway,
                    drop_prob=0.7,
                    seed=7,
                ),
            }
        )
    for sensor in sensors[7:10]:
        scenarios.append(
            {
                "name": f"weak_noise_benign_{sensor}",
                "expected_label": "none",
                "group": "weak_noise_benign",
                "scenario": AttackScenario(
                    attack_type="random_noise",
                    sensor=sensor,
                    noise_sigma_db=1.5,
                    seed=11,
                ),
            }
        )
    return scenarios


def evaluate_mode(
    *,
    server: GatewayCoordinatorServer,
    mode: str,
    scenarios: list[dict],
    llm_model: str,
    ollama_endpoint: str,
) -> dict:
    llm_client = None
    if mode != "off":
        llm_client = OllamaAdjudicator(OllamaConfig(model=llm_model, endpoint=ollama_endpoint))
    agent = ReActGatewayAgent(server, llm_client=llm_client, llm_mode=mode)

    rows: list[dict] = []
    for item in scenarios:
        report = agent.investigate(item["scenario"]).to_dict()
        rows.append(
            {
                "name": item["name"],
                "group": item["group"],
                "expected_label": item["expected_label"],
                "predicted_label": report["predicted_attack_type"],
                "correct": report["predicted_attack_type"] == item["expected_label"],
                "attack_detected": report["attack_detected"],
                "expected_attack_detected": item["expected_label"] != "none",
                "llm_invoked": report["llm"]["invoked"],
                "llm_available": report["llm"]["available"],
                "llm_faithful": report["llm"]["faithful"],
                "request_more_evidence": report["llm"]["request_more_evidence"],
                "heuristic_triggered": report["heuristic"]["should_invoke_llm"],
                "heuristic_reasons": report["heuristic"]["trigger_reasons"],
                "heuristic_prediction": report["heuristic"]["predicted_attack_type"],
                "heuristic_confidence": report["heuristic"]["confidence"],
                "final_confidence": report["confidence"],
            }
        )

    total = len(rows)
    ambiguous = [row for row in rows if row["heuristic_triggered"]]
    cleanish = [row for row in rows if row["group"] in {"clean", "weak_noise_benign"}]
    case_studies = [
        row
        for row in rows
        if row["heuristic_triggered"] or row["llm_invoked"] or not row["correct"]
    ][:3]
    return {
        "mode": mode,
        "overall_accuracy": round(sum(row["correct"] for row in rows) / total, 3),
        "attack_detection_accuracy": round(
            sum(row["attack_detected"] == row["expected_attack_detected"] for row in rows) / total,
            3,
        ),
        "ambiguous_case_accuracy": round(
            sum(row["correct"] for row in ambiguous) / max(len(ambiguous), 1),
            3,
        ),
        "false_positive_rate_clean_and_weak_noise": round(
            sum(row["attack_detected"] for row in cleanish) / max(len(cleanish), 1),
            3,
        ),
        "llm_invocation_rate": round(sum(row["llm_invoked"] for row in rows) / total, 3),
        "faithful_explanations_rate": round(
            sum(row["llm_faithful"] for row in rows if row["llm_invoked"]) / max(sum(row["llm_invoked"] for row in rows), 1),
            3,
        ),
        "case_studies": case_studies,
        "rows": rows,
    }


def main() -> None:
    args = build_parser().parse_args()
    server = GatewayCoordinatorServer()
    server.ensure_metadata()
    scenarios = make_scenarios(server)
    summary = {
        "scenario_count": len(scenarios),
        "modes": [
            evaluate_mode(
                server=server,
                mode=mode,
                scenarios=scenarios,
                llm_model=args.llm_model,
                ollama_endpoint=args.ollama_endpoint,
            )
            for mode in args.modes
        ],
    }

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
