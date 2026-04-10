from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.react_agent import (
    AttackScenario,
    GatewayCoordinatorServer,
    OllamaAdjudicator,
    OllamaConfig,
    ReActGatewayAgent,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the server-side ReAct LoRaWAN attack detection agent."
    )
    parser.add_argument(
        "--attack-type",
        default="none",
        choices=["none", "sensor_foil", "gateway_bias", "random_noise", "packet_drop"],
    )
    parser.add_argument("--sensor", default=None, help="Target sensor for sensor-side attacks.")
    parser.add_argument("--gateway", default=None, help="Target gateway for gateway-side attacks.")
    parser.add_argument("--rssi-shift-db", type=float, default=0.0)
    parser.add_argument("--noise-sigma-db", type=float, default=0.0)
    parser.add_argument("--drop-prob", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", default=None, help="Optional path for the full JSON report.")
    parser.add_argument(
        "--llm-mode",
        default="adjudicate",
        choices=["off", "explain", "adjudicate"],
        help="Use the LLM only for explanations or bounded adjudication on ambiguous cases.",
    )
    parser.add_argument("--llm-model", default="qwen2.5:7b", help="Ollama model name.")
    parser.add_argument(
        "--ollama-endpoint",
        default="http://127.0.0.1:11434/api/generate",
        help="Ollama HTTP endpoint for non-streaming generation.",
    )
    return parser


def format_trace(report: dict) -> str:
    lines = [
        f"Predicted attack: {report['predicted_attack_type']}",
        f"Attack detected: {report['attack_detected']}",
        f"Confidence: {report['confidence']:.2f}",
        f"Suspicious sensor: {report['suspicious_sensor']}",
        f"Suspicious gateway: {report['suspicious_gateway']}",
        f"Heuristic verdict: {report['heuristic']['verdict']}",
        f"Heuristic prediction: {report['heuristic']['predicted_attack_type']}",
        f"Heuristic confidence: {report['heuristic']['confidence']:.2f}",
        f"LLM invoked: {report['llm']['invoked']}",
        f"LLM mode: {report['llm']['mode']}",
        "Evidence:",
    ]
    for key, value in report["evidence"].items():
        lines.append(f"  - {key}: {value}")
    if report["heuristic"]["trigger_reasons"]:
        lines.append("Heuristic trigger reasons:")
        for reason in report["heuristic"]["trigger_reasons"]:
            lines.append(f"  - {reason}")
    if report["llm"]["invoked"]:
        lines.append("LLM adjudication:")
        lines.append(f"  - available: {report['llm']['available']}")
        lines.append(f"  - model: {report['llm']['model']}")
        lines.append(f"  - final_label: {report['llm']['final_label']}")
        lines.append(f"  - confidence: {report['llm']['confidence']}")
        lines.append(f"  - faithful: {report['llm']['faithful']}")
        lines.append(f"  - request_more_evidence: {report['llm']['request_more_evidence']}")
        lines.append(f"  - next_tool: {report['llm']['next_tool']}")
        lines.append(f"  - evidence_used: {report['llm']['evidence_used']}")
        lines.append(f"  - fallback_reason: {report['llm']['fallback_reason']}")
        if report["llm"]["rationale"]:
            lines.append(f"  - rationale: {report['llm']['rationale']}")
    lines.append("Trace:")
    for idx, step in enumerate(report["trace"], start=1):
        lines.append(f"  {idx}. Thought: {step['thought']}")
        lines.append(f"     Action: {step['action']}")
        lines.append(f"     Observation: {step['observation']}")
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()

    scenario = AttackScenario(
        attack_type=args.attack_type,
        sensor=args.sensor,
        gateway=args.gateway,
        rssi_shift_db=args.rssi_shift_db,
        noise_sigma_db=args.noise_sigma_db,
        drop_prob=args.drop_prob,
        seed=args.seed,
    )

    server = GatewayCoordinatorServer()
    server.ensure_metadata()
    llm_client = None
    if args.llm_mode != "off":
        llm_client = OllamaAdjudicator(
            OllamaConfig(model=args.llm_model, endpoint=args.ollama_endpoint)
        )
    agent = ReActGatewayAgent(server, llm_client=llm_client, llm_mode=args.llm_mode)
    report = agent.investigate(scenario).to_dict()

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))

    print(format_trace(report))


if __name__ == "__main__":
    main()
