from __future__ import annotations

from typing import Any

import numpy as np

from .llm import OllamaAdjudicator
from .server import GatewayCoordinatorServer
from .types import (
    AttackScenario,
    DetectionReport,
    HeuristicAssessment,
    LLMAdjudication,
    TraceStep,
)


class ReActGatewayAgent:
    """
    Two-stage ReAct-style agent.

    Deterministic heuristics remain the first-stage detector. An optional LLM
    is only invoked to explain or adjudicate borderline cases where the
    heuristic evidence is ambiguous or conflicting.
    """

    ALLOWED_LABELS = ["none", "sensor_foil", "gateway_bias", "random_noise", "packet_drop"]
    ALLOWED_TOOLS = ["inspect_sensor", "inspect_gateway", "trilaterate", "stop"]

    def __init__(
        self,
        server: GatewayCoordinatorServer,
        llm_client: OllamaAdjudicator | None = None,
        llm_mode: str = "adjudicate",
    ):
        self.server = server
        self.llm_client = llm_client
        self.llm_mode = llm_mode

    def investigate(self, scenario: AttackScenario) -> DetectionReport:
        trace: list[TraceStep] = []
        snapshot = self.server.get_network_snapshot(scenario)
        delta = snapshot["delta"]
        sensor_rank = self.server.rank_sensors(delta)
        gateway_rank = self.server.rank_gateways(delta)

        trace.append(
            TraceStep(
                thought="Start from a network-wide snapshot so I can see whether anomalies cluster by sensor, gateway, or packet volume.",
                action="load_network_snapshot",
                observation=(
                    f"Compared {int(delta.shape[0])} sensor-gateway links against the clean baseline; "
                    f"top sensor anomaly score is {self._top_score(sensor_rank):.2f} and top gateway anomaly score is {self._top_score(gateway_rank):.2f}."
                ),
            )
        )

        suspicious_sensor = self._choose_sensor(sensor_rank, scenario)
        suspicious_gateway = self._choose_gateway(gateway_rank, scenario)

        sensor_view = self.server.sensor_view(delta, suspicious_sensor) if suspicious_sensor else None
        gateway_view = self.server.gateway_view(delta, suspicious_gateway) if suspicious_gateway else None

        if sensor_view is not None and not sensor_view.empty:
            trace.append(
                TraceStep(
                    thought="Inspect the most suspicious sensor across all gateways to see whether the evidence is consistent with a compromised end device.",
                    action=f"inspect_sensor({suspicious_sensor})",
                    observation=self._summarize_sensor_view(sensor_view),
                )
            )

        if gateway_view is not None and not gateway_view.empty:
            trace.append(
                TraceStep(
                    thought="Check whether one gateway is producing the same anomaly for many different sensors, which would point to gateway-side corruption.",
                    action=f"inspect_gateway({suspicious_gateway})",
                    observation=self._summarize_gateway_view(gateway_view),
                )
            )

        trilateration = (
            self.server.trilateration_view(snapshot["packets"], suspicious_sensor)
            if suspicious_sensor
            else {"available": False, "reason": "No suspicious sensor was selected."}
        )
        baseline_trilateration = (
            self.server.baseline_trilateration_view(suspicious_sensor)
            if suspicious_sensor
            else {"available": False, "reason": "No suspicious sensor was selected."}
        )

        trace.append(
            TraceStep(
                thought="Use trilateration as a physics-grounded cross-check; a real attack should distort the inferred geometry, not just one scalar feature.",
                action=f"trilaterate({suspicious_sensor})" if suspicious_sensor else "trilaterate(None)",
                observation=self._summarize_trilateration(trilateration, baseline_trilateration),
            )
        )

        heuristic = self._classify(
            sensor_view=sensor_view,
            gateway_view=gateway_view,
            sensor_rank=sensor_rank,
            gateway_rank=gateway_rank,
            trilateration=trilateration,
            baseline_trilateration=baseline_trilateration,
        )
        trace.append(
            TraceStep(
                thought="Run deterministic scoring first so the numeric detector remains the calibration anchor.",
                action="heuristic_classification",
                observation=(
                    f"Heuristic verdict `{heuristic.verdict}` predicted `{heuristic.predicted_attack_type}` with confidence "
                    f"{heuristic.confidence:.2f}. Trigger LLM: {heuristic.should_invoke_llm}. "
                    f"Reasons: {heuristic.trigger_reasons or ['none']}."
                ),
            )
        )

        llm = self._default_llm_result()
        final_label = heuristic.predicted_attack_type
        final_confidence = heuristic.confidence

        if self.llm_mode != "off" and heuristic.should_invoke_llm:
            llm_payload = self._build_llm_payload(
                scenario=scenario,
                heuristic=heuristic,
                suspicious_sensor=suspicious_sensor,
                suspicious_gateway=suspicious_gateway,
                sensor_view=sensor_view,
                gateway_view=gateway_view,
                trilateration=trilateration,
                baseline_trilateration=baseline_trilateration,
            )
            llm = self._run_llm(llm_payload)
            if llm.available and llm.fallback_reason is None:
                trace.append(
                    TraceStep(
                        thought="The heuristic evidence is borderline, so escalate to the LLM only as a bounded adjudicator over structured tool outputs.",
                        action=f"llm_{self.llm_mode}",
                        observation=(
                            f"LLM returned label `{llm.final_label}` with confidence {llm.confidence:.2f}, "
                            f"requested more evidence={llm.request_more_evidence}, and cited {llm.evidence_used}."
                        ),
                    )
                )
                if self.llm_mode == "adjudicate" and llm.final_label is not None:
                    final_label = llm.final_label
                    final_confidence = float(llm.confidence or heuristic.confidence)
            else:
                trace.append(
                    TraceStep(
                        thought="The LLM is optional, so malformed or unavailable responses must fall back to the deterministic detector.",
                        action=f"llm_{self.llm_mode}_fallback",
                        observation=llm.fallback_reason or "LLM was skipped.",
                    )
                )

        attack_detected = final_label != "none" or llm.request_more_evidence
        return DetectionReport(
            scenario=scenario.to_dict(),
            predicted_attack_type=final_label,
            suspicious_sensor=suspicious_sensor,
            suspicious_gateway=suspicious_gateway,
            attack_detected=attack_detected,
            confidence=final_confidence,
            evidence=heuristic.evidence,
            heuristic=heuristic,
            llm=llm,
            trace=trace,
        )

    def _run_llm(self, payload: dict[str, Any]) -> LLMAdjudication:
        if self.llm_client is None:
            return LLMAdjudication(
                invoked=True,
                mode=self.llm_mode,
                available=False,
                fallback_reason="No LLM client was configured.",
            )
        return self.llm_client.adjudicate(
            payload,
            mode=self.llm_mode,
            allowed_labels=self.ALLOWED_LABELS,
            allowed_tools=self.ALLOWED_TOOLS,
        )

    def _default_llm_result(self) -> LLMAdjudication:
        return LLMAdjudication(invoked=False, mode=self.llm_mode)

    def _build_llm_payload(
        self,
        *,
        scenario: AttackScenario,
        heuristic: HeuristicAssessment,
        suspicious_sensor: str | None,
        suspicious_gateway: str | None,
        sensor_view,
        gateway_view,
        trilateration: dict[str, Any],
        baseline_trilateration: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "scenario": scenario.to_dict(),
            "heuristic": heuristic.to_dict(),
            "suspicious_sensor": suspicious_sensor,
            "suspicious_gateway": suspicious_gateway,
            "sensor_summary": self._compact_frame(sensor_view, limit=3),
            "gateway_summary": self._compact_frame(gateway_view, limit=4),
            "trilateration": self._compact_trilateration(trilateration),
            "baseline_trilateration": self._compact_trilateration(baseline_trilateration),
            "evidence": heuristic.evidence,
        }

    @staticmethod
    def _compact_frame(frame, limit: int) -> list[dict[str, Any]]:
        if frame is None or frame.empty:
            return []
        payload = frame.head(limit).to_dict(orient="records")
        return [{key: ReActGatewayAgent._json_safe(value) for key, value in row.items()} for row in payload]

    @staticmethod
    def _compact_trilateration(payload: dict[str, Any]) -> dict[str, Any]:
        keep = [
            "available",
            "sensor",
            "error_m",
            "residual_rmse_m",
            "pair_count",
            "x_est",
            "y_est",
            "reason",
        ]
        return {key: ReActGatewayAgent._json_safe(payload.get(key)) for key in keep if key in payload}

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, (np.floating, np.integer)):
            return float(value)
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    @staticmethod
    def _top_score(rank: Any) -> float:
        if rank is None or getattr(rank, "empty", True):
            return 0.0
        return float(rank.iloc[0]["anomaly_score"])

    @staticmethod
    def _choose_sensor(sensor_rank: Any, scenario: AttackScenario) -> str | None:
        if scenario.sensor:
            return scenario.sensor
        if sensor_rank is None or sensor_rank.empty:
            return None
        return str(sensor_rank.iloc[0]["sensor"])

    @staticmethod
    def _choose_gateway(gateway_rank: Any, scenario: AttackScenario) -> str | None:
        if scenario.gateway:
            return scenario.gateway
        if gateway_rank is None or gateway_rank.empty:
            return None
        return str(gateway_rank.iloc[0]["gateway"])

    @staticmethod
    def _summarize_sensor_view(sensor_view) -> str:
        mean_shift = float(sensor_view["rssi_shift_db"].mean())
        mean_abs_shift = float(sensor_view["rssi_shift_db"].abs().mean())
        min_ratio = float(sensor_view["packet_ratio"].min())
        mean_std_delta = float(sensor_view["std_delta_db"].mean())
        return (
            f"Across {int(sensor_view.shape[0])} gateways, the sensor shows mean RSSI shift {mean_shift:.2f} dB "
            f"(mean absolute {mean_abs_shift:.2f} dB), worst packet-retention ratio {min_ratio:.2f}, "
            f"and mean RSSI spread change {mean_std_delta:.2f} dB."
        )

    @staticmethod
    def _summarize_gateway_view(gateway_view) -> str:
        mean_shift = float(gateway_view["rssi_shift_db"].mean())
        mean_abs_shift = float(gateway_view["rssi_shift_db"].abs().mean())
        impacted = int((gateway_view["rssi_shift_db"].abs() >= 6.0).sum())
        min_ratio = float(gateway_view["packet_ratio"].min())
        return (
            f"This gateway affects {int(gateway_view.shape[0])} sensors with mean RSSI shift {mean_shift:.2f} dB "
            f"(mean absolute {mean_abs_shift:.2f} dB); {impacted} sensors exceed the 6 dB shift threshold and "
            f"the worst packet-retention ratio is {min_ratio:.2f}."
        )

    @staticmethod
    def _summarize_trilateration(current: dict[str, Any], baseline: dict[str, Any]) -> str:
        if not current.get("available"):
            return current.get("reason", "Trilateration was unavailable.")

        baseline_error = float(baseline.get("error_m", np.nan)) if baseline.get("available") else np.nan
        current_error = float(current.get("error_m", np.nan))
        baseline_residual = (
            float(baseline.get("residual_rmse_m", np.nan)) if baseline.get("available") else np.nan
        )
        current_residual = float(current.get("residual_rmse_m", np.nan))
        return (
            f"Estimated location ({current['x_est']:.1f}, {current['y_est']:.1f}) m using {current['pair_count']} gateways. "
            f"Localization error changed from {baseline_error:.2f} m to {current_error:.2f} m and the multilateration residual "
            f"changed from {baseline_residual:.2f} m to {current_residual:.2f} m."
        )

    def _classify(
        self,
        sensor_view,
        gateway_view,
        sensor_rank,
        gateway_rank,
        trilateration: dict[str, Any],
        baseline_trilateration: dict[str, Any],
    ) -> HeuristicAssessment:
        sensor_shift = self._safe_mean_abs(sensor_view, "rssi_shift_db")
        sensor_ratio = self._safe_min(sensor_view, "packet_ratio", default=1.0)
        sensor_std_delta = self._safe_mean(sensor_view, "std_delta_db")

        gateway_shift = self._safe_mean_abs(gateway_view, "rssi_shift_db")
        gateway_ratio = self._safe_min(gateway_view, "packet_ratio", default=1.0)
        gateway_impacted = (
            int((gateway_view["rssi_shift_db"].abs() >= 6.0).sum())
            if gateway_view is not None and not gateway_view.empty
            else 0
        )

        baseline_error = float(baseline_trilateration.get("error_m", np.nan))
        current_error = float(trilateration.get("error_m", np.nan))
        baseline_residual = float(baseline_trilateration.get("residual_rmse_m", np.nan))
        current_residual = float(trilateration.get("residual_rmse_m", np.nan))

        error_delta = (
            current_error - baseline_error
            if np.isfinite(current_error) and np.isfinite(baseline_error)
            else 0.0
        )
        residual_delta = (
            current_residual - baseline_residual
            if np.isfinite(current_residual) and np.isfinite(baseline_residual)
            else 0.0
        )

        evidence = {
            "sensor_mean_abs_rssi_shift_db": round(sensor_shift, 3),
            "sensor_worst_packet_ratio": round(sensor_ratio, 3),
            "sensor_mean_std_delta_db": round(sensor_std_delta, 3),
            "gateway_mean_abs_rssi_shift_db": round(gateway_shift, 3),
            "gateway_worst_packet_ratio": round(gateway_ratio, 3),
            "gateway_impacted_sensor_count": gateway_impacted,
            "localization_error_delta_m": round(error_delta, 3),
            "trilateration_residual_delta_m": round(residual_delta, 3),
            "top_sensor_score": round(self._top_score(sensor_rank), 3),
            "top_gateway_score": round(self._top_score(gateway_rank), 3),
        }

        predicted_attack_type = "none"
        confidence = 0.6
        verdict = "uncertain"

        if max(evidence["top_sensor_score"], evidence["top_gateway_score"]) < 2.5:
            predicted_attack_type = "none"
            confidence = 0.92
            verdict = "accept"
        elif sensor_ratio < 0.55 or gateway_ratio < 0.55:
            predicted_attack_type = "packet_drop"
            confidence = self._bounded_confidence(0.75 + (1.0 - min(sensor_ratio, gateway_ratio)))
            verdict = "reject"
        elif gateway_shift >= 6.0 and gateway_impacted >= 4 and sensor_shift < gateway_shift:
            predicted_attack_type = "gateway_bias"
            confidence = self._bounded_confidence(0.78 + gateway_shift / 25.0)
            verdict = "reject"
        elif sensor_shift >= 6.0 and gateway_shift < sensor_shift * 0.8:
            predicted_attack_type = "sensor_foil"
            confidence = self._bounded_confidence(0.76 + sensor_shift / 25.0)
            verdict = "reject"
        elif sensor_std_delta >= 2.0 or residual_delta >= 10.0:
            predicted_attack_type = "random_noise"
            confidence = self._bounded_confidence(
                0.72 + max(sensor_std_delta, residual_delta / 10.0) / 10.0
            )
            verdict = "reject"

        trigger_reasons = self._ambiguity_reasons(
            evidence=evidence,
            predicted_attack_type=predicted_attack_type,
            confidence=confidence,
            verdict=verdict,
        )

        should_invoke_llm = bool(trigger_reasons)
        if should_invoke_llm:
            verdict = "uncertain"

        return HeuristicAssessment(
            verdict=verdict,
            predicted_attack_type=predicted_attack_type,
            confidence=confidence,
            should_invoke_llm=should_invoke_llm,
            trigger_reasons=trigger_reasons,
            evidence=evidence,
        )

    @staticmethod
    def _ambiguity_reasons(
        *,
        evidence: dict[str, Any],
        predicted_attack_type: str,
        confidence: float,
        verdict: str,
    ) -> list[str]:
        reasons: list[str] = []
        top_sensor = float(evidence["top_sensor_score"])
        top_gateway = float(evidence["top_gateway_score"])
        sensor_shift = float(evidence["sensor_mean_abs_rssi_shift_db"])
        gateway_shift = float(evidence["gateway_mean_abs_rssi_shift_db"])
        sensor_ratio = float(evidence["sensor_worst_packet_ratio"])
        gateway_ratio = float(evidence["gateway_worst_packet_ratio"])
        std_delta = float(evidence["sensor_mean_std_delta_db"])
        residual_delta = float(evidence["trilateration_residual_delta_m"])

        if confidence < 0.8:
            reasons.append("low_heuristic_confidence")
        if verdict == "uncertain":
            reasons.append("no_strong_rule_fired")
        if top_sensor >= 2.5 and top_gateway >= 2.5 and abs(top_sensor - top_gateway) <= 1.0:
            reasons.append("sensor_and_gateway_hypotheses_conflict")
        if sensor_shift >= 5.0 and gateway_shift >= 5.0:
            reasons.append("sensor_and_gateway_shifts_overlap")
        if min(sensor_ratio, gateway_ratio) < 0.7 and max(sensor_shift, gateway_shift) >= 4.0:
            reasons.append("packet_loss_and_rssi_shift_overlap")
        if 1.5 <= std_delta < 2.5 or 6.0 <= residual_delta < 12.0:
            reasons.append("borderline_noise_signature")
        if predicted_attack_type == "none" and max(top_sensor, top_gateway) >= 2.5:
            reasons.append("anomaly_score_above_clean_threshold")
        return reasons

    @staticmethod
    def _safe_mean_abs(df, column: str) -> float:
        if df is None or df.empty:
            return 0.0
        return float(df[column].abs().mean())

    @staticmethod
    def _safe_mean(df, column: str) -> float:
        if df is None or df.empty:
            return 0.0
        return float(df[column].mean())

    @staticmethod
    def _safe_min(df, column: str, default: float) -> float:
        if df is None or df.empty:
            return default
        return float(df[column].min())

    @staticmethod
    def _bounded_confidence(value: float) -> float:
        return float(np.clip(value, 0.0, 0.99))
