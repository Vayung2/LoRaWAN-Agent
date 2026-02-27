from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from flask import Flask, request, render_template_string

from src.environment import LoRaWANEnvironment, LoRaWANEnvironmentConfig

app = Flask(__name__)
SENSORS = [f"sensor{str(i).zfill(2)}" for i in range(1, 11)]
GATEWAYS = ["gatewayA", "gatewayB", "gatewayC"]

# -----------------------------------------------------------------------------
# UI goals (as requested):
# - Keep defaults wherever possible (like CLI: run_inference with minimal args)
# - Only expose: method + attack + attack params + clean reports
# - No need to input metadata/data paths unless you later add an "Advanced" section
# - Output: deterministic for baseline, organized for attacks
# -----------------------------------------------------------------------------

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>LoRaWAN Localization Runner</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; max-width: 920px; }
    .row { display: flex; gap: 16px; margin-bottom: 12px; align-items: center; }
    label { min-width: 190px; font-weight: 700; }
    input, select { padding: 7px 9px; width: 280px; }
    .small { width: 140px; }
    .box { border: 1px solid #ddd; padding: 16px; border-radius: 10px; margin-top: 16px; }
    .btn { padding: 10px 14px; font-weight: 800; cursor: pointer; }
    .note { color: #444; font-size: 13px; }
    .ok { color: #0a7; font-weight: 800; }
    .err { color: #c00; font-weight: 800; white-space: pre-wrap; }
    code { background: #f6f6f6; padding: 2px 6px; border-radius: 6px; }
    .muted { color: #666; font-size: 12px; }
  </style>
</head>
<body>
  <h2>LoRaWAN Localization Runner (Localhost)</h2>
  <div class="note">
    Runs <code>LoRaWANEnvironment</code> using config defaults (same idea as your CLI).
    Writes outputs under <code>reports/</code>.
  </div>

  <form method="POST" class="box">
    <div class="row">
      <label>Method</label>
      <select name="method">
        <option value="traditional" {{'selected' if form.method=='traditional' else ''}}>traditional</option>
        <option value="regression" {{'selected' if form.method=='regression' else ''}}>regression</option>
      </select>
    </div>

    <div class="row">
      <label>Attack</label>
      <select name="attack">
        <option value="none" {{'selected' if form.attack=='none' else ''}}>none</option>
        <option value="sensor_foil" {{'selected' if form.attack=='sensor_foil' else ''}}>sensor foil</option>
        <option value="gateway_bias" {{'selected' if form.attack=='gateway_bias' else ''}}>gateway bias</option>
        <option value="random_noise" {{'selected' if form.attack=='random_noise' else ''}}>random noise</option>
        <option value="packet_drop" {{'selected' if form.attack=='packet_drop' else ''}}>packet drop / jamming</option>
      </select>
    </div>

    <div class="row">
        <label>Sensor (if needed)</label>
        <select name="sensor">
            <option value="none" {{'selected' if form.sensor=='none' else ''}}>none</option>
            {% for s in sensors %}
            <option value="{{s}}" {{'selected' if form.sensor==s else ''}}>{{s}}</option>
            {% endfor %}
        </select>
        <div class="muted">Used by sensor foil, or to target noise/drop to a sensor.</div>
    </div>

    <div class="row">
        <label>Gateway (if needed)</label>
        <select name="gateway">
            <option value="none" {{'selected' if form.gateway=='none' else ''}}>none</option>
            {% for g in gateways %}
            <option value="{{g}}" {{'selected' if form.gateway==g else ''}}>{{g}}</option>
            {% endfor %}
        </select>
        <div class="muted">Used by gateway bias, or to target noise/drop to a gateway.</div>
    </div>

    <div class="row">
      <label>RSSI shift (dB)</label>
      <input class="small" name="rssi_shift_db" placeholder="-10 or +10" value="{{form.rssi_shift_db}}">
      <div class="muted">Used by foil/bias. Negative = attenuation.</div>
    </div>

    <div class="row">
      <label>Noise sigma (dB)</label>
      <input class="small" name="noise_sigma_db" placeholder="0..6" value="{{form.noise_sigma_db}}">
      <div class="muted">Used by random noise.</div>
    </div>

    <div class="row">
      <label>Drop probability</label>
      <input class="small" name="drop_prob" placeholder="0..1" value="{{form.drop_prob}}">
      <div class="muted">Used by packet drop / jamming.</div>
    </div>

    <div class="row">
      <label>Seed</label>
      <input class="small" name="seed" placeholder="0" value="{{form.seed}}">
      <div class="muted">Reproducibility for noise/drop.</div>
    </div>

    <hr>

    <div class="row">
      <label>Clean reports/ before run</label>
      <input type="checkbox" name="clean_reports" {{'checked' if form.clean_reports else ''}}>
      <div class="note">Deletes everything under <code>reports/</code> before running.</div>
    </div>

    <div class="row">
      <button class="btn" type="submit">Run</button>
    </div>
  </form>

  {% if result %}
    <div class="box">
      {% if result.ok %}
        <div class="ok">✅ Run complete</div>
      {% else %}
        <div class="err">❌ Run failed\n{{result.error}}</div>
      {% endif %}
      <p><b>Output CSV:</b> <code>{{result.out_csv}}</code></p>
      <p><b>Median error (m):</b> <code>{{result.median_error}}</code></p>
      <p class="note"><b>Attack config:</b> {{result.attack_config}}</p>
      <p class="muted"><b>Used defaults from:</b> <code>LoRaWANEnvironmentConfig()</code> (unless overridden by attack inputs)</p>
    </div>
  {% endif %}

</body>
</html>
"""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def safe_clean_reports(reports_dir: str = "reports") -> None:
    """Deletes contents of reports/ safely (won't delete outside)."""
    p = Path(reports_dir).resolve()
    if p.name != "reports":
        raise ValueError(f"Refusing to clean non-reports directory: {p}")
    if not p.exists():
        return
    for child in p.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except OSError:
                pass


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def build_attack_kwargs(form: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map UI selection -> LoRaWANEnvironmentConfig attack fields.

    Assumes LoRaWANEnvironmentConfig includes:
      attack_scope, attack_sensor, attack_gateway,
      rssi_shift_db, rssi_noise_sigma_db, drop_prob, seed

    And load_and_prepare_packets() uses these to apply masked shift/noise/drop.
    """
    attack = form["attack"]
    sensor_raw = (form.get("sensor") or "").strip()
    gateway_raw = (form.get("gateway") or "").strip()
    sensor = None if sensor_raw in {"", "none", "None"} else sensor_raw
    gateway = None if gateway_raw in {"", "none", "None"} else gateway_raw

    rssi_shift_db = parse_float(form.get("rssi_shift_db", "0"), 0.0)
    noise_sigma_db = parse_float(form.get("noise_sigma_db", "0"), 0.0)
    drop_prob = parse_float(form.get("drop_prob", "0"), 0.0)
    seed = parse_int(form.get("seed", "0"), 0)

    # Default: no-attack
    scope = "global"
    attack_sensor = None
    attack_gateway = None
    rssi_noise_sigma_db = 0.0

    if attack == "none":
        return dict(
            attack_scope="global",
            attack_sensor=None,
            attack_gateway=None,
            rssi_shift_db=0.0,
            rssi_noise_sigma_db=0.0,
            drop_prob=0.0,
            seed=seed,
        )

    if attack == "sensor_foil":
        scope = "sensor"
        attack_sensor = sensor
        _require(attack_sensor is not None, "sensor foil requires Sensor (e.g., sensor08)")
        _require(rssi_shift_db != 0.0, "sensor foil requires a non-zero RSSI shift (e.g., -10)")
        # Leave sign to user, but typical is negative.

    elif attack == "gateway_bias":
        scope = "gateway"
        attack_gateway = gateway
        _require(attack_gateway is not None, "gateway bias requires Gateway (e.g., gatewayA)")
        _require(rssi_shift_db != 0.0, "gateway bias requires a non-zero RSSI shift (e.g., +10 or -10)")

    elif attack == "random_noise":
        # Prefer targeting: gateway if provided, else sensor if provided, else global
        if gateway:
            scope = "gateway"
            attack_gateway = gateway
        elif sensor:
            scope = "sensor"
            attack_sensor = sensor
        else:
            scope = "global"
        _require(noise_sigma_db > 0.0, "random noise requires Noise sigma (dB) > 0 (e.g., 2)")
        rssi_shift_db = 0.0
        drop_prob = 0.0
        rssi_noise_sigma_db = noise_sigma_db

    elif attack == "packet_drop":
        if gateway:
            scope = "gateway"
            attack_gateway = gateway
        elif sensor:
            scope = "sensor"
            attack_sensor = sensor
        else:
            scope = "global"
        _require(0.0 < drop_prob < 1.0, "packet drop requires Drop probability in (0,1), e.g. 0.3")
        rssi_shift_db = 0.0
        rssi_noise_sigma_db = 0.0
        noise_sigma_db = 0.0

    else:
        raise ValueError(f"Unknown attack selection: {attack}")

    return dict(
        attack_scope=scope,
        attack_sensor=attack_sensor,
        attack_gateway=attack_gateway,
        rssi_shift_db=rssi_shift_db,
        rssi_noise_sigma_db=rssi_noise_sigma_db if attack == "random_noise" else 0.0,
        drop_prob=drop_prob if attack == "packet_drop" else 0.0,
        seed=seed,
    )


def make_out_csv(method: str, attack: str, attack_kwargs: Dict[str, Any]) -> str:
    """
    Keep it simple + predictable:
      - baseline: reports/loc_{method}.csv   (overwrite like CLI)
      - attack:   reports/attacks/{attack}/loc_{method}_{tag}_{ts}.csv
    """
    if attack == "none":
        os.makedirs("reports", exist_ok=True)
        return os.path.join("reports", f"loc_{method}.csv")

    # attack output
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join("reports", "attacks", attack)
    os.makedirs(base, exist_ok=True)

    parts = [attack_kwargs.get("attack_scope", "global")]
    if attack_kwargs.get("attack_sensor"):
        parts.append(str(attack_kwargs["attack_sensor"]))
    if attack_kwargs.get("attack_gateway"):
        parts.append(str(attack_kwargs["attack_gateway"]))

    # include the relevant magnitude
    if attack in {"sensor_foil", "gateway_bias"}:
        parts.append(f"shift{attack_kwargs.get('rssi_shift_db', 0):g}dB")
    elif attack == "random_noise":
        parts.append(f"sigma{attack_kwargs.get('rssi_noise_sigma_db', 0):g}dB")
    elif attack == "packet_drop":
        parts.append(f"drop{attack_kwargs.get('drop_prob', 0):g}")

    parts.append(f"seed{attack_kwargs.get('seed', 0)}")

    tag = "_".join(parts).replace("+", "p").replace("-", "m")
    return os.path.join(base, f"loc_{method}_{tag}_{ts}.csv")


# -----------------------------------------------------------------------------
# Flask route
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    # Minimal defaults (keep CLI-like behavior)
    default_form = {
        "method": "traditional",
        "attack": "none",
        "sensor": "none",
        "gateway": "none",
        "rssi_shift_db": "0",
        "noise_sigma_db": "0",
        "drop_prob": "0",
        "seed": "0",
        "clean_reports": False,  # safer default: off
    }

    result: Optional[Dict[str, Any]] = None

    if request.method == "POST":
        form = default_form.copy()
        for k in list(form.keys()):
            if k == "clean_reports":
                form[k] = (request.form.get("clean_reports") == "on")
            else:
                form[k] = request.form.get(k, form[k])

        try:
            if form["clean_reports"]:
                safe_clean_reports("reports")

            method = form["method"]
            attack = form["attack"]

            attack_kwargs = build_attack_kwargs(form)

            # IMPORTANT: keep defaults wherever possible
            cfg = LoRaWANEnvironmentConfig()  # uses defaults like CLI
            for k, v in attack_kwargs.items():
                setattr(cfg, k, v)

            env = LoRaWANEnvironment(cfg)
            loc_df = env.run(method)

            out_csv = make_out_csv(method, attack, attack_kwargs)
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)
            loc_df.to_csv(out_csv, index=False)

            med = LoRaWANEnvironment.median_error(loc_df)
            result = {
                "ok": True,
                "out_csv": out_csv,
                "median_error": f"{med:.2f}" if med == med else "nan",
                "attack_config": {
                    "method": method,
                    "attack": attack,
                    **attack_kwargs,
                },
            }

        except Exception as e:
            result = {
                "ok": False,
                "error": repr(e),
                "out_csv": "",
                "median_error": "nan",
                "attack_config": {},
            }
        return render_template_string(HTML, form=form, result=result, sensors=SENSORS, gateways=GATEWAYS)

    return render_template_string(HTML, form=default_form, result=None, sensors=SENSORS, gateways=GATEWAYS)


if __name__ == "__main__":
    # http://127.0.0.1:5000
    app.run(host="127.0.0.1", port=5000, debug=True)