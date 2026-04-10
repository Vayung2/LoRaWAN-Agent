## ReAct LoRaWAN Agent

This repo is now centered on a server-side ReAct agent that coordinates gateway observations, detects synthetic attacks injected from `dataset/lorawan_metadata`, and explains the action sequence it used to justify the decision.

The detector is intentionally two-stage:

- deterministic RF heuristics remain the first-stage measurement and calibration anchor
- an optional local LLM only runs on ambiguous cases to adjudicate conflicts and explain the evidence

### Core idea

- The server loads a clean baseline snapshot from the dataset.
- A synthetic scenario is created by perturbing RSSI or packet volume during load time.
- The ReAct agent inspects network-wide gateway evidence, drills into suspicious sensors and gateways, runs trilateration, and emits an explicit reasoning trace.

### Setup

1. Create a virtual environment and install `requirements.txt`.
2. Make sure `models/metadata.json` and `models/traditional_params.json` exist.

If you need to rebuild metadata:

```bash
python -m src.build_metadata
```

If you need to recalibrate the traditional path-loss model:

```bash
python -m src.traditional.calibrate --data_dir dataset/lorawan_metadata --meta models/metadata.json --out models/traditional_params.json
```

### Run the ReAct agent

Baseline check:

```bash
python3 -m src.run_react_agent --attack-type none
```

Synthetic sensor foil example:

```bash
python3 -m src.run_react_agent --attack-type sensor_foil --sensor sensor08 --rssi-shift-db -12
```

Synthetic gateway bias example:

```bash
python3 -m src.run_react_agent --attack-type gateway_bias --gateway gatewayA --rssi-shift-db -10
```

Synthetic packet drop example:

```bash
python3 -m src.run_react_agent --attack-type packet_drop --gateway gatewayB --drop-prob 0.7 --seed 7
```

Synthetic random noise example:

```bash
python3 -m src.run_react_agent --attack-type random_noise --sensor sensor05 --noise-sigma-db 6 --seed 7
```

Optional JSON report output:

```bash
python3 -m src.run_react_agent --attack-type sensor_foil --sensor sensor08 --rssi-shift-db -12 --json-out outputs/sensor08_report.json
```

### LLM modes

The agent supports three modes:

- `--llm-mode off`: pure heuristic baseline
- `--llm-mode explain`: keep the heuristic label and use the LLM only for rationale on ambiguous cases
- `--llm-mode adjudicate`: allow the LLM to override the heuristic only on ambiguous cases, with deterministic fallback if Ollama is unavailable or malformed

Example:

```bash
python3 -m src.run_react_agent --attack-type random_noise --sensor sensor05 --noise-sigma-db 6 --llm-mode adjudicate --llm-model qwen2.5:7b
```

### Ablation evaluation

To compare heuristic-only, explanation-only, and adjudication settings:

```bash
python3 -m src.evaluate_react_agent --json-out outputs/eval_summary.json
```

The evaluation summary reports:

- overall label accuracy
- ambiguous-case accuracy
- false-positive rate on clean and weak-noise benign scenarios
- LLM invocation rate
- explanation faithfulness rate based on cited evidence keys
