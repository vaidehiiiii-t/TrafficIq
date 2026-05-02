# TrafficIQ

TrafficIQ is a deployable Streamlit application for adaptive traffic-signal control. It keeps the original junction-control logic from the restored legacy source while replacing the non-deployable desktop detector stack with a modern web-facing pipeline.

## Architecture

TrafficIQ is split into two layers:

1. `Deployed app layer`
   - `trafficiq_app.py`
   - `trafficiq/detection.py`
   - `trafficiq/traffic_logic.py`
   - `trafficiq/legacy_bridge.py`

2. `Preserved legacy core`
   - `Code/YOLO/darkflow/simulation.py`
   - `Code/YOLO/darkflow/vehicle_detection.py`
   - supporting legacy assets and config files required by that original implementation

The intent is simple:

- keep the old signal-control logic
- keep the original source files in the repo
- build a new deployable product on top of that logic

## What Is Reused From The Old Code

TrafficIQ reads the original timing defaults and vehicle pass-time constants from:

- `Code/YOLO/darkflow/simulation.py`

That means the new deployed app inherits:

- minimum green time
- maximum green time
- yellow time
- lane assumptions
- vehicle pass-time weights for car, bike, bus, truck, and rickshaw

The bridge that loads those values is:

- `trafficiq/legacy_bridge.py`

## Why The Detector Was Modernized

The old detector entrypoint in:

- `Code/YOLO/darkflow/vehicle_detection.py`

depends on `darkflow` and an older TensorFlow 1.x style runtime. That is not a good fit for Streamlit deployment in 2026. So TrafficIQ preserves that file as the original reference implementation, but uses Ultralytics YOLO inside the Streamlit app for practical deployment.

This gives you:

- preserved original logic
- a working hosted app
- room to swap in your own trained model later

## Features

- Static image analysis for up to four directions
- Browser camera snapshot capture
- Video upload with sampled frame analysis
- Direction-wise vehicle counts
- Adaptive signal timing recommendations
- Signal-cycle simulation over multiple rounds

## Project Structure

```text
TrafficIQ/
|-- .streamlit/
|-- Code/
|   `-- YOLO/
|       `-- darkflow/
|           |-- simulation.py
|           |-- vehicle_detection.py
|           `-- ...
|-- trafficiq/
|   |-- __init__.py
|   |-- detection.py
|   |-- legacy_bridge.py
|   `-- traffic_logic.py
|-- trafficiq_app.py
|-- requirements.txt
|-- runtime.txt
|-- LICENSE
`-- README.md
```

## Run Locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run trafficiq_app.py
```

## Deploy On Streamlit

Use these settings:

- Main file path: `trafficiq_app.py`
- Python version: `3.11`

Notes:

- the default model path is `yolov8n.pt`
- Ultralytics may download the weight file automatically at first run
- for better accuracy, replace the default model with a custom traffic-trained YOLO checkpoint

## App Modes

### Images

Upload one image per approach and inspect the detected vehicles before calculating timings.

### Video + Camera

Upload traffic videos or capture a live camera snapshot. The app samples frames and estimates the peak load for each approach.

### Simulation

Run a cycle-based timing simulation using the legacy signal logic extracted from `simulation.py`.

## Current Logic Boundary

The deployed app currently reuses the old signal-control logic directly and preserves the old detection source as a reference. It does not execute the old `darkflow` detector in production mode.

That is intentional:

- old timing logic stays
- old source files stay
- deployment becomes practical

## Next Improvement Path

If you want even tighter legacy continuity, the next step would be:

1. isolate the detection decision logic from `vehicle_detection.py`
2. wrap it behind a provider interface
3. allow switching between `legacy-darkflow` and `ultralytics-yolo`

## License

Apache 2.0. See `LICENSE`.
