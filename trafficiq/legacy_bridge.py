from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LegacySignalConfig:
    default_red: int
    default_yellow: int
    default_green: int
    default_minimum: int
    default_maximum: int
    detection_time: int
    lanes: int
    vehicle_pass_times: dict[str, float]
    vehicle_speeds: dict[str, float]


LEGACY_SIMULATION_PATH = Path(__file__).resolve().parent.parent / "Code" / "YOLO" / "darkflow" / "simulation.py"
LEGACY_DETECTION_PATH = Path(__file__).resolve().parent.parent / "Code" / "YOLO" / "darkflow" / "vehicle_detection.py"


def _literal_assignments(source_path: Path) -> dict[str, object]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    assignments: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                assignments[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                continue
    return assignments


def load_legacy_signal_config() -> LegacySignalConfig:
    assignments = _literal_assignments(LEGACY_SIMULATION_PATH)
    return LegacySignalConfig(
        default_red=int(assignments["defaultRed"]),
        default_yellow=int(assignments["defaultYellow"]),
        default_green=int(assignments["defaultGreen"]),
        default_minimum=int(assignments["defaultMinimum"]),
        default_maximum=int(assignments["defaultMaximum"]),
        detection_time=int(assignments["detectionTime"]),
        lanes=int(assignments["noOfLanes"]),
        vehicle_pass_times={
            "car": float(assignments["carTime"]),
            "motorcycle": float(assignments["bikeTime"]),
            "rickshaw": float(assignments["rickshawTime"]),
            "bus": float(assignments["busTime"]),
            "truck": float(assignments["truckTime"]),
        },
        vehicle_speeds=dict(assignments["speeds"]),
    )


def legacy_source_status() -> dict[str, str | bool]:
    return {
        "simulation_exists": LEGACY_SIMULATION_PATH.exists(),
        "vehicle_detection_exists": LEGACY_DETECTION_PATH.exists(),
        "simulation_path": str(LEGACY_SIMULATION_PATH),
        "vehicle_detection_path": str(LEGACY_DETECTION_PATH),
    }
