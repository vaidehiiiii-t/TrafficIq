from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from trafficiq.legacy_bridge import load_legacy_signal_config


LEGACY_CONFIG = load_legacy_signal_config()
VEHICLE_PASSING_TIME = LEGACY_CONFIG.vehicle_pass_times


@dataclass(frozen=True)
class SignalTimingConfig:
    min_green: int = LEGACY_CONFIG.default_minimum
    max_green: int = LEGACY_CONFIG.default_maximum
    yellow_time: int = LEGACY_CONFIG.default_yellow
    lanes_per_direction: int = LEGACY_CONFIG.lanes


def normalize_vehicle_counts(counts: dict[str, int]) -> dict[str, int]:
    normalized = {key: 0 for key in VEHICLE_PASSING_TIME}
    for key, value in counts.items():
        if key in normalized:
            normalized[key] = max(0, int(value))
    return normalized


def weighted_load(counts: dict[str, int]) -> float:
    normalized = normalize_vehicle_counts(counts)
    return sum(
        normalized[vehicle] * VEHICLE_PASSING_TIME[vehicle]
        for vehicle in VEHICLE_PASSING_TIME
    )


def compute_green_time(
    counts: dict[str, int],
    config: SignalTimingConfig,
) -> tuple[int, float]:
    load = weighted_load(counts)
    raw_green_time = ceil(load / (config.lanes_per_direction + 1))
    green_time = min(config.max_green, max(config.min_green, raw_green_time))
    return green_time, load


def build_signal_plan(
    direction_counts: dict[str, dict[str, int]],
    config: SignalTimingConfig,
) -> list[dict[str, float | int | str]]:
    plan: list[dict[str, float | int | str]] = []
    red_offset = 0
    for direction, counts in direction_counts.items():
        green_time, load = compute_green_time(counts, config)
        plan.append(
            {
                "direction": direction,
                "green_time": green_time,
                "yellow_time": config.yellow_time,
                "red_time_before_green": red_offset,
                "weighted_load": round(load, 2),
                "total_vehicles": sum(normalize_vehicle_counts(counts).values()),
            }
        )
        red_offset += green_time + config.yellow_time
    return plan


def simulate_signal_cycles(
    direction_counts: dict[str, dict[str, int]],
    config: SignalTimingConfig,
    cycles: int = 2,
) -> list[dict[str, float | int | str]]:
    plan = build_signal_plan(direction_counts, config)
    remaining_load = {
        item["direction"]: weighted_load(direction_counts[item["direction"]])
        for item in plan
    }
    timeline: list[dict[str, float | int | str]] = []
    second_counter = 0
    service_rate = config.lanes_per_direction + 1

    for cycle_index in range(1, cycles + 1):
        for item in plan:
            direction = str(item["direction"])
            green_time = int(item["green_time"])
            yellow_time = int(item["yellow_time"])

            for green_second in range(1, green_time + 1):
                second_counter += 1
                remaining_load[direction] = max(0.0, remaining_load[direction] - service_rate)
                timeline.append(
                    {
                        "cycle": cycle_index,
                        "global_second": second_counter,
                        "direction": direction,
                        "signal_state": "GREEN",
                        "state_second": green_second,
                        "remaining_weighted_load": round(remaining_load[direction], 2),
                        "service_rate": service_rate,
                    }
                )

            for yellow_second in range(1, yellow_time + 1):
                second_counter += 1
                timeline.append(
                    {
                        "cycle": cycle_index,
                        "global_second": second_counter,
                        "direction": direction,
                        "signal_state": "YELLOW",
                        "state_second": yellow_second,
                        "remaining_weighted_load": round(remaining_load[direction], 2),
                        "service_rate": 0,
                    }
                )

    return timeline
