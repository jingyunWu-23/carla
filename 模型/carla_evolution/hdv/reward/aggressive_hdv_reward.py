from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from carla_evolution.hdv.irl.features import CarlaHDVFeatureTracker, FEATURE_NAMES


@dataclass
class HDVRewardResult:
    reward: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    info: Dict[str, float] = field(default_factory=dict)


class AggressiveHDVReward:
    """CARLA HDV reward learned from NGSIM aggressive-style IRL.

    If a weights file contains all 10 feature names from ``FEATURE_NAMES``, the
    reward is ``theta dot feature`` plus a crash penalty. Otherwise the file is
    treated as hand-written fallback weights.
    """

    DEFAULT_WEIGHTS = {
        "crash_penalty": -100.0,
        "progress": 0.08,
        "speed": 1.0,
        "target_speed": 24.0,
        "low_speed_penalty": -0.5,
        "min_speed": 5.0,
        "thw": -0.35,
        "target_thw": 1.2,
        "unsafe_thw_penalty": -4.0,
        "min_safe_thw": 0.7,
        "ttc_penalty": -5.0,
        "min_safe_ttc": 1.5,
        "accel_penalty": -0.04,
        "jerk_penalty": -0.02,
        "lane_change": 0.10,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        weights_path: Optional[str] = None,
        lane_width: float = 4.0,
    ):
        self.weights = dict(self.DEFAULT_WEIGHTS)
        self.irl_theta = None
        self.feature_tracker = CarlaHDVFeatureTracker(lane_width=lane_width)
        if weights_path:
            self._apply_weights(self._load_weights(weights_path))
        if weights:
            self._apply_weights(weights)
        self.reset()

    def reset(self):
        self.last_x = None
        self.last_speed = None
        self.last_accel = 0.0
        self.last_lane_id = None
        self.feature_tracker.reset()

    def compute(
        self,
        scenario: Any,
        vehicle: Any,
        action: Optional[Any] = None,
        done: bool = False,
        info: Optional[Dict[str, Any]] = None,
    ) -> HDVRewardResult:
        if vehicle is None:
            return HDVRewardResult()

        if self.irl_theta is not None:
            feature_vector = self.feature_tracker.observe(scenario, vehicle)
            components = {
                name: float(weight * feature)
                for name, weight, feature in zip(FEATURE_NAMES, self.irl_theta, feature_vector)
            }
            crash = self.weights["crash_penalty"] if getattr(vehicle, "crashed", False) else 0.0
            components["crash"] = float(crash)
            reward = float(np.dot(self.irl_theta, feature_vector) + crash)
            diagnostics = {
                "irl_feature_" + name: float(value)
                for name, value in zip(FEATURE_NAMES, feature_vector)
            }
            return HDVRewardResult(reward=reward, components=components, info=diagnostics)

        w = self.weights
        x = float(getattr(vehicle, "route_s", None) or getattr(vehicle, "x", vehicle.position[0]))
        speed = float(getattr(vehicle, "speed", 0.0))
        dt = float(getattr(scenario, "dt", 0.2) or 0.2)
        progress = 0.0 if self.last_x is None else x - self.last_x
        accel = 0.0 if self.last_speed is None else (speed - self.last_speed) / dt
        jerk = 0.0 if self.last_speed is None else (accel - self.last_accel) / dt
        lane_id = int(getattr(vehicle, "lane_id", 0))
        lane_changed = int(self.last_lane_id is not None and lane_id != self.last_lane_id)
        headway, thw, ttc = self._front_metrics(scenario, vehicle)

        components = {
            "crash": w["crash_penalty"] if getattr(vehicle, "crashed", False) else 0.0,
            "progress": w["progress"] * max(progress, 0.0),
            "speed": w["speed"] * min(speed / max(w["target_speed"], 1e-6), 1.2),
            "low_speed": w["low_speed_penalty"] if speed < w["min_speed"] else 0.0,
            "accel": w["accel_penalty"] * abs(accel),
            "jerk": w["jerk_penalty"] * abs(jerk),
            "lane_change": w["lane_change"] * lane_changed,
        }
        if np.isfinite(thw):
            components["thw"] = w["thw"] * abs(thw - w["target_thw"])
            components["unsafe_thw"] = w["unsafe_thw_penalty"] if thw < w["min_safe_thw"] else 0.0
        else:
            components["thw"] = 0.0
            components["unsafe_thw"] = 0.0
        if np.isfinite(ttc) and ttc < w["min_safe_ttc"]:
            components["ttc"] = w["ttc_penalty"] * (w["min_safe_ttc"] - max(ttc, 0.0))
        else:
            components["ttc"] = 0.0

        self.last_x = x
        self.last_speed = speed
        self.last_accel = accel
        self.last_lane_id = lane_id

        diagnostics = {
            "hdv_speed": speed,
            "hdv_accel": accel,
            "hdv_jerk": jerk,
            "hdv_headway": headway,
            "hdv_thw": thw,
            "hdv_ttc": ttc,
            "hdv_lane_changed": float(lane_changed),
        }
        return HDVRewardResult(reward=float(sum(components.values())), components=components, info=diagnostics)

    def _apply_weights(self, weights: Dict[str, float]):
        if all(name in weights for name in FEATURE_NAMES):
            self.irl_theta = np.asarray([weights[name] for name in FEATURE_NAMES], dtype=float)
        else:
            self.weights.update({str(key): float(value) for key, value in weights.items()})

    @staticmethod
    def _load_weights(path: str) -> Dict[str, float]:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {str(key): float(value) for key, value in raw.items()}

    @staticmethod
    def _front_metrics(scenario: Any, vehicle: Any):
        try:
            front_vehicle, _ = scenario.road.surrounding_vehicles(vehicle)
        except Exception:
            front_vehicle = None
        if front_vehicle is None:
            return math.nan, math.nan, math.nan
        try:
            headway = float(vehicle.lane_distance_to(front_vehicle))
        except Exception:
            return math.nan, math.nan, math.nan
        if not np.isfinite(headway) or headway <= 0.0:
            return math.nan, math.nan, math.nan
        thw = headway / max(float(vehicle.speed), 1e-6)
        closing_speed = float(vehicle.speed) - float(front_vehicle.speed)
        ttc = headway / closing_speed if closing_speed > 1e-6 else math.nan
        return headway, thw, ttc
