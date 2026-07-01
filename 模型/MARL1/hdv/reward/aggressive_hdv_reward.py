import json
import math
from typing import Any, Dict, Optional

import numpy as np

from ego.reward.base_reward import BaseReward, RewardInfo
from hdv.irl.features import EnvFeatureTracker, FEATURE_NAMES


class AggressiveHDVReward(BaseReward):
    """
    Aggressive human-like reward for fine-tuning PPOmodel889.

    This class is intentionally linear and feature-based so an IRL result from
    HAD-Gen can be plugged in as a JSON weight file. If the learned reward is a
    neural network, replace compute() with that model's forward pass while
    keeping the same wrapper/training script.
    """

    DEFAULT_WEIGHTS = {
        "crash_penalty": -100.0,
        "progress": 0.08,
        "speed": 1.0,
        "target_speed": 27.0,
        "low_speed_penalty": -0.5,
        "min_speed": 8.0,
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

    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 weights_path: Optional[str] = None):
        super().__init__(name="AggressiveHDVReward")
        self.weights = dict(self.DEFAULT_WEIGHTS)
        self.irl_theta = None
        self.feature_tracker = EnvFeatureTracker()
        if weights_path:
            loaded_weights = self._load_weights(weights_path)
            if all(name in loaded_weights for name in FEATURE_NAMES):
                self.irl_theta = np.asarray(
                    [loaded_weights[name] for name in FEATURE_NAMES],
                    dtype=float,
                )
            else:
                self.weights.update(loaded_weights)
        if weights:
            if all(name in weights for name in FEATURE_NAMES):
                self.irl_theta = np.asarray(
                    [weights[name] for name in FEATURE_NAMES],
                    dtype=float,
                )
            else:
                self.weights.update(weights)
        self.reset()

    def reset(self):
        self.last_x = None
        self.last_speed = None
        self.last_accel = 0.0
        self.last_lane_index = None
        self.feature_tracker.reset()

    def compute(
        self,
        env: Any,
        vehicle: Any = None,
        action: Optional[Any] = None,
        done: bool = False,
        info: Optional[Dict[str, Any]] = None,
    ) -> RewardInfo:
        if vehicle is None:
            return RewardInfo()

        if self.irl_theta is not None:
            feature_vector = self.feature_tracker.observe(env, vehicle)
            components = {
                name: float(weight * feature)
                for name, weight, feature in zip(FEATURE_NAMES, self.irl_theta, feature_vector)
            }
            safety = self.weights["crash_penalty"] if vehicle.crashed else 0.0
            components["crash"] = safety
            reward = float(np.dot(self.irl_theta, feature_vector) + safety)
            diagnostics = {
                "irl_feature_" + name: float(value)
                for name, value in zip(FEATURE_NAMES, feature_vector)
            }
            return RewardInfo(reward=reward, components=components, info=diagnostics)

        w = self.weights
        x = float(vehicle.position[0])
        speed = float(vehicle.speed)
        dt = 1.0 / float(env.config.get("policy_frequency", 5))

        progress = 0.0 if self.last_x is None else x - self.last_x
        accel = 0.0 if self.last_speed is None else (speed - self.last_speed) / dt
        jerk = (accel - self.last_accel) / dt if self.last_speed is not None else 0.0
        lane_changed = int(
            self.last_lane_index is not None and vehicle.lane_index != self.last_lane_index
        )

        headway, thw, ttc = self._front_metrics(env, vehicle)

        components = {}
        components["crash"] = w["crash_penalty"] if vehicle.crashed else 0.0
        components["progress"] = w["progress"] * max(progress, 0.0)
        components["speed"] = w["speed"] * min(speed / max(w["target_speed"], 1e-6), 1.2)
        components["low_speed"] = (
            w["low_speed_penalty"] if speed < w["min_speed"] else 0.0
        )

        if np.isfinite(thw):
            components["thw"] = w["thw"] * abs(thw - w["target_thw"])
            components["unsafe_thw"] = (
                w["unsafe_thw_penalty"]
                if thw < w["min_safe_thw"]
                else 0.0
            )
        else:
            components["thw"] = 0.0
            components["unsafe_thw"] = 0.0

        if np.isfinite(ttc) and ttc < w["min_safe_ttc"]:
            components["ttc"] = w["ttc_penalty"] * (
                w["min_safe_ttc"] - max(ttc, 0.0)
            )
        else:
            components["ttc"] = 0.0

        components["accel"] = w["accel_penalty"] * abs(accel)
        components["jerk"] = w["jerk_penalty"] * abs(jerk)
        components["lane_change"] = w["lane_change"] * lane_changed

        reward = float(sum(components.values()))

        self.last_x = x
        self.last_speed = speed
        self.last_accel = accel
        self.last_lane_index = vehicle.lane_index

        diagnostics = {
            "hdv_speed": speed,
            "hdv_accel": accel,
            "hdv_jerk": jerk,
            "hdv_headway": headway,
            "hdv_thw": thw,
            "hdv_ttc": ttc,
            "hdv_lane_changed": lane_changed,
        }
        return RewardInfo(reward=reward, components=components, info=diagnostics)

    @staticmethod
    def _load_weights(path: str) -> Dict[str, float]:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {str(k): float(v) for k, v in raw.items()}

    @staticmethod
    def _front_metrics(env: Any, vehicle: Any):
        front_vehicle, _ = env.road.surrounding_vehicles(vehicle)
        if front_vehicle is None:
            return math.nan, math.nan, math.nan

        try:
            headway = float(vehicle.lane_distance_to(front_vehicle))
        except Exception:
            return math.nan, math.nan, float(front_vehicle.speed)

        if not np.isfinite(headway) or headway <= 0:
            return math.nan, math.nan, float(front_vehicle.speed)

        thw = headway / max(float(vehicle.speed), 1e-6)
        closing_speed = float(vehicle.speed) - float(front_vehicle.speed)
        ttc = headway / closing_speed if closing_speed > 1e-6 else math.nan
        return headway, thw, ttc
