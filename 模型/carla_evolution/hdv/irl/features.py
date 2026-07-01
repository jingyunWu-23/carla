from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from carla_evolution.envs.vehicle_state import longitudinal_gap


FEATURE_NAMES = [
    "speed",
    "long_acc",
    "lat_acc",
    "long_jerk",
    "thw_front",
    "thw_rear",
    "centerline_dev",
    "lane_dev_rate",
    "left_lane_available",
    "right_lane_available",
]


DEFAULT_NORMALIZATION = {
    "speed": 35.0,
    "long_acc": 5.0,
    "lat_acc": 3.0,
    "long_jerk": 10.0,
    "centerline_dev": 4.0,
    "lane_dev_rate": 4.0,
}


def normalize_feature_dict(features: Dict[str, float]) -> np.ndarray:
    values = []
    for name in FEATURE_NAMES:
        value = features.get(name, 0.0)
        if value is None or not np.isfinite(value):
            value = 0.0

        if name in DEFAULT_NORMALIZATION:
            value = abs(float(value)) / DEFAULT_NORMALIZATION[name]
            value = min(max(value, 0.0), 1.0)
        elif name in ("thw_front", "thw_rear"):
            value = _thw_to_feature(float(value))
        else:
            value = 1.0 if value else 0.0
        values.append(value)
    return np.asarray(values, dtype=float)


def aggregate_step_features(step_features: Iterable[np.ndarray]) -> np.ndarray:
    features = [np.asarray(item, dtype=float) for item in step_features if item is not None]
    if not features:
        return np.zeros(len(FEATURE_NAMES), dtype=float)
    return np.mean(features, axis=0)


class CarlaHDVFeatureTracker:
    """Extract the 10D HAD-Gen-style feature vector from CARLA VehicleState."""

    def __init__(self, lane_width: float = 4.0):
        self.lane_width = float(lane_width)
        self.reset()

    def reset(self):
        self.last_speed = None
        self.last_y = None
        self.last_lat_speed = 0.0
        self.last_long_acc = 0.0
        self.last_center_dev = None

    def observe(self, scenario: Any, vehicle: Any) -> np.ndarray:
        dt = float(getattr(scenario, "dt", 0.2) or 0.2)
        y = float(getattr(vehicle, "y", vehicle.position[1]))
        speed = float(getattr(vehicle, "speed", 0.0))

        long_acc = 0.0 if self.last_speed is None else (speed - self.last_speed) / dt
        long_jerk = 0.0 if self.last_speed is None else (long_acc - self.last_long_acc) / dt
        lat_speed = 0.0 if self.last_y is None else (y - self.last_y) / dt
        lat_acc = 0.0 if self.last_y is None else (lat_speed - self.last_lat_speed) / dt
        thw_front, thw_rear = self._thw(scenario, vehicle)
        center_dev = self._centerline_deviation(vehicle)
        lane_dev_rate = 0.0 if self.last_center_dev is None else abs(center_dev - self.last_center_dev) / dt
        left_available, right_available = self._adjacent_lane_availability(scenario, vehicle)

        self.last_speed = speed
        self.last_y = y
        self.last_lat_speed = lat_speed
        self.last_long_acc = long_acc
        self.last_center_dev = center_dev

        return normalize_feature_dict(
            {
                "speed": speed,
                "long_acc": long_acc,
                "lat_acc": lat_acc,
                "long_jerk": long_jerk,
                "thw_front": thw_front,
                "thw_rear": thw_rear,
                "centerline_dev": center_dev,
                "lane_dev_rate": lane_dev_rate,
                "left_lane_available": left_available,
                "right_lane_available": right_available,
            }
        )

    def _centerline_deviation(self, vehicle: Any) -> float:
        lateral_offset = getattr(vehicle, "lateral_offset", None)
        if lateral_offset is not None:
            return abs(float(lateral_offset))
        y = float(getattr(vehicle, "y", vehicle.position[1]))
        nearest_center = round(y / self.lane_width) * self.lane_width
        return abs(y - nearest_center)

    @staticmethod
    def _thw(scenario: Any, vehicle: Any) -> Tuple[float, float]:
        try:
            front_vehicle, rear_vehicle = scenario.road.surrounding_vehicles(vehicle)
        except Exception:
            front_vehicle, rear_vehicle = None, None

        thw_front = math.inf
        if front_vehicle is not None:
            gap = longitudinal_gap(vehicle, front_vehicle)
            if gap > 0.0:
                thw_front = gap / max(float(vehicle.speed), 1e-6)

        thw_rear = math.inf
        if rear_vehicle is not None:
            rear_gap = longitudinal_gap(rear_vehicle, vehicle)
            if rear_gap > 0.0:
                thw_rear = rear_gap / max(float(rear_vehicle.speed), 1e-6)
        return thw_front, thw_rear

    @staticmethod
    def _adjacent_lane_availability(scenario: Any, vehicle: Any) -> Tuple[bool, bool]:
        lane_ids = [
            int(other.lane_id)
            for other in getattr(scenario, "vehicles", [])
            if int(getattr(other, "road_id", 0)) == int(getattr(vehicle, "road_id", 0))
            and int(getattr(other, "section_id", 0)) == int(getattr(vehicle, "section_id", 0))
        ]
        max_lane = max(lane_ids) if lane_ids else 3
        lane_id = int(getattr(vehicle, "lane_id", 0))
        return lane_id > 0, lane_id < max_lane


def extract_expert_trajectory_features(
    csv_path: str,
    aggressive_only: bool = True,
    target_style: str = None,
    episode_col: str = "episode",
    agent_col: str = "agent_id",
    time_col: str = "time",
    style_col: str = "style",
    lane_width: float = 4.0,
) -> Tuple[List[np.ndarray], pd.DataFrame]:
    df = pd.read_csv(csv_path)
    df = _ensure_columns(df, episode_col, agent_col, time_col)

    rows = []
    for (episode_id, agent_id), traj in df.groupby([episode_col, agent_col], sort=False):
        traj = traj.sort_values(time_col)
        vector = _trajectory_feature(traj, time_col, lane_width)
        summary = _trajectory_summary(traj, time_col, lane_width)
        summary.update({"episode": episode_id, "agent_id": agent_id, "feature_vector": vector})
        if style_col in traj.columns:
            summary["style"] = str(traj[style_col].iloc[0])
        rows.append(summary)

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return [], summary_df

    summary_df = mark_aggressive_trajectories(summary_df)
    if target_style is not None and str(target_style).lower() != "all":
        selected = summary_df[summary_df["driving_style"] == str(target_style).lower()]
    elif aggressive_only:
        selected = summary_df[summary_df["is_aggressive"]]
    else:
        selected = summary_df
    features = [np.asarray(item, dtype=float) for item in selected["feature_vector"]]
    return features, summary_df.drop(columns=["feature_vector"], errors="ignore")


def mark_aggressive_trajectories(summary_df: pd.DataFrame) -> pd.DataFrame:
    if "style" in summary_df.columns:
        result = summary_df.copy()
        result["driving_style"] = result["style"].astype(str).str.lower()
        result["is_aggressive"] = result["driving_style"] == "aggressive"
        return result

    cluster_cols = ["mean_speed", "max_speed", "mean_abs_acc", "mean_abs_jerk", "lane_change_count"]
    values = summary_df[cluster_cols].fillna(0.0).values
    result = summary_df.copy()
    if len(result) < 3:
        score = _aggressive_score(result)
        result["cluster"] = 0
        result["is_aggressive"] = score >= np.percentile(score, 70)
        return result

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        scaled = StandardScaler().fit_transform(values)
        result["cluster"] = KMeans(n_clusters=3, random_state=0, n_init=10).fit_predict(scaled)
    except Exception:
        scaled = _standard_scale(values)
        result["cluster"] = _numpy_kmeans(scaled, n_clusters=3, seed=0)

    score = _aggressive_score(result)
    cluster_scores = score.groupby(result["cluster"]).mean()
    sorted_clusters = list(cluster_scores.sort_values().index)
    conservative_cluster = int(sorted_clusters[0])
    aggressive_cluster = int(sorted_clusters[-1])
    neutral_cluster = int(sorted_clusters[len(sorted_clusters) // 2])
    style_by_cluster = {
        conservative_cluster: "conservative",
        neutral_cluster: "neutral",
        aggressive_cluster: "aggressive",
    }
    if len(sorted_clusters) == 1:
        style_by_cluster[aggressive_cluster] = "neutral"
    result["is_aggressive"] = result["cluster"] == aggressive_cluster
    result["driving_style"] = result["cluster"].map(style_by_cluster).fillna("neutral")
    return result


def _ensure_columns(df: pd.DataFrame, episode_col: str, agent_col: str, time_col: str) -> pd.DataFrame:
    result = df.copy()
    if episode_col not in result.columns:
        result[episode_col] = 0
    if agent_col not in result.columns:
        result[agent_col] = "human"
    if time_col not in result.columns:
        result[time_col] = result.groupby([episode_col, agent_col]).cumcount()
    return result


def _trajectory_feature(traj: pd.DataFrame, time_col: str, lane_width: float) -> np.ndarray:
    speeds = _speed_series(traj)
    ys = _column_or_default(traj, ["y", "pos_y", "Local_X", "Global_Y"], 0.0)
    times = traj[time_col].astype(float).values
    dts = _safe_dts(times)
    long_acc = np.diff(speeds, prepend=speeds[0]) / dts
    lat_speed = np.diff(ys, prepend=ys[0]) / dts
    lat_acc = np.diff(lat_speed, prepend=lat_speed[0]) / dts
    long_jerk = np.diff(long_acc, prepend=long_acc[0]) / dts
    front_gap = _optional_column(traj, ["front_gap", "headway", "Space_Headway", "space_headway"])
    rear_gap = _optional_column(traj, ["rear_gap", "rear_headway"])
    thw_front = front_gap / np.maximum(speeds, 1e-6) if front_gap is not None else np.full(len(traj), math.inf)
    thw_rear = rear_gap / np.maximum(speeds, 1e-6) if rear_gap is not None else np.full(len(traj), math.inf)
    center_dev = np.abs(ys - np.round(ys / lane_width) * lane_width)
    lane_dev_rate = np.abs(np.diff(center_dev, prepend=center_dev[0])) / dts
    lane_ids = _optional_column(traj, ["lane_id", "Lane_ID"])
    left_available = np.ones(len(traj), dtype=float)
    right_available = np.ones(len(traj), dtype=float)
    if lane_ids is not None:
        left_available = (lane_ids > np.nanmin(lane_ids)).astype(float)
        right_available = (lane_ids < np.nanmax(lane_ids)).astype(float)

    step_features = []
    for i in range(len(traj)):
        step_features.append(
            normalize_feature_dict(
                {
                    "speed": speeds[i],
                    "long_acc": long_acc[i],
                    "lat_acc": lat_acc[i],
                    "long_jerk": long_jerk[i],
                    "thw_front": thw_front[i],
                    "thw_rear": thw_rear[i],
                    "centerline_dev": center_dev[i],
                    "lane_dev_rate": lane_dev_rate[i],
                    "left_lane_available": left_available[i],
                    "right_lane_available": right_available[i],
                }
            )
        )
    return aggregate_step_features(step_features)


def _trajectory_summary(traj: pd.DataFrame, time_col: str, lane_width: float) -> Dict[str, float]:
    speeds = _speed_series(traj)
    times = traj[time_col].astype(float).values
    dts = _safe_dts(times)
    acc = np.diff(speeds, prepend=speeds[0]) / dts
    jerk = np.diff(acc, prepend=acc[0]) / dts
    ys = _column_or_default(traj, ["y", "pos_y", "Local_X", "Global_Y"], 0.0)
    lane_ids = _optional_column(traj, ["lane_id", "Lane_ID"])
    if lane_ids is None:
        lane_ids = np.round(ys / lane_width)
    lane_change_count = int(np.sum(np.abs(np.diff(lane_ids)) > 0))
    return {
        "mean_speed": float(np.mean(speeds)),
        "max_speed": float(np.max(speeds)),
        "mean_abs_acc": float(np.mean(np.abs(acc))),
        "mean_abs_jerk": float(np.mean(np.abs(jerk))),
        "lane_change_count": lane_change_count,
    }


def _safe_dts(times: np.ndarray) -> np.ndarray:
    dts = np.diff(times, prepend=times[0])
    fallback = np.median(dts[dts > 1e-6]) if np.any(dts > 1e-6) else 0.2
    dts[dts <= 1e-6] = fallback
    return dts


def _aggressive_score(df: pd.DataFrame) -> pd.Series:
    return (
        df["mean_speed"].rank(pct=True)
        + df["max_speed"].rank(pct=True)
        + df["mean_abs_acc"].rank(pct=True)
        + df["lane_change_count"].rank(pct=True)
        + 0.5 * df["mean_abs_jerk"].rank(pct=True)
    )


def _standard_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std[std < 1e-8] = 1.0
    return (values - mean) / std


def _numpy_kmeans(values: np.ndarray, n_clusters: int, seed: int = 0, n_iters: int = 100) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.zeros(0, dtype=int)
    n_clusters = int(min(max(n_clusters, 1), n))
    rng = np.random.RandomState(seed)
    centers = values[rng.choice(n, size=n_clusters, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(int(n_iters)):
        distances = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(n_clusters):
            members = values[labels == cluster]
            if len(members) > 0:
                centers[cluster] = members.mean(axis=0)
    return labels


def _speed_series(traj: pd.DataFrame) -> np.ndarray:
    speed = _optional_column(traj, ["speed", "v", "velocity", "v_Vel"])
    if speed is not None:
        return np.asarray(speed, dtype=float)
    vx = _optional_column(traj, ["vx", "v_x"])
    vy = _optional_column(traj, ["vy", "v_y"])
    if vx is not None and vy is not None:
        return np.sqrt(np.asarray(vx, dtype=float) ** 2 + np.asarray(vy, dtype=float) ** 2)
    return np.zeros(len(traj), dtype=float)


def _optional_column(traj: pd.DataFrame, names: List[str]):
    for name in names:
        if name in traj.columns:
            return pd.to_numeric(traj[name], errors="coerce").fillna(0.0).values
    return None


def _column_or_default(traj: pd.DataFrame, names: List[str], default: float) -> np.ndarray:
    values = _optional_column(traj, names)
    if values is None:
        return np.full(len(traj), default, dtype=float)
    return np.asarray(values, dtype=float)


def _thw_to_feature(thw: float) -> float:
    if thw is None or not np.isfinite(thw) or thw <= 0.0:
        return 1.0
    return float(np.exp(-1.0 / max(thw, 1e-6)))
