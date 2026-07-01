import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


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
            value = abs(value) / DEFAULT_NORMALIZATION[name]
            value = min(max(value, 0.0), 1.0)
        elif name in ("thw_front", "thw_rear"):
            value = _thw_to_feature(value)
        else:
            value = 1.0 if value else 0.0
        values.append(value)
    return np.asarray(values, dtype=float)


def aggregate_step_features(step_features: Iterable[np.ndarray]) -> np.ndarray:
    features = [np.asarray(f, dtype=float) for f in step_features if f is not None]
    if not features:
        return np.zeros(len(FEATURE_NAMES), dtype=float)
    return np.mean(features, axis=0)


class EnvFeatureTracker:
    """Extract HAD-Gen-style 10D features for a highway_env vehicle."""

    def __init__(self, lane_width: float = 4.0):
        self.lane_width = lane_width
        self.reset()

    def reset(self):
        self.last_speed = None
        self.last_x = None
        self.last_y = None
        self.last_lat_speed = 0.0
        self.last_long_acc = 0.0
        self.last_center_dev = None

    def observe(self, env: Any, vehicle: Any) -> np.ndarray:
        dt = 1.0 / float(env.config.get("policy_frequency", 5))
        x = float(vehicle.position[0])
        y = float(vehicle.position[1])
        speed = float(vehicle.speed)

        long_acc = 0.0 if self.last_speed is None else (speed - self.last_speed) / dt
        long_jerk = (
            0.0 if self.last_speed is None else (long_acc - self.last_long_acc) / dt
        )

        lat_speed = 0.0 if self.last_y is None else (y - self.last_y) / dt
        lat_acc = (
            0.0 if self.last_y is None else (lat_speed - self.last_lat_speed) / dt
        )

        thw_front, thw_rear = self._thw(env, vehicle)
        center_dev = self._centerline_deviation(vehicle)
        lane_dev_rate = (
            0.0
            if self.last_center_dev is None
            else abs(center_dev - self.last_center_dev) / dt
        )
        left_available, right_available = self._adjacent_lane_availability(env, vehicle)

        self.last_speed = speed
        self.last_x = x
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
        lane = None
        try:
            lane = vehicle.road.network.get_lane(vehicle.lane_index)
        except Exception:
            lane = None

        if lane is not None:
            try:
                _, lateral = lane.local_coordinates(vehicle.position)
                return abs(float(lateral))
            except Exception:
                pass

        y = float(vehicle.position[1])
        nearest_center = round(y / self.lane_width) * self.lane_width
        return abs(y - nearest_center)

    @staticmethod
    def _thw(env: Any, vehicle: Any) -> Tuple[float, float]:
        front_vehicle, rear_vehicle = env.road.surrounding_vehicles(vehicle)

        thw_front = math.inf
        if front_vehicle is not None:
            try:
                headway = float(vehicle.lane_distance_to(front_vehicle))
                if headway > 0:
                    thw_front = headway / max(float(vehicle.speed), 1e-6)
            except Exception:
                pass

        thw_rear = math.inf
        if rear_vehicle is not None:
            try:
                rear_gap = float(rear_vehicle.lane_distance_to(vehicle))
                if rear_gap > 0:
                    thw_rear = rear_gap / max(float(rear_vehicle.speed), 1e-6)
            except Exception:
                pass

        return thw_front, thw_rear

    @staticmethod
    def _adjacent_lane_availability(env: Any, vehicle: Any) -> Tuple[bool, bool]:
        try:
            side_lanes = env.road.network.side_lanes(vehicle.lane_index)
        except Exception:
            side_lanes = []

        current_lane = vehicle.lane_index[-1] if isinstance(vehicle.lane_index, tuple) else 0
        left_available = any(lane[-1] < current_lane for lane in side_lanes)
        right_available = any(lane[-1] > current_lane for lane in side_lanes)
        return left_available, right_available


def extract_expert_trajectory_features(
    csv_path: str,
    aggressive_only: bool = True,
    episode_col: str = "episode",
    agent_col: str = "agent_id",
    time_col: str = "time",
    style_col: str = "style",
    lane_width: float = 4.0,
) -> Tuple[List[np.ndarray], pd.DataFrame]:
    """
    Load human trajectory CSV and return trajectory-level 10D features.

    Expected columns are flexible. Recommended:
    episode, agent_id, time, x, y, speed or vx/vy, optional style.
    NGSIM aliases such as Vehicle_ID, Global_Time, Local_X, Local_Y,
    v_Vel, Space_Headway, and Time_Headway are also supported.
    If style is absent, aggressive trajectories are inferred by clustering.
    """
    df = pd.read_csv(csv_path)
    df = _ensure_columns(df, episode_col, agent_col, time_col)

    grouped = df.groupby([episode_col, agent_col], sort=False)
    rows = []
    features = []
    for (episode_id, agent_id), traj in grouped:
        traj = traj.sort_values(time_col)
        traj_features = _trajectory_feature(traj, time_col, lane_width)
        summary = _trajectory_summary(traj, time_col)
        summary.update({"episode": episode_id, "agent_id": agent_id})
        summary["feature_vector"] = traj_features
        if style_col in traj.columns:
            summary["style"] = str(traj[style_col].iloc[0])
        rows.append(summary)

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return [], summary_df

    if aggressive_only:
        summary_df = mark_aggressive_trajectories(summary_df)
        summary_df = summary_df[summary_df["is_aggressive"]]

    for vector in summary_df["feature_vector"]:
        features.append(np.asarray(vector, dtype=float))

    return features, summary_df.drop(columns=["feature_vector"], errors="ignore")


def mark_aggressive_trajectories(summary_df: pd.DataFrame) -> pd.DataFrame:
    if "style" in summary_df.columns:
        result = summary_df.copy()
        result["is_aggressive"] = (
            result["style"].astype(str).str.lower() == "aggressive"
        )
        return result

    cluster_cols = [
        "mean_speed",
        "max_speed",
        "mean_abs_acc",
        "mean_abs_jerk",
        "lane_change_count",
    ]
    values = summary_df[cluster_cols].fillna(0.0).values

    if len(summary_df) < 3:
        score = _aggressive_score(summary_df)
        threshold = np.percentile(score, 70)
        result = summary_df.copy()
        result["is_aggressive"] = score >= threshold
        return result

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        scaled = StandardScaler().fit_transform(values)
        labels = KMeans(n_clusters=3, random_state=0, n_init=10).fit_predict(scaled)
        result = summary_df.copy()
        result["cluster"] = labels
        cluster_scores = result.groupby("cluster").apply(_aggressive_score).groupby(level=0).mean()
        aggressive_cluster = int(cluster_scores.idxmax())
        result["is_aggressive"] = result["cluster"] == aggressive_cluster
        return result
    except Exception:
        score = _aggressive_score(summary_df)
        threshold = np.percentile(score, 70)
        result = summary_df.copy()
        result["is_aggressive"] = score >= threshold
        return result


def _ensure_columns(
    df: pd.DataFrame, episode_col: str, agent_col: str, time_col: str
) -> pd.DataFrame:
    result = df.copy()
    if episode_col not in result.columns:
        result[episode_col] = 0
    if agent_col not in result.columns:
        result[agent_col] = "human"
    if time_col not in result.columns:
        result[time_col] = result.groupby([episode_col, agent_col]).cumcount()
    return result


def _trajectory_feature(traj: pd.DataFrame, time_col: str, lane_width: float) -> np.ndarray:
    step_features = []
    speeds = _speed_series(traj)
    xs = _column_or_default(traj, ["x", "pos_x", "hdv_x", "Local_X", "Global_X"], 0.0)
    ys = _column_or_default(traj, ["y", "pos_y", "hdv_y", "Local_Y", "Global_Y"], 0.0)
    times = traj[time_col].astype(float).values
    dts = np.diff(times, prepend=times[0])
    dts[dts <= 1e-6] = np.median(dts[dts > 1e-6]) if np.any(dts > 1e-6) else 0.2

    long_acc = np.diff(speeds, prepend=speeds[0]) / dts
    lat_speed = np.diff(ys, prepend=ys[0]) / dts
    lat_acc = np.diff(lat_speed, prepend=lat_speed[0]) / dts
    long_jerk = np.diff(long_acc, prepend=long_acc[0]) / dts

    front_gap = _optional_column(
        traj, ["front_gap", "headway", "space_headway", "Space_Headway"]
    )
    rear_gap = _optional_column(traj, ["rear_gap", "rear_headway"])
    thw_front = front_gap / np.maximum(speeds, 1e-6) if front_gap is not None else np.inf
    thw_rear = rear_gap / np.maximum(speeds, 1e-6) if rear_gap is not None else np.inf

    center_dev = np.abs(ys - np.round(ys / lane_width) * lane_width)
    lane_dev_rate = np.abs(np.diff(center_dev, prepend=center_dev[0])) / dts

    for i in range(len(traj)):
        step_features.append(
            normalize_feature_dict(
                {
                    "speed": speeds[i],
                    "long_acc": long_acc[i],
                    "lat_acc": lat_acc[i],
                    "long_jerk": long_jerk[i],
                    "thw_front": thw_front[i] if np.ndim(thw_front) else thw_front,
                    "thw_rear": thw_rear[i] if np.ndim(thw_rear) else thw_rear,
                    "centerline_dev": center_dev[i],
                    "lane_dev_rate": lane_dev_rate[i],
                    "left_lane_available": 1.0,
                    "right_lane_available": 1.0,
                }
            )
        )
    return aggregate_step_features(step_features)


def _trajectory_summary(traj: pd.DataFrame, time_col: str) -> Dict[str, float]:
    speeds = _speed_series(traj)
    times = traj[time_col].astype(float).values
    dts = np.diff(times, prepend=times[0])
    dts[dts <= 1e-6] = np.median(dts[dts > 1e-6]) if np.any(dts > 1e-6) else 0.2
    acc = np.diff(speeds, prepend=speeds[0]) / dts
    jerk = np.diff(acc, prepend=acc[0]) / dts
    ys = _column_or_default(traj, ["y", "pos_y", "hdv_y", "Local_Y", "Global_Y"], 0.0)
    lane_change_count = int(np.sum(np.abs(np.diff(np.round(ys / 4.0))) > 0))
    return {
        "mean_speed": float(np.mean(speeds)),
        "max_speed": float(np.max(speeds)),
        "mean_abs_acc": float(np.mean(np.abs(acc))),
        "mean_abs_jerk": float(np.mean(np.abs(jerk))),
        "lane_change_count": lane_change_count,
    }


def _aggressive_score(df: pd.DataFrame) -> pd.Series:
    return (
        df["mean_speed"].rank(pct=True)
        + df["max_speed"].rank(pct=True)
        + df["mean_abs_acc"].rank(pct=True)
        + df["lane_change_count"].rank(pct=True)
        + 0.5 * df["mean_abs_jerk"].rank(pct=True)
    )


def _speed_series(traj: pd.DataFrame) -> np.ndarray:
    speed = _optional_column(traj, ["speed", "v", "velocity", "hdv_speed", "v_Vel"])
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
            return traj[name].astype(float).values
    return None


def _column_or_default(traj: pd.DataFrame, names: List[str], default: float) -> np.ndarray:
    values = _optional_column(traj, names)
    if values is None:
        return np.full(len(traj), default, dtype=float)
    return np.asarray(values, dtype=float)


def _thw_to_feature(thw: float) -> float:
    if thw is None or not np.isfinite(thw) or thw <= 0:
        return 1.0
    return float(np.exp(-1.0 / max(thw, 1e-6)))
