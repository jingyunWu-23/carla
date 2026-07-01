#!/usr/bin/env python3
"""Visualize 100-round CARLA and highway co-training results.

The CARLA runs are often split across multiple result directories. By default
this script reads every CARLA ``joint_round_log.csv`` under
``模型/carla_evolution/results`` and keeps the newest record for duplicate
rounds. Highway results are aggregated from per-window ``joint_train_log.csv``.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CARLA_GLOB = "模型/carla_evolution/results/joint_*/joint_round_log.csv"
DEFAULT_HIGHWAY_LOG = "模型/MARL1/joint_results/joint_train_log.csv"
DEFAULT_OUTPUT_DIR = "模型/carla_evolution/results/visualizations"


def _to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _rolling(values: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return values
    return values.rolling(window=window, min_periods=1).mean()


def load_carla_rounds(patterns: list[str], max_round: int) -> pd.DataFrame:
    paths: list[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(matches)
        elif os.path.isfile(pattern):
            paths.append(pattern)

    frames: list[pd.DataFrame] = []
    for path in sorted(set(paths), key=lambda item: os.path.getmtime(item)):
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty or "round" not in df.columns:
            continue
        df = _to_numeric(
            df,
            [
                "round",
                "ego_avg_reward",
                "ego_crash_rate",
                "ego_road_completion_rate",
                "ego_avg_speed",
                "adv_avg_reward",
                "adv_crash_rate",
                "adv_road_completion_rate",
            ],
        )
        df = df.dropna(subset=["round"])
        df["round"] = df["round"].astype(int)
        df["_source"] = path
        df["_source_mtime"] = os.path.getmtime(path)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No CARLA round logs matched: {patterns}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[(combined["round"] >= 1) & (combined["round"] <= max_round)]
    combined = combined.sort_values(["round", "_source_mtime"]).drop_duplicates("round", keep="last")
    combined = combined.sort_values("round")

    return pd.DataFrame(
        {
            "round": combined["round"],
            "env": "CARLA",
            "ego_reward": combined.get("ego_avg_reward"),
            "ego_crash_rate": combined.get("ego_crash_rate"),
            "ego_completion": combined.get("ego_road_completion_rate"),
            "ego_avg_speed": combined.get("ego_avg_speed"),
            "adv_reward": combined.get("adv_avg_reward"),
            "adv_crash_rate": combined.get("adv_crash_rate"),
            "adv_completion": combined.get("adv_road_completion_rate"),
            "source": combined["_source"],
        }
    )


def load_highway_rounds(path: str, max_round: int, aggregation: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _to_numeric(
        df,
        [
            "round",
            "adversarial_reward",
            "ego_reward",
            "ego_crash_rate",
            "ego_road_completion_rate",
            "ego_avg_speed",
            "adversarial_crash_rate",
        ],
    )
    df = df.dropna(subset=["round"])
    df["round"] = df["round"].astype(int)
    df = df[(df["round"] >= 1) & (df["round"] <= max_round)]

    rows: list[dict[str, float | int | str | None]] = []
    for round_no, group in df.groupby("round", sort=True):
        ego = group[group["phase"] == "ego"] if "phase" in group.columns else group
        adv = group[group["phase"].isin(["adversarial", "adv"])] if "phase" in group.columns else group

        if aggregation == "last":
            ego_row = ego.tail(1)
            adv_row = adv.tail(1)
            value = lambda frame, column: float(frame[column].iloc[-1]) if not frame.empty and column in frame else np.nan
        else:
            value = lambda frame, column: float(frame[column].mean()) if not frame.empty and column in frame else np.nan

        rows.append(
            {
                "round": int(round_no),
                "env": "Highway",
                "ego_reward": value(ego, "ego_reward"),
                "ego_crash_rate": value(ego, "ego_crash_rate"),
                "ego_completion": value(ego, "ego_road_completion_rate"),
                "ego_avg_speed": value(ego, "ego_avg_speed"),
                "adv_reward": value(adv, "adversarial_reward"),
                "adv_crash_rate": value(adv, "adversarial_crash_rate"),
                "adv_completion": np.nan,
                "source": path,
            }
        )

    return pd.DataFrame(rows)


def plot_metric(ax, data: pd.DataFrame, metric: str, title: str, ylabel: str, smooth_window: int) -> None:
    colors = {"CARLA": "#1f77b4", "Highway": "#ff7f0e"}
    for env, env_df in data.groupby("env", sort=False):
        env_df = env_df.sort_values("round")
        ax.plot(
            env_df["round"],
            env_df[metric],
            color=colors.get(env, None),
            alpha=0.25,
            linewidth=1.0,
        )
        ax.plot(
            env_df["round"],
            _rolling(env_df[metric], smooth_window),
            label=env,
            color=colors.get(env, None),
            linewidth=2.0,
        )
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()


def plot_results(data: pd.DataFrame, output_dir: Path, smooth_window: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    metrics = [
        ("ego_reward", "Ego Reward", "reward"),
        ("ego_crash_rate", "Ego Crash Rate", "rate"),
        ("ego_completion", "Ego Road Completion", "completion"),
        ("ego_avg_speed", "Ego Average Speed", "speed"),
        ("adv_reward", "Adversarial Reward", "reward"),
        ("adv_crash_rate", "Adversarial Crash Rate", "rate"),
    ]
    for ax, (metric, title, ylabel) in zip(axes.ravel(), metrics):
        plot_metric(ax, data, metric, title, ylabel, smooth_window)

    fig.suptitle(f"CARLA vs Highway Co-training Results (1-100 rounds, smooth={smooth_window})", fontsize=14)
    output_path = output_dir / "carla_highway_100_rounds.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_summary(data: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = output_dir / "carla_highway_100_rounds_merged.csv"
    summary_path = output_dir / "carla_highway_100_rounds_summary.csv"

    data.sort_values(["env", "round"]).to_csv(merged_path, index=False)

    summary = (
        data.groupby("env")
        .agg(
            rounds=("round", "count"),
            first_round=("round", "min"),
            last_round=("round", "max"),
            ego_reward_mean=("ego_reward", "mean"),
            ego_reward_last=("ego_reward", "last"),
            ego_crash_rate_mean=("ego_crash_rate", "mean"),
            ego_crash_rate_last=("ego_crash_rate", "last"),
            ego_completion_mean=("ego_completion", "mean"),
            ego_completion_last=("ego_completion", "last"),
            ego_avg_speed_mean=("ego_avg_speed", "mean"),
            ego_avg_speed_last=("ego_avg_speed", "last"),
            adv_reward_mean=("adv_reward", "mean"),
            adv_reward_last=("adv_reward", "last"),
            adv_crash_rate_mean=("adv_crash_rate", "mean"),
            adv_crash_rate_last=("adv_crash_rate", "last"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)
    return merged_path, summary_path


def print_coverage(name: str, df: pd.DataFrame, max_round: int) -> None:
    rounds = set(int(item) for item in df["round"].dropna())
    missing = [round_no for round_no in range(1, max_round + 1) if round_no not in rounds]
    print(f"{name}: {len(rounds)} rounds, range={min(rounds) if rounds else '-'}-{max(rounds) if rounds else '-'}")
    if missing:
        preview = ", ".join(str(item) for item in missing[:20])
        suffix = "..." if len(missing) > 20 else ""
        print(f"{name}: missing rounds: {preview}{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--carla-round-log",
        action="append",
        default=None,
        help=(
            "CARLA joint_round_log.csv path or glob. Can be passed multiple times. "
            f"Default: {DEFAULT_CARLA_GLOB}"
        ),
    )
    parser.add_argument("--highway-log", default=DEFAULT_HIGHWAY_LOG, help="Highway joint_train_log.csv path.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for PNG and CSV outputs.")
    parser.add_argument("--max-round", type=int, default=100, help="Maximum round to visualize.")
    parser.add_argument("--smooth-window", type=int, default=5, help="Rolling mean window for thick lines.")
    parser.add_argument(
        "--highway-aggregation",
        choices=["mean", "last"],
        default="mean",
        help="Aggregate highway per-window records into round-level values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    carla_patterns = args.carla_round_log or [DEFAULT_CARLA_GLOB]
    output_dir = Path(args.output_dir)

    carla = load_carla_rounds(carla_patterns, args.max_round)
    highway = load_highway_rounds(args.highway_log, args.max_round, args.highway_aggregation)
    data = pd.concat([carla, highway], ignore_index=True)

    image_path = plot_results(data, output_dir, args.smooth_window)
    merged_path, summary_path = save_summary(data, output_dir)

    print_coverage("CARLA", carla, args.max_round)
    print_coverage("Highway", highway, args.max_round)
    print(f"Saved figure: {image_path}")
    print(f"Saved merged data: {merged_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
