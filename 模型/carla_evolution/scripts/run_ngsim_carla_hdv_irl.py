#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_evolution.hdv.irl.candidate_generation import build_candidate_scenes
from carla_evolution.hdv.irl.features import FEATURE_NAMES, extract_expert_trajectory_features
from carla_evolution.hdv.irl.maxent import MaxEntIRL


DEFAULT_OUTPUT_DIR = "模型/carla_evolution/results/hdv_irl"
DEFAULT_EXPERT_CSV = "模型/carla_evolution/results/hdv_irl/ngsim_expert.csv"
DEFAULT_CONFIG = "模型/carla_evolution/configs/carla_0915.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn CARLA aggressive HDV reward from NGSIM and CARLA candidate rollouts.")
    parser.add_argument("--raw-ngsim-csv", default=None, help="Raw NGSIM CSV. If set, conversion runs first.")
    parser.add_argument("--expert-csv", default=DEFAULT_EXPERT_CSV, help="Converted compact NGSIM expert CSV.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--location", default="us-101", choices=["us-101", "i-80", "all"])
    parser.add_argument("--max-vehicles", type=int, default=500)
    parser.add_argument("--min-frames", type=int, default=80)
    parser.add_argument("--chunksize", type=int, default=300000)
    parser.add_argument("--cluster-only", action="store_true", help="Stop after NGSIM clustering.")
    parser.add_argument("--include-non-aggressive", action="store_true", help="Use all NGSIM trajectories for IRL.")
    parser.add_argument(
        "--target-style",
        default="aggressive",
        choices=["aggressive", "neutral", "conservative", "all"],
        help="Which NGSIM driving-style cluster is used as IRL expert data.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="CARLA evolution YAML config.")
    parser.add_argument("--backend", default="mock", choices=["mock", "carla"], help="Candidate rollout backend.")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seeds", default=",".join(str(i) for i in range(0, 600, 20)))
    parser.add_argument("--candidate-depth", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=25)
    parser.add_argument("--num-cav", type=int, default=3)
    parser.add_argument("--num-hdv", type=int, default=3)
    parser.add_argument("--num-background", type=int, default=0)
    parser.add_argument("--target-hdv-index", type=int, default=0)
    parser.add_argument("--ego-action", type=int, default=1)
    parser.add_argument("--adv-action", type=int, default=1)
    parser.add_argument("--other-hdv-action", type=int, default=1)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lam", type=float, default=0.01)
    return parser.parse_args()


def convert_ngsim(args: argparse.Namespace, expert_csv: Path) -> None:
    raw_path = Path(args.raw_ngsim_csv)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw NGSIM CSV not found: {raw_path}. "
            "Do not use /path/to/NGSIM.csv literally; pass the actual NGSIM CSV path."
        )
    cmd = [
        sys.executable,
        "模型/carla_evolution/hdv/irl/prepare_ngsim.py",
        "--input",
        str(raw_path),
        "--output",
        str(expert_csv),
        "--location",
        args.location,
        "--max-vehicles",
        str(args.max_vehicles),
        "--min-frames",
        str(args.min_frames),
        "--chunksize",
        str(args.chunksize),
    ]
    subprocess.run(cmd, check=True)


def save_cluster_outputs(output_dir: Path, summary: pd.DataFrame) -> tuple[Path, Path]:
    summary_path = output_dir / "ngsim_trajectory_cluster_summary.csv"
    stats_path = output_dir / "ngsim_cluster_stats.csv"
    summary.to_csv(summary_path, index=False)
    group_cols = ["cluster"] if "cluster" in summary.columns else ["is_aggressive"]
    stats = (
        summary.groupby(group_cols, dropna=False)
        .agg(
            trajectories=("agent_id", "count"),
            aggressive_ratio=("is_aggressive", "mean"),
            styles=("driving_style", lambda values: ",".join(sorted(set(map(str, values))))),
            mean_speed=("mean_speed", "mean"),
            max_speed=("max_speed", "mean"),
            mean_abs_acc=("mean_abs_acc", "mean"),
            mean_abs_jerk=("mean_abs_jerk", "mean"),
            lane_change_count=("lane_change_count", "mean"),
        )
        .reset_index()
    )
    stats.to_csv(stats_path, index=False)
    return summary_path, stats_path


def save_theta(output_dir: Path, theta, target_style: str) -> Path:
    path = output_dir / f"theta_{target_style}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump({name: float(value) for name, value in zip(FEATURE_NAMES, theta)}, f, indent=2)
    return path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expert_csv = Path(args.expert_csv)

    if args.raw_ngsim_csv:
        convert_ngsim(args, expert_csv)
    if not expert_csv.exists():
        raise FileNotFoundError(f"Expert CSV not found: {expert_csv}. Pass --raw-ngsim-csv first.")

    selected_features, summary = extract_expert_trajectory_features(
        str(expert_csv),
        aggressive_only=not args.include_non_aggressive,
        target_style=None if args.include_non_aggressive else args.target_style,
    )
    if not selected_features:
        raise RuntimeError("No NGSIM expert trajectories were selected.")

    summary_path, stats_path = save_cluster_outputs(output_dir, summary)
    if args.cluster_only:
        print(f"Selected expert trajectories: {len(selected_features)}")
        print(f"Saved cluster summary: {summary_path}")
        print(f"Saved cluster stats: {stats_path}")
        return

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    candidate_scenes = build_candidate_scenes(
        config_path=args.config,
        episodes=args.episodes,
        horizon=args.horizon,
        seeds=seeds,
        backend=args.backend,
        num_cav=args.num_cav,
        num_hdv=args.num_hdv,
        num_background=args.num_background,
        target_hdv_index=args.target_hdv_index,
        candidate_depth=args.candidate_depth,
        max_candidates=args.max_candidates,
        ego_action=args.ego_action,
        adv_action=args.adv_action,
        other_hdv_action=args.other_hdv_action,
    )
    irl = MaxEntIRL(feature_dim=len(FEATURE_NAMES), n_iters=args.iters, lr=args.lr, lam=args.lam)
    log = irl.fit(selected_features, candidate_scenes)

    theta_path = save_theta(output_dir, irl.theta, args.target_style)
    log_path = output_dir / f"theta_{args.target_style}_training_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    print(f"Target style: {args.target_style}")
    print(f"Selected expert trajectories: {len(selected_features)}")
    print(f"Learned CARLA {args.target_style} HDV reward theta:")
    for name, value in zip(FEATURE_NAMES, irl.theta):
        print(f"  {name:>22s}: {value:.6f}")
    print(f"Saved cluster summary: {summary_path}")
    print(f"Saved cluster stats: {stats_path}")
    print(f"Saved theta: {theta_path}")
    print(f"Saved IRL training log: {log_path}")


if __name__ == "__main__":
    main()
