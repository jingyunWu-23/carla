"""Run CARLA co-evolution stages at different CAV penetration rates.

This is a thin orchestration layer over ``carla_evolution.training.train``.
Each penetration stage runs regular joint training, then the latest MAPPO and
EgoPPO checkpoints are passed to the next stage so the curriculum is continuous.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = REPO_ROOT / "模型"
TRAIN_SCRIPT = MODEL_ROOT / "carla_evolution" / "training" / "train.py"
HDV_TRAIN_SCRIPT = MODEL_ROOT / "carla_evolution" / "scripts" / "train_carla_hdv_aggressive.py"


def parse_args():
    parser = argparse.ArgumentParser(description="CARLA penetration-rate co-evolution runner.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch train.py.")
    parser.add_argument("--config", default=str(MODEL_ROOT / "carla_evolution" / "configs" / "carla_0915.yaml"))
    parser.add_argument("--backend", choices=["mock", "carla"], default="carla")
    parser.add_argument("--base-dir", default=str(MODEL_ROOT / "carla_evolution" / "results" / "penetration_coevolution"))
    parser.add_argument("--total-vehicles", type=int, default=4, help="CAV + HDV vehicles used to compute penetration.")
    parser.add_argument(
        "--penetrations",
        default="0.25,0.50,0.75",
        help="Comma-separated CAV penetration rates. Ignored when --stage-schedule is set.",
    )
    parser.add_argument(
        "--stage-rounds",
        type=int,
        default=10,
        help="Rounds per penetration stage when --penetrations is used.",
    )
    parser.add_argument("--round-offset", type=int, default=0, help="Initial round offset for the first stage.")
    parser.add_argument(
        "--stage-schedule",
        default=None,
        help=(
            "Explicit stages: name:rounds:num_cav:num_hdv[,name:rounds:num_cav:num_hdv]. "
            "Example: cav25:10:1:3,cav50:10:2:2,cav75:10:3:1"
        ),
    )
    parser.add_argument("--num-background", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--reward-type", choices=["global_R", "regionalR", "agents_rewards"], default="agents_rewards")
    parser.add_argument("--joint-strategy", choices=["fixed", "adaptive", "staged-pairs"], default="staged-pairs")
    parser.add_argument(
        "--stage-cycles",
        default=None,
        help="Stage cycles passed to train.py. Default is 0,0,<stage_rounds> for staged-pairs.",
    )
    parser.add_argument("--adv-episodes", type=int, default=80, help="Used by fixed/adaptive strategies.")
    parser.add_argument("--ego-episodes", type=int, default=800, help="Used by fixed/adaptive strategies.")
    parser.add_argument("--adv-min-eps", type=int, default=80)
    parser.add_argument("--adv-max-eps", type=int, default=150)
    parser.add_argument("--ego-min-eps", type=int, default=800)
    parser.add_argument("--ego-max-eps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--carla-rpc-timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=669)
    parser.add_argument("--torch-seed", type=int, default=669)
    parser.add_argument("--scenario-seed-span", type=int, default=1000000)
    parser.add_argument("--no-scenario-randomization", action="store_true", default=False)
    parser.add_argument("--hdv-model", default=str(MODEL_ROOT / "carla_evolution" / "results" / "hdv_models" / "aggressive_hdv_ppo.zip"))
    parser.add_argument("--hdv-action", default="keep_lane")
    parser.add_argument("--hdv-train-timesteps", type=int, default=2000, help="HDV PPO fine-tuning timesteps after each joint round. Set 0 to freeze HDV.")
    parser.add_argument("--hdv-reward-weights", default=str(MODEL_ROOT / "carla_evolution" / "results" / "hdv_irl" / "theta_aggressive.json"))
    parser.add_argument("--hdv-log-interval-episodes", type=int, default=100)
    parser.add_argument("--deterministic-cav-for-hdv", action="store_true", default=True)
    parser.add_argument("--adaptive-stage-switch", action="store_true", default=False)
    parser.add_argument("--performance-eval-episodes", type=int, default=5)
    parser.add_argument("--performance-reward-threshold", type=float, default=None)
    parser.add_argument("--performance-crash-threshold", type=float, default=None)
    parser.add_argument("--performance-completion-threshold", type=float, default=None)
    parser.add_argument("--performance-patience", type=int, default=1)
    parser.add_argument("--forget-eval-interval", type=int, default=0, help="Evaluate previous penetration stages every N rounds. <=0 disables it.")
    parser.add_argument("--forget-eval-episodes", type=int, default=5)
    parser.add_argument("--forget-retention-threshold", type=float, default=None)
    parser.add_argument("--forget-crash-increase-threshold", type=float, default=None)
    parser.add_argument("--forget-completion-drop-threshold", type=float, default=None)
    parser.add_argument("--replay-train-interval", type=int, default=0, help="Run task-level replay every N rounds. <=0 disables it.")
    parser.add_argument("--replay-train-max-stages", type=int, default=1)
    parser.add_argument("--replay-train-selection", choices=["lowest-retention", "recent"], default="lowest-retention")
    parser.add_argument("--replay-train-hdv-timesteps", type=int, default=1000)
    parser.add_argument("--replay-train-priority-eval-episodes", type=int, default=0)
    parser.add_argument("--resume-adv-model-dir", default=None)
    parser.add_argument("--resume-adv-step", type=int, default=None)
    parser.add_argument("--resume-ego-checkpoint", default=None)
    parser.add_argument("--initial-adv-episodes", type=int, default=0)
    parser.add_argument("--initial-ego-episodes", type=int, default=0)
    parser.add_argument("--extra-train-args", nargs=argparse.REMAINDER, default=None)
    return parser.parse_args()


def parse_stages(args):
    if args.stage_schedule:
        stages = []
        for raw in args.stage_schedule.split(","):
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split(":")
            if len(parts) != 4:
                raise ValueError(
                    "Invalid --stage-schedule item '{}'. Expected name:rounds:num_cav:num_hdv.".format(raw)
                )
            name, rounds, num_cav, num_hdv = parts
            stages.append(make_stage(name, int(rounds), int(num_cav), int(num_hdv)))
        if not stages:
            raise ValueError("--stage-schedule did not contain any stages.")
        return stages

    stages = []
    total = int(args.total_vehicles)
    if total <= 0:
        raise ValueError("--total-vehicles must be positive.")
    for raw_rate in args.penetrations.split(","):
        raw_rate = raw_rate.strip()
        if not raw_rate:
            continue
        rate = float(raw_rate)
        if rate <= 0.0 or rate >= 1.0:
            raise ValueError("Penetration rates must be in (0, 1): {}".format(raw_rate))
        num_cav = int(round(total * rate))
        num_cav = min(max(num_cav, 1), total - 1)
        num_hdv = total - num_cav
        name = "cav_{:02d}".format(int(round(rate * 100.0)))
        stages.append(make_stage(name, int(args.stage_rounds), num_cav, num_hdv))
    if not stages:
        raise ValueError("--penetrations did not contain any stages.")
    return stages


def make_stage(name: str, rounds: int, num_cav: int, num_hdv: int):
    total = num_cav + num_hdv
    if rounds <= 0:
        raise ValueError("Stage '{}' has non-positive rounds.".format(name))
    if num_cav <= 0 or num_hdv < 0 or total <= 0:
        raise ValueError("Stage '{}' has invalid num_cav/num_hdv.".format(name))
    return {
        "name": name,
        "rounds": rounds,
        "num_cav": num_cav,
        "num_hdv": num_hdv,
        "penetration": float(num_cav) / float(total),
    }


def latest_joint_output(stage_base: Path) -> Path:
    candidates = [p for p in stage_base.iterdir() if p.is_dir() and p.name.startswith("joint_")]
    if not candidates:
        raise FileNotFoundError("No joint_* output directory found under {}".format(stage_base))
    return max(candidates, key=lambda p: p.stat().st_mtime)


def checkpoint_steps(model_dir: Path, pattern: str):
    regex = re.compile(pattern)
    steps = []
    if not model_dir.exists():
        return steps
    for path in model_dir.iterdir():
        match = regex.fullmatch(path.name)
        if match:
            steps.append(int(match.group(1)))
    return steps


def latest_mappo_step(adv_dir: Path) -> int:
    actor_steps = set(checkpoint_steps(adv_dir, r"actor_(\d+)\.pt"))
    critic_steps = set(checkpoint_steps(adv_dir, r"critic_(\d+)\.pt"))
    common = actor_steps & critic_steps
    if not common:
        raise FileNotFoundError("No matching actor/critic checkpoints found in {}".format(adv_dir))
    return max(common)


def latest_ego_checkpoint(ego_dir: Path) -> Path:
    candidates = []
    for step in checkpoint_steps(ego_dir, r"checkpoint-(\d+)\.pt"):
        candidates.append((step, ego_dir / "checkpoint-{}.pt".format(step)))
    if not candidates:
        raise FileNotFoundError("No checkpoint-N.pt files found in {}".format(ego_dir))
    return max(candidates)[1]


def read_last_round_metrics(round_log: Path):
    if not round_log.exists():
        return {}
    with round_log.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def build_train_command(args, stage, stage_base, round_offset, resume):
    stage_cycles = args.stage_cycles
    if args.joint_strategy == "staged-pairs" and not stage_cycles:
        stage_cycles = "0,0,{}".format(stage["rounds"])

    cmd = [
        args.python,
        str(TRAIN_SCRIPT),
        "--mode", "joint",
        "--backend", args.backend,
        "--config", args.config,
        "--base-dir", str(stage_base),
        "--joint-strategy", args.joint_strategy,
        "--rounds", str(stage["rounds"]),
        "--round-offset", str(round_offset),
        "--num-cav", str(stage["num_cav"]),
        "--num-hdv", str(stage["num_hdv"]),
        "--num-background", str(args.num_background),
        "--max-steps", str(args.max_steps),
        "--reward-type", args.reward_type,
        "--eval-interval", str(args.eval_interval),
        "--save-interval", str(args.save_interval),
        "--carla-rpc-timeout", str(args.carla_rpc_timeout),
        "--seed", str(args.seed),
        "--torch-seed", str(args.torch_seed),
        "--scenario-seed-span", str(args.scenario_seed_span),
        "--hdv-action", args.hdv_action,
        "--initial-adv-episodes", str(resume["initial_adv_episodes"]),
        "--initial-ego-episodes", str(resume["initial_ego_episodes"]),
    ]
    if stage_cycles:
        cmd.extend(["--stage-cycles", stage_cycles])
    if args.joint_strategy != "staged-pairs":
        cmd.extend([
            "--adv-episodes", str(args.adv_episodes),
            "--ego-episodes", str(args.ego_episodes),
            "--adv-min-eps", str(args.adv_min_eps),
            "--adv-max-eps", str(args.adv_max_eps),
            "--ego-min-eps", str(args.ego_min_eps),
            "--ego-max-eps", str(args.ego_max_eps),
        ])
    if args.no_scenario_randomization:
        cmd.append("--no-scenario-randomization")
    if args.hdv_model:
        cmd.extend(["--hdv-model", args.hdv_model])
    if resume["adv_model_dir"]:
        cmd.extend(["--resume-adv-model-dir", resume["adv_model_dir"]])
    if resume["adv_step"] is not None:
        cmd.extend(["--resume-adv-step", str(resume["adv_step"])])
    if resume["ego_checkpoint"]:
        cmd.extend(["--resume-ego-checkpoint", resume["ego_checkpoint"]])
    if args.extra_train_args:
        cmd.extend(args.extra_train_args)
    return cmd


def build_hdv_train_command(
    args,
    stage,
    stage_base,
    stage_index,
    stage_round,
    resume,
    hdv_model_path,
    timesteps=None,
    save_name=None,
):
    output_dir = stage_base / "hdv_models"
    save_name = save_name or "{}_round_{:03d}_hdv".format(stage["name"], stage_round)
    timesteps = int(args.hdv_train_timesteps if timesteps is None else timesteps)
    cmd = [
        args.python,
        str(HDV_TRAIN_SCRIPT),
        "--backend", args.backend,
        "--config", args.config,
        "--reward-weights", args.hdv_reward_weights,
        "--output-dir", str(output_dir),
        "--save-name", save_name,
        "--timesteps", str(timesteps),
        "--num-cav", str(stage["num_cav"]),
        "--num-hdv", str(stage["num_hdv"]),
        "--num-background", str(args.num_background),
        "--seed", str(args.seed + 1000 * stage_index + stage_round),
        "--torch-seed", str(args.torch_seed),
        "--max-steps", str(args.max_steps),
        "--log-interval-episodes", str(args.hdv_log_interval_episodes),
        "--adv-model-dir", resume["adv_model_dir"],
        "--adv-step", str(resume["adv_step"]),
        "--ego-checkpoint", resume["ego_checkpoint"],
    ]
    if hdv_model_path:
        cmd.extend(["--hdv-model", str(hdv_model_path)])
    if args.deterministic_cav_for_hdv:
        cmd.append("--deterministic-cav")
    return cmd, output_dir / "{}.zip".format(save_name)


def append_stage_log(path: Path, row: dict):
    fieldnames = [
        "stage_index",
        "stage",
        "rounds",
        "round_offset",
        "num_cav",
        "num_hdv",
        "cav_penetration",
        "output_dir",
        "resume_adv_model_dir",
        "resume_adv_step",
        "resume_ego_checkpoint",
        "final_adv_step",
        "final_ego_checkpoint",
        "final_hdv_model",
        "hdv_train_timesteps",
        "ego_avg_reward",
        "ego_crash_rate",
        "ego_road_completion_rate",
        "ego_avg_speed",
        "adv_avg_reward",
        "adv_crash_rate",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_performance_log(path: Path, row: dict):
    fieldnames = [
        "stage_index",
        "stage",
        "stage_round",
        "global_round",
        "num_cav",
        "num_hdv",
        "cav_penetration",
        "eval_episodes",
        "mean_reward",
        "std_reward",
        "crash_rate",
        "road_completion_rate",
        "avg_speed",
        "mean_length",
        "reward_threshold",
        "crash_threshold",
        "completion_threshold",
        "reward_pass",
        "crash_pass",
        "completion_pass",
        "stage_learned",
        "consecutive_passes",
        "patience",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_forgetting_log(path: Path, row: dict):
    fieldnames = [
        "global_round",
        "current_stage",
        "current_stage_round",
        "replay_stage",
        "replay_num_cav",
        "replay_num_hdv",
        "replay_cav_penetration",
        "eval_episodes",
        "baseline_round",
        "baseline_reward",
        "current_reward",
        "reward_drop",
        "reward_retention_ratio",
        "baseline_crash_rate",
        "current_crash_rate",
        "crash_rate_increase",
        "baseline_completion",
        "current_completion",
        "completion_drop",
        "mean_length",
        "avg_speed",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_replay_log(path: Path, row: dict):
    fieldnames = [
        "global_round",
        "current_stage",
        "current_stage_round",
        "replay_stage",
        "replay_num_cav",
        "replay_num_hdv",
        "replay_cav_penetration",
        "selection",
        "priority_rank",
        "priority_retention",
        "priority_current_reward",
        "priority_baseline_reward",
        "priority_crash_rate_increase",
        "priority_completion_drop",
        "joint_output_dir",
        "final_adv_step",
        "final_ego_checkpoint",
        "final_hdv_model",
        "hdv_train_timesteps",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def adaptive_switch_enabled(args):
    return bool(
        args.adaptive_stage_switch
        and any(
            value is not None
            for value in (
                args.performance_reward_threshold,
                args.performance_crash_threshold,
                args.performance_completion_threshold,
            )
        )
    )


def evaluate_current_stage(args, stage, resume, hdv_model_path, eval_seed, eval_episodes=None):
    from stable_baselines3 import PPO
    from carla_evolution.agents.ego_ppo import EgoPPOAdapter
    from carla_evolution.agents.mappo import MAPPOAgent
    from carla_evolution.envs.factory import make_env
    from carla_evolution.training import train as train_mod

    env_config = {
        "backend": args.backend,
        "max_episode_steps": args.max_steps,
        "num_cav": stage["num_cav"],
        "num_hdv": stage["num_hdv"],
        "num_background": args.num_background,
        "timeout": args.carla_rpc_timeout,
        "randomize_scenarios": not args.no_scenario_randomization,
    }
    env = make_env(env_config, config_path=args.config)
    mappo = MAPPOAgent.from_config(
        {
            "reward_type": args.reward_type,
            "use_cuda": not getattr(args, "no_cuda", False),
            "torch_seed": args.torch_seed,
        },
        state_dim=env.n_s,
        action_dim=env.n_a,
    )
    ego = EgoPPOAdapter.from_config(
        {"use_cuda": not getattr(args, "no_cuda", False), "seed": args.torch_seed},
        state_dim=env.n_s,
        action_dim=env.n_a,
    )
    mappo.load(resume["adv_model_dir"], int(resume["adv_step"]), train_mode=False)
    ego.load(resume["ego_checkpoint"])
    hdv_policy = PPO.load(str(hdv_model_path)) if hdv_model_path else None
    eval_args = SimpleNamespace(
        hdv_action=args.hdv_action,
        _hdv_policy=hdv_policy,
    )

    records = []
    try:
        eval_count = int(args.performance_eval_episodes if eval_episodes is None else eval_episodes)
        for episode in range(eval_count):
            records.append(
                run_eval_episode(
                    env=env,
                    mappo=mappo,
                    ego=ego,
                    train_mod=train_mod,
                    eval_args=eval_args,
                    seed=int(eval_seed) + episode,
                    max_steps=int(args.max_steps),
                )
            )
    finally:
        env.close()

    if not records:
        return {
            "mean_reward": 0.0,
            "std_reward": 0.0,
            "crash_rate": 1.0,
            "road_completion_rate": 0.0,
            "avg_speed": 0.0,
            "mean_length": 0.0,
        }
    rewards = [float(item.get("episode_reward", 0.0)) for item in records]
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "crash_rate": float(np.mean([float(item.get("crash", False)) for item in records])),
        "road_completion_rate": float(np.mean([float(item.get("route_completion", 0.0)) for item in records])),
        "avg_speed": float(np.mean([float(item.get("average_speed", 0.0)) for item in records])),
        "mean_length": float(np.mean([float(item.get("steps", 0)) for item in records])),
    }


def run_eval_episode(env, mappo, ego, train_mod, eval_args, seed: int, max_steps: int):
    state, reset_info = env.reset(seed=seed)
    n_agents = len(env.controlled_vehicles)
    ego_state = np.asarray(env.obs2, dtype=np.float32).flatten()
    episode_reward = 0.0
    last_info = {"route_completion": reset_info.get("route_completion", 0.0)}
    episode_diagnostics = train_mod.init_episode_diagnostics()

    for step in range(max_steps):
        adv_actions = mappo.action(state, n_agents)
        ego_action, _, _ = ego.select_action(ego_state, deterministic=True)
        full_action = train_mod.append_hdv_actions(env, list(adv_actions) + [ego_action], eval_args)
        next_state, global_reward, terminated, truncated, info = env.step(full_action)
        next_ego_state = np.asarray(env.obs2, dtype=np.float32).flatten()
        ego_reward = float(info.get("ego_reward", global_reward))
        episode_reward += ego_reward
        last_info = info
        train_mod.update_episode_diagnostics(episode_diagnostics, info)
        state = next_state
        ego_state = next_ego_state
        if terminated or truncated:
            break

    reward_array = np.asarray(last_info.get("agents_rewards", [0.0]), dtype=np.float32)
    return train_mod.episode_metrics(
        step + 1,
        episode_reward,
        reward_array,
        train_mod.merge_episode_diagnostics(last_info, episode_diagnostics),
    )


def performance_passes(args, metrics):
    reward_pass = (
        True
        if args.performance_reward_threshold is None
        else float(metrics.get("mean_reward", 0.0)) >= float(args.performance_reward_threshold)
    )
    crash_pass = (
        True
        if args.performance_crash_threshold is None
        else float(metrics.get("crash_rate", 1.0)) <= float(args.performance_crash_threshold)
    )
    completion_pass = (
        True
        if args.performance_completion_threshold is None
        else float(metrics.get("road_completion_rate", 0.0)) >= float(args.performance_completion_threshold)
    )
    return bool(reward_pass and crash_pass and completion_pass), reward_pass, crash_pass, completion_pass


def reward_retention(current_reward: float, baseline_reward: float) -> float:
    scale = max(abs(float(baseline_reward)), 1.0)
    drop = max(0.0, float(baseline_reward) - float(current_reward))
    return float(1.0 - drop / scale)


def forgetting_metrics(args, item, current_metrics):
    baseline = item["baseline_metrics"]
    baseline_reward = float(baseline.get("mean_reward", 0.0))
    current_reward = float(current_metrics.get("mean_reward", 0.0))
    baseline_crash = float(baseline.get("crash_rate", 0.0))
    current_crash = float(current_metrics.get("crash_rate", 0.0))
    baseline_completion = float(baseline.get("road_completion_rate", 0.0))
    current_completion = float(current_metrics.get("road_completion_rate", 0.0))
    return {
        "baseline_reward": baseline_reward,
        "current_reward": current_reward,
        "reward_drop": baseline_reward - current_reward,
        "reward_retention_ratio": reward_retention(current_reward, baseline_reward),
        "baseline_crash_rate": baseline_crash,
        "current_crash_rate": current_crash,
        "crash_rate_increase": current_crash - baseline_crash,
        "baseline_completion": baseline_completion,
        "current_completion": current_completion,
        "completion_drop": baseline_completion - current_completion,
        "mean_length": float(current_metrics.get("mean_length", 0.0)),
        "avg_speed": float(current_metrics.get("avg_speed", 0.0)),
    }


def evaluate_forgetting(
    args,
    replay_buffer,
    current_stage,
    current_stage_round,
    global_round,
    resume,
    hdv_model_path,
    log_path,
    eval_episodes=None,
):
    rows = []
    eval_episodes = int(args.forget_eval_episodes if eval_episodes is None else eval_episodes)
    for item_index, item in enumerate(replay_buffer):
        replay_stage = item["stage"]
        metrics = evaluate_current_stage(
            args=args,
            stage=replay_stage,
            resume=resume,
            hdv_model_path=hdv_model_path,
            eval_seed=int(args.seed) + 200000 + int(global_round) * 100 + item_index * 1000,
            eval_episodes=eval_episodes,
        )
        row = {
            "global_round": global_round,
            "current_stage": current_stage["name"],
            "current_stage_round": current_stage_round,
            "replay_stage": replay_stage["name"],
            "replay_num_cav": replay_stage["num_cav"],
            "replay_num_hdv": replay_stage["num_hdv"],
            "replay_cav_penetration": replay_stage["penetration"],
            "eval_episodes": eval_episodes,
            "baseline_round": item["baseline_round"],
            **forgetting_metrics(args, item, metrics),
        }
        append_forgetting_log(log_path, row)
        rows.append({"item": item, "metrics": row})
    if rows:
        print(
            "  [FORGET] min_retention={:.3f} max_crash_inc={:.3f} max_completion_drop={:.3f}".format(
                min(float(row["metrics"]["reward_retention_ratio"]) for row in rows),
                max(float(row["metrics"]["crash_rate_increase"]) for row in rows),
                max(float(row["metrics"]["completion_drop"]) for row in rows),
            )
        )
    return rows


def forgetting_passes(args, forgetting_rows):
    if not forgetting_rows:
        return True
    min_retention = min(float(row["metrics"]["reward_retention_ratio"]) for row in forgetting_rows)
    max_crash_increase = max(float(row["metrics"]["crash_rate_increase"]) for row in forgetting_rows)
    max_completion_drop = max(float(row["metrics"]["completion_drop"]) for row in forgetting_rows)
    if args.forget_retention_threshold is not None and min_retention < float(args.forget_retention_threshold):
        return False
    if args.forget_crash_increase_threshold is not None and max_crash_increase > float(args.forget_crash_increase_threshold):
        return False
    if args.forget_completion_drop_threshold is not None and max_completion_drop > float(args.forget_completion_drop_threshold):
        return False
    return True


def select_replay_items(args, replay_buffer, forgetting_rows):
    if not replay_buffer:
        return []
    max_stages = int(args.replay_train_max_stages)
    if max_stages <= 0:
        max_stages = len(replay_buffer)
    if args.replay_train_selection == "recent":
        selected = [{"item": item, "metrics": {}} for item in replay_buffer[-max_stages:]]
        selected.reverse()
        return selected
    rows = list(forgetting_rows)
    if not rows:
        rows = [{"item": item, "metrics": {}} for item in replay_buffer]
    rows.sort(key=lambda row: float(row["metrics"].get("reward_retention_ratio", 1.0)))
    return rows[:max_stages]


def run_replay_training(
    args,
    selected_items,
    run_dir,
    current_stage,
    current_stage_round,
    global_round,
    resume,
    hdv_model_path,
    env,
    replay_log,
):
    if not selected_items:
        return resume, hdv_model_path
    print("\n{} TASK REPLAY TRAINING | stages={} {}".format("#" * 16, len(selected_items), "#" * 16))
    for rank, selected in enumerate(selected_items, start=1):
        item = selected["item"]
        priority = selected.get("metrics", {})
        replay_stage = item["stage"]
        replay_base = run_dir / "replay" / "round_{:04d}_rank_{:02d}_{}".format(
            int(global_round), rank, replay_stage["name"]
        )
        replay_base.mkdir(parents=True, exist_ok=True)
        train_stage = dict(replay_stage)
        train_stage["rounds"] = 1
        if hdv_model_path:
            args.hdv_model = str(hdv_model_path)
        cmd = build_train_command(args, train_stage, replay_base, int(global_round), resume)
        print(
            "  [REPLAY] current={} R{} -> replay={} | retention={} | CAV={} HDV={}".format(
                current_stage["name"],
                current_stage_round,
                replay_stage["name"],
                priority.get("reward_retention_ratio", ""),
                replay_stage["num_cav"],
                replay_stage["num_hdv"],
            )
        )
        print("  Replay joint command:")
        print(" ".join(str(item) for item in cmd))
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)

        output_dir = latest_joint_output(replay_base)
        adv_dir = output_dir / "models" / "adv"
        ego_dir = output_dir / "models" / "ego"
        final_adv_step = latest_mappo_step(adv_dir)
        final_ego_checkpoint = latest_ego_checkpoint(ego_dir)
        resume = {
            "adv_model_dir": str(adv_dir),
            "adv_step": final_adv_step,
            "ego_checkpoint": str(final_ego_checkpoint),
            "initial_adv_episodes": final_adv_step,
            "initial_ego_episodes": checkpoint_step_from_name(final_ego_checkpoint),
        }

        if int(args.replay_train_hdv_timesteps) > 0 and replay_stage["num_hdv"] > 0:
            hdv_cmd, next_hdv_model_path = build_hdv_train_command(
                args=args,
                stage=replay_stage,
                stage_base=replay_base,
                stage_index=9000 + rank,
                stage_round=int(global_round),
                resume=resume,
                hdv_model_path=hdv_model_path,
                timesteps=int(args.replay_train_hdv_timesteps),
                save_name="replay_{}_round_{:04d}_rank_{:02d}_hdv".format(
                    replay_stage["name"], int(global_round), rank
                ),
            )
            print("  Replay HDV command:")
            print(" ".join(str(item) for item in hdv_cmd))
            subprocess.run(hdv_cmd, cwd=str(REPO_ROOT), env=env, check=True)
            hdv_model_path = str(next_hdv_model_path)

        append_replay_log(replay_log, {
            "global_round": global_round,
            "current_stage": current_stage["name"],
            "current_stage_round": current_stage_round,
            "replay_stage": replay_stage["name"],
            "replay_num_cav": replay_stage["num_cav"],
            "replay_num_hdv": replay_stage["num_hdv"],
            "replay_cav_penetration": replay_stage["penetration"],
            "selection": args.replay_train_selection,
            "priority_rank": rank,
            "priority_retention": priority.get("reward_retention_ratio", ""),
            "priority_current_reward": priority.get("current_reward", ""),
            "priority_baseline_reward": priority.get("baseline_reward", ""),
            "priority_crash_rate_increase": priority.get("crash_rate_increase", ""),
            "priority_completion_drop": priority.get("completion_drop", ""),
            "joint_output_dir": str(output_dir),
            "final_adv_step": final_adv_step,
            "final_ego_checkpoint": str(final_ego_checkpoint),
            "final_hdv_model": hdv_model_path,
            "hdv_train_timesteps": args.replay_train_hdv_timesteps,
        })
    return resume, hdv_model_path


def main():
    args = parse_args()
    stages = parse_stages(args)
    run_dir = Path(args.base_dir) / datetime.utcnow().strftime("penetration_%b_%d_%H_%M_%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_log = run_dir / "penetration_stage_log.csv"
    performance_log = run_dir / "performance_eval_log.csv"
    forgetting_log = run_dir / "forgetting_eval_log.csv"
    replay_log = run_dir / "replay_train_log.csv"

    resume = {
        "adv_model_dir": args.resume_adv_model_dir,
        "adv_step": args.resume_adv_step,
        "ego_checkpoint": args.resume_ego_checkpoint,
        "initial_adv_episodes": int(args.initial_adv_episodes),
        "initial_ego_episodes": int(args.initial_ego_episodes),
    }

    print("=" * 70)
    print("CARLA PENETRATION CO-EVOLUTION")
    print("Run dir: {}".format(run_dir))
    print("Stages: {}".format(stages))
    print("Initial HDV model: {}".format(args.hdv_model or "new PPO"))
    print("HDV train timesteps/round: {}".format(args.hdv_train_timesteps))
    print("Adaptive stage switch: {}".format(adaptive_switch_enabled(args)))
    print("Replay training interval: {}".format(args.replay_train_interval))
    print("=" * 70)

    round_offset = int(args.round_offset)
    env = os.environ.copy()
    pythonpath = str(MODEL_ROOT)
    env["PYTHONPATH"] = pythonpath if not env.get("PYTHONPATH") else pythonpath + os.pathsep + env["PYTHONPATH"]

    hdv_model_path = args.hdv_model
    replay_buffer = []
    for stage_index, stage in enumerate(stages, start=1):
        stage_base = run_dir / "{:02d}_{}".format(stage_index, stage["name"])
        stage_base.mkdir(parents=True, exist_ok=True)
        round_by_round = int(args.hdv_train_timesteps) > 0 or adaptive_switch_enabled(args)
        round_units = int(stage["rounds"]) if round_by_round else 1
        consecutive_passes = 0
        for stage_round in range(1, round_units + 1):
            train_stage = dict(stage)
            if round_by_round:
                train_stage["rounds"] = 1
                train_base = stage_base / "round_{:03d}".format(stage_round)
            else:
                train_base = stage_base
            if hdv_model_path:
                args.hdv_model = str(hdv_model_path)
            cmd = build_train_command(args, train_stage, train_base, round_offset, resume)

            print("\n" + "=" * 70)
            print("Stage {}/{}: {} round {}/{} | CAV={} | HDV={} | penetration={:.0%}".format(
                stage_index,
                len(stages),
                stage["name"],
                stage_round,
                round_units,
                stage["num_cav"],
                stage["num_hdv"],
                stage["penetration"],
            ))
            print("Joint command:")
            print(" ".join(cmd))
            print("=" * 70)
            subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)

            output_dir = latest_joint_output(train_base)
            adv_dir = output_dir / "models" / "adv"
            ego_dir = output_dir / "models" / "ego"
            final_adv_step = latest_mappo_step(adv_dir)
            final_ego_checkpoint = latest_ego_checkpoint(ego_dir)
            metrics = read_last_round_metrics(output_dir / "joint_round_log.csv")

            resume = {
                "adv_model_dir": str(adv_dir),
                "adv_step": final_adv_step,
                "ego_checkpoint": str(final_ego_checkpoint),
                "initial_adv_episodes": final_adv_step,
                "initial_ego_episodes": checkpoint_step_from_name(final_ego_checkpoint),
            }

            if int(args.hdv_train_timesteps) > 0 and stage["num_hdv"] > 0:
                hdv_cmd, next_hdv_model_path = build_hdv_train_command(
                    args=args,
                    stage=stage,
                    stage_base=stage_base,
                    stage_index=stage_index,
                    stage_round=stage_round,
                    resume=resume,
                    hdv_model_path=hdv_model_path,
                )
                print("\nHDV co-evolution command:")
                print(" ".join(str(item) for item in hdv_cmd))
                subprocess.run(hdv_cmd, cwd=str(REPO_ROOT), env=env, check=True)
                hdv_model_path = str(next_hdv_model_path)

            global_round = round_offset + int(train_stage["rounds"])

            append_stage_log(summary_log, {
                "stage_index": stage_index,
                "stage": stage["name"],
                "rounds": train_stage["rounds"],
                "round_offset": round_offset,
                "num_cav": stage["num_cav"],
                "num_hdv": stage["num_hdv"],
                "cav_penetration": stage["penetration"],
                "output_dir": str(output_dir),
                "resume_adv_model_dir": resume["adv_model_dir"],
                "resume_adv_step": resume["adv_step"],
                "resume_ego_checkpoint": resume["ego_checkpoint"],
                "final_adv_step": final_adv_step,
                "final_ego_checkpoint": str(final_ego_checkpoint),
                "final_hdv_model": hdv_model_path,
                "hdv_train_timesteps": args.hdv_train_timesteps,
                "ego_avg_reward": metrics.get("ego_avg_reward", ""),
                "ego_crash_rate": metrics.get("ego_crash_rate", ""),
                "ego_road_completion_rate": metrics.get("ego_road_completion_rate", ""),
                "ego_avg_speed": metrics.get("ego_avg_speed", ""),
                "adv_avg_reward": metrics.get("adv_avg_reward", ""),
                "adv_crash_rate": metrics.get("adv_crash_rate", ""),
            })

            if adaptive_switch_enabled(args):
                eval_metrics = evaluate_current_stage(
                    args=args,
                    stage=stage,
                    resume=resume,
                    hdv_model_path=hdv_model_path,
                    eval_seed=int(args.seed) + 100000 * stage_index + 1000 * stage_round,
                )
                passed, reward_pass, crash_pass, completion_pass = performance_passes(args, eval_metrics)
                forgetting_rows = []
                forget_gate_enabled = any(
                    value is not None
                    for value in (
                        args.forget_retention_threshold,
                        args.forget_crash_increase_threshold,
                        args.forget_completion_drop_threshold,
                    )
                )
                if forget_gate_enabled and replay_buffer:
                    forgetting_rows = evaluate_forgetting(
                        args=args,
                        replay_buffer=replay_buffer,
                        current_stage=stage,
                        current_stage_round=stage_round,
                        global_round=global_round,
                        resume=resume,
                        hdv_model_path=hdv_model_path,
                        log_path=forgetting_log,
                    )
                    passed = bool(passed and forgetting_passes(args, forgetting_rows))
                if passed:
                    consecutive_passes += 1
                else:
                    consecutive_passes = 0
                stage_learned = consecutive_passes >= max(int(args.performance_patience), 1)
                append_performance_log(performance_log, {
                    "stage_index": stage_index,
                    "stage": stage["name"],
                    "stage_round": stage_round,
                    "global_round": global_round,
                    "num_cav": stage["num_cav"],
                    "num_hdv": stage["num_hdv"],
                    "cav_penetration": stage["penetration"],
                    "eval_episodes": args.performance_eval_episodes,
                    "mean_reward": eval_metrics.get("mean_reward", 0.0),
                    "std_reward": eval_metrics.get("std_reward", 0.0),
                    "crash_rate": eval_metrics.get("crash_rate", 0.0),
                    "road_completion_rate": eval_metrics.get("road_completion_rate", 0.0),
                    "avg_speed": eval_metrics.get("avg_speed", 0.0),
                    "mean_length": eval_metrics.get("mean_length", 0.0),
                    "reward_threshold": args.performance_reward_threshold,
                    "crash_threshold": args.performance_crash_threshold,
                    "completion_threshold": args.performance_completion_threshold,
                    "reward_pass": reward_pass,
                    "crash_pass": crash_pass,
                    "completion_pass": completion_pass,
                    "stage_learned": stage_learned,
                    "consecutive_passes": consecutive_passes,
                    "patience": args.performance_patience,
                })
                print(
                    "  [PERF] {} R{} | reward={:.2f} crash={:.3f} completion={:.3f} "
                    "speed={:.2f} passes={}/{} learned={}".format(
                        stage["name"],
                        stage_round,
                        float(eval_metrics.get("mean_reward", 0.0)),
                        float(eval_metrics.get("crash_rate", 0.0)),
                        float(eval_metrics.get("road_completion_rate", 0.0)),
                        float(eval_metrics.get("avg_speed", 0.0)),
                        consecutive_passes,
                        max(int(args.performance_patience), 1),
                        stage_learned,
                    )
                )

            forgetting_rows_for_replay = []
            if (
                int(args.forget_eval_interval) > 0
                and replay_buffer
                and global_round % int(args.forget_eval_interval) == 0
            ):
                forgetting_rows_for_replay = evaluate_forgetting(
                    args=args,
                    replay_buffer=replay_buffer,
                    current_stage=stage,
                    current_stage_round=stage_round,
                    global_round=global_round,
                    resume=resume,
                    hdv_model_path=hdv_model_path,
                    log_path=forgetting_log,
                )

            if (
                int(args.replay_train_interval) > 0
                and replay_buffer
                and global_round % int(args.replay_train_interval) == 0
            ):
                if not forgetting_rows_for_replay and args.replay_train_selection == "lowest-retention":
                    priority_eval_episodes = (
                        int(args.replay_train_priority_eval_episodes)
                        if int(args.replay_train_priority_eval_episodes) > 0
                        else int(args.forget_eval_episodes)
                    )
                    forgetting_rows_for_replay = evaluate_forgetting(
                        args=args,
                        replay_buffer=replay_buffer,
                        current_stage=stage,
                        current_stage_round=stage_round,
                        global_round=global_round,
                        resume=resume,
                        hdv_model_path=hdv_model_path,
                        log_path=forgetting_log,
                        eval_episodes=priority_eval_episodes,
                    )
                selected_items = select_replay_items(args, replay_buffer, forgetting_rows_for_replay)
                resume, hdv_model_path = run_replay_training(
                    args=args,
                    selected_items=selected_items,
                    run_dir=run_dir,
                    current_stage=stage,
                    current_stage_round=stage_round,
                    global_round=global_round,
                    resume=resume,
                    hdv_model_path=hdv_model_path,
                    env=env,
                    replay_log=replay_log,
                )

            round_offset += int(train_stage["rounds"])

            print("Round complete: {} R{}".format(stage["name"], stage_round))
            print("  output_dir: {}".format(output_dir))
            print("  next MAPPO: {} step={}".format(adv_dir, final_adv_step))
            print("  next Ego:   {}".format(final_ego_checkpoint))
            print("  next HDV:   {}".format(hdv_model_path))

            if adaptive_switch_enabled(args) and stage_learned:
                print(
                    "Adaptive switch: stage {} learned at round {} after {} consecutive passes.".format(
                        stage["name"], stage_round, consecutive_passes
                    )
                )
                break

        baseline_metrics = evaluate_current_stage(
            args=args,
            stage=stage,
            resume=resume,
            hdv_model_path=hdv_model_path,
            eval_seed=int(args.seed) + 300000 + stage_index * 1000,
            eval_episodes=int(args.forget_eval_episodes),
        )
        replay_buffer.append({
            "stage": dict(stage),
            "baseline_metrics": dict(baseline_metrics),
            "baseline_round": int(round_offset),
        })
        print(
            "Replay baseline saved: {} | reward={:.2f} crash={:.3f} completion={:.3f}".format(
                stage["name"],
                float(baseline_metrics.get("mean_reward", 0.0)),
                float(baseline_metrics.get("crash_rate", 0.0)),
                float(baseline_metrics.get("road_completion_rate", 0.0)),
            )
        )
        print("Stage complete: {}".format(stage["name"]))

    print("\n" + "=" * 70)
    print("PENETRATION CO-EVOLUTION COMPLETE")
    print("Summary log: {}".format(summary_log))
    print("Performance log: {}".format(performance_log))
    print("Forgetting log: {}".format(forgetting_log))
    print("Replay log: {}".format(replay_log))
    print("=" * 70)


def checkpoint_step_from_name(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)\.pt$", str(path))
    if not match:
        return 0
    return int(match.group(1))


if __name__ == "__main__":
    main()
