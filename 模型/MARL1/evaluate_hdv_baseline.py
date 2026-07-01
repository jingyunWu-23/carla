"""
Evaluate PPOmodel889 as the current HDV/background vehicle policy.

This script is intentionally separate from MAPPO and joint training. It freezes
all policies, runs the existing merge environment, and records metrics for
road.vehicles[4], which is the current natural/background vehicle slot.

Usage:
    python evaluate_hdv_baseline.py
    python evaluate_hdv_baseline.py --episodes 50 --output-dir hdv_eval_results
"""

import argparse
import configparser
import csv
import os
import sys
from collections import Counter

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import highway_env  # noqa: E402,F401
from highway_env.envs import merge_env_v1  # noqa: E402,F401

import gym  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402


ACTION_NAMES = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate PPOmodel889 background vehicle baseline behavior."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join(SCRIPT_DIR, "PPOmodel889.zip"),
        help="Path to PPOmodel889.zip.",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "configs", "configs_ppo.ini"),
        help="Path to environment config ini.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "hdv_eval_results"),
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(i) for i in range(0, 600, 20)),
        help="Comma-separated evaluation seeds.",
    )
    parser.add_argument(
        "--other-action",
        type=int,
        default=1,
        choices=list(ACTION_NAMES.keys()),
        help="Fixed action for non-HDV externally-controlled slots.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Use deterministic PPO actions.",
    )
    return parser.parse_args()


def configure_env(env, config_path):
    config = configparser.ConfigParser()
    read_files = config.read(config_path)
    if not read_files:
        raise FileNotFoundError("Config file was not found: {}".format(config_path))
    if not config.has_section("ENV_CONFIG"):
        raise configparser.NoSectionError("ENV_CONFIG")

    env_cfg = config["ENV_CONFIG"]
    env.config["seed"] = env_cfg.getint("seed")
    env.config["simulation_frequency"] = env_cfg.getint("simulation_frequency")
    env.config["duration"] = env_cfg.getint("duration")
    env.config["policy_frequency"] = env_cfg.getint("policy_frequency")
    env.config["COLLISION_REWARD"] = env_cfg.getint("COLLISION_REWARD")
    env.config["HIGH_SPEED_REWARD"] = env_cfg.getint("HIGH_SPEED_REWARD")
    env.config["HEADWAY_COST"] = env_cfg.getint("HEADWAY_COST")
    env.config["HEADWAY_TIME"] = env_cfg.getfloat("HEADWAY_TIME")
    env.config["MERGING_LANE_COST"] = env_cfg.getint("MERGING_LANE_COST")
    env.config["traffic_density"] = env_cfg.getint("traffic_density")
    env.config["action_masking"] = (
        config.getboolean("MODEL_CONFIG", "action_masking")
        if config.has_section("MODEL_CONFIG") and config.has_option("MODEL_CONFIG", "action_masking")
        else False
    )
    return env


def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def get_front_metrics(env, hdv):
    front_vehicle, _ = env.road.surrounding_vehicles(hdv)
    if front_vehicle is None:
        return np.nan, np.nan, np.nan

    headway = safe_float(hdv.lane_distance_to(front_vehicle))
    if not np.isfinite(headway) or headway <= 0:
        return np.nan, np.nan, safe_float(front_vehicle.speed)

    thw = headway / max(hdv.speed, 1e-6)
    closing_speed = hdv.speed - front_vehicle.speed
    ttc = headway / closing_speed if closing_speed > 1e-6 else np.nan
    return headway, thw, ttc


def summarize_episode(rows, episode, seed, action_counts, initial_x, final_x, done_reason):
    speeds = np.array([r["hdv_speed"] for r in rows], dtype=float)
    accels = np.array([r["hdv_accel"] for r in rows], dtype=float)
    jerks = np.array([r["hdv_jerk"] for r in rows], dtype=float)
    ttcs = np.array([r["ttc"] for r in rows], dtype=float)
    thws = np.array([r["thw"] for r in rows], dtype=float)

    finite_ttc = ttcs[np.isfinite(ttcs)]
    finite_thw = thws[np.isfinite(thws)]

    return {
        "episode": episode,
        "seed": seed,
        "steps": len(rows),
        "done_reason": done_reason,
        "hdv_crashed": int(any(r["hdv_crashed"] for r in rows)),
        "any_crashed": int(any(r["any_crashed"] for r in rows)),
        "completion_ratio": max(0.0, min(final_x / 510.0, 1.0)),
        "delta_x": final_x - initial_x,
        "mean_speed": float(np.mean(speeds)) if len(speeds) else 0.0,
        "min_speed": float(np.min(speeds)) if len(speeds) else 0.0,
        "stagnation_rate": float(np.mean(speeds < 1.0)) if len(speeds) else 0.0,
        "mean_abs_accel": float(np.nanmean(np.abs(accels))) if len(accels) else 0.0,
        "max_decel": float(np.nanmin(accels)) if len(accels) else 0.0,
        "mean_abs_jerk": float(np.nanmean(np.abs(jerks))) if len(jerks) else 0.0,
        "min_ttc": float(np.nanmin(finite_ttc)) if len(finite_ttc) else np.nan,
        "mean_thw": float(np.nanmean(finite_thw)) if len(finite_thw) else np.nan,
        "lane_changes": int(sum(r["lane_changed"] for r in rows)),
        "emergency_brakes": int(sum(r["hdv_accel"] < -3.0 for r in rows)),
        "action_lane_left": action_counts.get(0, 0),
        "action_idle": action_counts.get(1, 0),
        "action_lane_right": action_counts.get(2, 0),
        "action_faster": action_counts.get(3, 0),
        "action_slower": action_counts.get(4, 0),
    }


def run_evaluation(args):
    os.makedirs(args.output_dir, exist_ok=True)
    step_csv = os.path.join(args.output_dir, "ppo889_hdv_steps.csv")
    summary_csv = os.path.join(args.output_dir, "ppo889_hdv_summary.csv")

    model = PPO.load(args.model_path)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    episodes = min(args.episodes, len(seeds))

    all_step_rows = []
    summaries = []

    for episode in range(episodes):
        seed = seeds[episode]
        env = configure_env(gym.make("merge-multi-agent-v0"), args.config_dir)
        state, action_mask, obs2, obs3 = env.reset(
            is_training=False,
            testing_seeds=seed,
        )

        dt = 1.0 / float(env.config.get("policy_frequency", 5))
        done = False
        step = 0
        prev_speed = None
        prev_accel = 0.0
        prev_lane = None
        action_counts = Counter()

        hdv = env.road.vehicles[4]
        initial_x = float(hdv.position[0])
        episode_rows = []

        while not done:
            action_hdv, _ = model.predict(obs3, deterministic=args.deterministic)
            action_hdv = int(action_hdv)
            action_counts[action_hdv] += 1

            n_agents = len(env.controlled_vehicles)
            full_action = [args.other_action] * max(n_agents, 3)
            full_action[3] = args.other_action     # ego/test-vehicle slot
            if len(full_action) > 4:
                full_action[4] = action_hdv        # HDV/background slot
            else:
                full_action.append(action_hdv)

            state, reward, done, info, obs2, obs3 = env.step(tuple(full_action))

            hdv = env.road.vehicles[4]
            speed = float(hdv.speed)
            accel = 0.0 if prev_speed is None else (speed - prev_speed) / dt
            jerk = 0.0 if prev_speed is None else (accel - prev_accel) / dt
            lane_changed = int(prev_lane is not None and hdv.lane_index != prev_lane)
            headway, thw, ttc = get_front_metrics(env, hdv)

            row = {
                "episode": episode,
                "seed": seed,
                "step": step,
                "hdv_action": action_hdv,
                "hdv_action_name": ACTION_NAMES.get(action_hdv, str(action_hdv)),
                "hdv_x": float(hdv.position[0]),
                "hdv_y": float(hdv.position[1]),
                "hdv_speed": speed,
                "hdv_accel": accel,
                "hdv_jerk": jerk,
                "hdv_lane": str(hdv.lane_index),
                "lane_changed": lane_changed,
                "headway": headway,
                "thw": thw,
                "ttc": ttc,
                "hdv_crashed": int(hdv.crashed),
                "any_crashed": int(any(v.crashed for v in env.road.vehicles)),
            }
            all_step_rows.append(row)
            episode_rows.append(row)

            prev_speed = speed
            prev_accel = accel
            prev_lane = hdv.lane_index
            step += 1

        final_hdv = env.road.vehicles[4]
        done_reason = "hdv_crash" if final_hdv.crashed else \
            "any_crash" if any(v.crashed for v in env.road.vehicles) else "timeout"
        summaries.append(
            summarize_episode(
                episode_rows,
                episode,
                seed,
                action_counts,
                initial_x,
                float(final_hdv.position[0]),
                done_reason,
            )
        )
        env.close()

    write_csv(step_csv, all_step_rows)
    write_csv(summary_csv, summaries)
    print(f"Step metrics saved to: {step_csv}")
    print(f"Episode summary saved to: {summary_csv}")
    print_overall_summary(summaries)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_overall_summary(summaries):
    if not summaries:
        print("No episodes evaluated.")
        return

    def mean(key):
        values = np.array([s[key] for s in summaries], dtype=float)
        return float(np.nanmean(values))

    print("\n========== PPOmodel889 HDV Baseline ==========")
    print(f"Episodes:          {len(summaries)}")
    print(f"HDV crash rate:    {mean('hdv_crashed'):.3f}")
    print(f"Any crash rate:    {mean('any_crashed'):.3f}")
    print(f"Completion ratio:  {mean('completion_ratio'):.3f}")
    print(f"Mean speed:        {mean('mean_speed'):.2f} m/s")
    print(f"Stagnation rate:   {mean('stagnation_rate'):.3f}")
    print(f"Mean |accel|:      {mean('mean_abs_accel'):.3f} m/s^2")
    print(f"Mean |jerk|:       {mean('mean_abs_jerk'):.3f} m/s^3")
    print(f"Mean lane changes: {mean('lane_changes'):.2f}")
    print(f"Mean min TTC:      {mean('min_ttc'):.2f} s")
    print(f"Mean THW:          {mean('mean_thw'):.2f} s")


if __name__ == "__main__":
    run_evaluation(parse_args())
