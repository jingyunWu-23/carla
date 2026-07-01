"""
Learn an aggressive HDV reward function for the current merge environment.

This script follows the HAD-Gen idea:
1. extract HAD-Gen-style 10D features from human trajectories;
2. keep/identify aggressive human trajectories;
3. generate candidate HDV trajectories in merge-multi-agent-v0;
4. learn theta with MaxEnt IRL;
5. save theta_aggressive.json for later HDV fine-tuning.

Example:
    python learn_hdv_aggressive_reward.py --expert-csv data/aggressive_human.csv
"""

import argparse
import configparser
import copy
import itertools
import json
import os
import sys

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import gym  # noqa: E402
import highway_env  # noqa: E402,F401
from highway_env.envs import merge_env_v1  # noqa: E402,F401

from hdv.irl.features import (  # noqa: E402
    EnvFeatureTracker,
    FEATURE_NAMES,
    aggregate_step_features,
    extract_expert_trajectory_features,
)
from hdv.irl.maxent import MaxEntIRL  # noqa: E402


ACTION_NAMES = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Learn aggressive HDV reward weights with MaxEnt IRL."
    )
    parser.add_argument("--expert-csv", required=True, help="Human trajectory CSV.")
    parser.add_argument(
        "--config-dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "configs", "configs_ppo.ini"),
        help="Path to environment config ini.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "hdv_reward_irl"),
        help="Directory for learned reward outputs.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="Number of environment scenes used to build candidate trajectories.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=8,
        help="Steps per candidate rollout.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(i) for i in range(0, 600, 20)),
        help="Comma-separated environment seeds.",
    )
    parser.add_argument("--iters", type=int, default=200, help="IRL iterations.")
    parser.add_argument("--lr", type=float, default=0.05, help="IRL learning rate.")
    parser.add_argument("--lam", type=float, default=0.01, help="L2 regularization.")
    parser.add_argument(
        "--other-action",
        type=int,
        default=1,
        choices=list(ACTION_NAMES.keys()),
        help="Fixed action for adversarial CAV slots.",
    )
    parser.add_argument(
        "--ego-action",
        type=int,
        default=1,
        choices=list(ACTION_NAMES.keys()),
        help="Fixed action for ego slot.",
    )
    parser.add_argument(
        "--include-non-aggressive",
        action="store_true",
        help="Use all expert trajectories instead of filtering aggressive ones.",
    )
    parser.add_argument(
        "--candidate-depth",
        type=int,
        default=2,
        help="Number of action segments in each candidate sequence.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=25,
        help="Maximum candidate trajectories per environment scene.",
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


def build_candidate_scenes(args):
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        seeds = list(range(args.episodes))

    candidate_scenes = []
    for episode in range(args.episodes):
        seed = seeds[episode % len(seeds)]
        env = configure_env(gym.make("merge-multi-agent-v0"), args.config_dir)
        env.reset(is_training=False, testing_seeds=seed)

        scene = []
        action_sequences = build_action_sequences(
            depth=args.candidate_depth,
            max_candidates=args.max_candidates,
        )
        for action_sequence in action_sequences:
            env_copy = copy.deepcopy(env)
            scene.append(
                rollout_candidate(
                    env_copy,
                    hdv_actions=action_sequence,
                    horizon=args.horizon,
                    ego_action=args.ego_action,
                    other_action=args.other_action,
                )
            )
            env_copy.close()

        candidate_scenes.append(scene)
        env.close()

    return candidate_scenes


def build_action_sequences(depth, max_candidates):
    actions = sorted(ACTION_NAMES)
    sequences = list(itertools.product(actions, repeat=max(depth, 1)))
    if len(sequences) <= max_candidates:
        return sequences

    # Keep all single-action archetypes and sample the remaining mixed sequences
    # deterministically for reproducible IRL runs.
    archetypes = [(action,) * max(depth, 1) for action in actions]
    remaining = [seq for seq in sequences if seq not in archetypes]
    rng = np.random.RandomState(0)
    keep = max(0, max_candidates - len(archetypes))
    sampled = []
    if keep > 0 and remaining:
        indices = rng.choice(len(remaining), size=min(keep, len(remaining)), replace=False)
        sampled = [remaining[int(i)] for i in indices]
    return archetypes + sampled


def rollout_candidate(env, hdv_actions, horizon, ego_action, other_action):
    tracker = EnvFeatureTracker()
    step_features = []
    if not hdv_actions:
        hdv_actions = (1,)
    segment = max(1, int(np.ceil(float(horizon) / len(hdv_actions))))

    for step in range(horizon):
        hdv = env.road.vehicles[4]
        step_features.append(tracker.observe(env, hdv))
        action_index = min(step // segment, len(hdv_actions) - 1)
        hdv_action = int(hdv_actions[action_index])

        n_agents = len(env.controlled_vehicles)
        full_action = [other_action] * max(n_agents, 3)
        full_action[3] = ego_action
        if len(full_action) > 4:
            full_action[4] = hdv_action
        else:
            full_action.append(hdv_action)
        _, _, done, _, _, _ = env.step(tuple(full_action))
        if done:
            break

    return aggregate_step_features(step_features)


def save_outputs(args, theta, training_log, expert_summary):
    os.makedirs(args.output_dir, exist_ok=True)

    theta_path = os.path.join(args.output_dir, "theta_aggressive.json")
    weights = {name: float(value) for name, value in zip(FEATURE_NAMES, theta)}
    with open(theta_path, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)

    log_path = os.path.join(args.output_dir, "theta_aggressive_training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)

    summary_path = os.path.join(args.output_dir, "aggressive_expert_summary.csv")
    expert_summary.to_csv(summary_path, index=False, encoding="utf-8")

    return theta_path, log_path, summary_path


def main():
    args = parse_args()

    expert_features, expert_summary = extract_expert_trajectory_features(
        args.expert_csv,
        aggressive_only=not args.include_non_aggressive,
    )
    if not expert_features:
        raise RuntimeError("No aggressive expert trajectories were found.")

    candidate_scenes = build_candidate_scenes(args)
    irl = MaxEntIRL(
        feature_dim=len(FEATURE_NAMES),
        n_iters=args.iters,
        lr=args.lr,
        lam=args.lam,
    )
    training_log = irl.fit(expert_features, candidate_scenes)
    theta_path, log_path, summary_path = save_outputs(
        args, irl.theta, training_log, expert_summary
    )

    print("Learned aggressive reward weights:")
    for name, value in zip(FEATURE_NAMES, irl.theta):
        print("  {:>22s}: {:.6f}".format(name, value))
    print("Saved theta to: {}".format(theta_path))
    print("Saved training log to: {}".format(log_path))
    print("Saved expert summary to: {}".format(summary_path))


if __name__ == "__main__":
    main()
