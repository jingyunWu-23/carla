"""
Fine-tune PPOmodel889 into an aggressive human-like HDV policy.

Usage:
    python train_hdv_aggressive.py
    python train_hdv_aggressive.py --reward-weights hadgen_aggressive_weights.json
"""

import argparse
import configparser
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import gym  # noqa: E402
import highway_env  # noqa: E402,F401
from highway_env.envs import merge_env_v1  # noqa: E402,F401
from stable_baselines3 import PPO  # noqa: E402

from hdv.hdv_env_wrapper import HDVEnvWrapper  # noqa: E402
from hdv.reward.aggressive_hdv_reward import AggressiveHDVReward  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune PPOmodel889 with aggressive HDV reward."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join(SCRIPT_DIR, "PPOmodel889.zip"),
        help="Warm-start PPO model path.",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "configs", "configs_ppo.ini"),
        help="Path to environment config ini.",
    )
    parser.add_argument(
        "--reward-weights",
        type=str,
        default=None,
        help="Optional JSON weights learned from aggressive-style IRL.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "hdv_models"),
        help="Directory for fine-tuned HDV model.",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default="HDV_aggressive_0",
        help="Saved model name without .zip.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Fine-tuning timesteps.",
    )
    parser.add_argument(
        "--other-action",
        type=int,
        default=1,
        help="Fixed action for adversarial CAV slots during HDV pretraining.",
    )
    parser.add_argument(
        "--ego-action",
        type=int,
        default=1,
        help="Fixed action for ego slot during HDV pretraining.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=889,
        help="Deterministic training seed.",
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


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    base_env = configure_env(gym.make("merge-multi-agent-v0"), args.config_dir)
    reward_fn = AggressiveHDVReward(weights_path=args.reward_weights)
    hdv_env = HDVEnvWrapper(
        base_env,
        reward_fn=reward_fn,
        ego_action=args.ego_action,
        other_action=args.other_action,
        deterministic_seed=args.seed,
    )

    model = PPO.load(args.model_path, env=hdv_env)
    model.learn(total_timesteps=args.timesteps)

    save_path = os.path.join(args.output_dir, args.save_name)
    model.save(save_path)
    hdv_env.close()
    print("Saved aggressive HDV model to: {}.zip".format(save_path))


if __name__ == "__main__":
    main()
