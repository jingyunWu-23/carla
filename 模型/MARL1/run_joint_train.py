"""
Joint Adversarial Training - Main Entry Point

Based on Safety-Critical Multi-Agent Joint Adversarial Framework,
using phased alternating training strategy:
    Phase 1: MAPPO trains 3 adversarial vehicles (N episodes)
    Phase 2: EgoPPO + GAE + cat-main reward trains ego vehicle (M episodes)
    -> Loop

Usage:
    python run_joint_train.py --option train
    python run_joint_train.py --option evaluate --model-dir ./joint_results/xxx
"""

import sys
sys.path.append("..")

import gym
import numpy as np
import highway_env
from highway_env.envs.merge_env_v1 import MergeEnv, MergeEnvMARL
import argparse
import configparser
import os
from datetime import datetime

from MAPPO import MAPPO
from ego.config import PPOConfig
from ego.joint_trainer import JointTrainer
from ego.reward.cat_reward import CatReward, CatSafetyReward


def parse_args():
    default_base_dir = "./joint_results/"
    default_config_dir = 'configs/configs_ppo.ini'

    parser = argparse.ArgumentParser(
        description='Joint Adversarial Training: MAPPO (adversarial) + EgoPPO (ego)'
    )
    parser.add_argument('--base-dir', type=str, default=default_base_dir,
                        help="experiment base dir")
    parser.add_argument('--option', type=str, default='train',
                        help="train or evaluate")
    parser.add_argument('--mode', type=str, default='adaptive',
                        help="training mode: fixed or adaptive")
    parser.add_argument('--n-rounds', type=int, default=100,
                        help="total training rounds")
    parser.add_argument('--adv-min-eps', type=int, default=150,
                        help="adversarial phase min episodes")
    parser.add_argument('--adv-max-eps', type=int, default=300,
                        help="adversarial phase max episodes")
    parser.add_argument('--ego-min-eps', type=int, default=300,
                        help="ego phase min episodes")
    parser.add_argument('--ego-max-eps', type=int, default=500,
                        help="ego phase max episodes")
    parser.add_argument('--adv-crash-threshold', type=float, default=0.15,
                        help="adversarial crash rate threshold for switching")
    parser.add_argument('--adv-window', type=int, default=50,
                        help="adversarial crash rate sliding window")
    parser.add_argument('--ego-window', type=int, default=100,
                        help="ego safety sliding window")
    parser.add_argument('--ego-improvement-threshold', type=float, default=0.01,
                        help="ego safety improvement threshold")
    parser.add_argument('--ppo-update-interval', type=int, default=256,
                        help="PPO network update interval (steps)")
    parser.add_argument('--config-dir', type=str, default=default_config_dir,
                        help="MAPPO config path")
    parser.add_argument('--ego-config-dir', type=str, default=None,
                        help="EgoPPO config path (optional, uses defaults if not set)")
    parser.add_argument('--model-dir', type=str, default='',
                        help="pretrained model dir for evaluation")
    parser.add_argument('--evaluation-seeds', type=str,
                        default=','.join([str(i) for i in range(0, 600, 20)]),
                        help="random seeds for evaluation, split by ,")
    parser.add_argument('--reward-preset', type=str, default='cat_safety_reward',
                        help="reward function preset: cat_default, cat_safety_merged, cat_safety_reward")
    parser.add_argument('--risk-alpha', type=float, default=1.0,
                        help="risk penalty weight for CatSafetyReward")
    parser.add_argument('--safe-distance', type=float, default=25.0,
                        help="safe distance threshold (m) for CatSafetyReward")
    parser.add_argument('--success-reward', type=float, default=30.0,
                        help="terminal success reward for CatSafetyReward")
    parser.add_argument('--no-progressive', action='store_true', default=False,
                        help="disable progressive episode allocation (use fixed ranges)")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="specific ego checkpoint to evaluate (e.g. checkpoint-28500.pt)")
    return parser.parse_args()


def build_mappo(config, env, test_seeds):
    BATCH_SIZE = config.getint('MODEL_CONFIG', 'BATCH_SIZE')
    MEMORY_CAPACITY = config.getint('MODEL_CONFIG', 'MEMORY_CAPACITY')
    ROLL_OUT_N_STEPS = config.getint('MODEL_CONFIG', 'ROLL_OUT_N_STEPS')
    reward_gamma = config.getfloat('MODEL_CONFIG', 'reward_gamma')
    actor_hidden_size = config.getint('MODEL_CONFIG', 'actor_hidden_size')
    critic_hidden_size = config.getint('MODEL_CONFIG', 'critic_hidden_size')
    MAX_GRAD_NORM = config.getfloat('MODEL_CONFIG', 'MAX_GRAD_NORM')
    ENTROPY_REG = config.getfloat('MODEL_CONFIG', 'ENTROPY_REG')
    reward_type = config.get('MODEL_CONFIG', 'reward_type')
    TARGET_UPDATE_STEPS = config.getint('MODEL_CONFIG', 'TARGET_UPDATE_STEPS')
    TARGET_TAU = config.getfloat('MODEL_CONFIG', 'TARGET_TAU')
    actor_lr = config.getfloat('TRAIN_CONFIG', 'actor_lr')
    critic_lr = config.getfloat('TRAIN_CONFIG', 'critic_lr')
    EPISODES_BEFORE_TRAIN = config.getint('TRAIN_CONFIG', 'EPISODES_BEFORE_TRAIN')
    reward_scale = config.getfloat('TRAIN_CONFIG', 'reward_scale')
    traffic_density = config.getint('ENV_CONFIG', 'traffic_density')

    state_dim = env.n_s
    action_dim = env.n_a

    mappo = MAPPO(
        env=env, memory_capacity=MEMORY_CAPACITY,
        state_dim=state_dim, action_dim=action_dim,
        batch_size=BATCH_SIZE, entropy_reg=ENTROPY_REG,
        roll_out_n_steps=ROLL_OUT_N_STEPS,
        actor_hidden_size=actor_hidden_size, critic_hidden_size=critic_hidden_size,
        actor_lr=actor_lr, critic_lr=critic_lr, reward_scale=reward_scale,
        target_update_steps=TARGET_UPDATE_STEPS, target_tau=TARGET_TAU,
        reward_gamma=reward_gamma, reward_type=reward_type,
        max_grad_norm=MAX_GRAD_NORM, test_seeds=test_seeds,
        episodes_before_train=EPISODES_BEFORE_TRAIN, traffic_density=traffic_density,
    )
    return mappo


def build_ego_config(config, env):
    ego_config = PPOConfig(
        state_dim=env.n_s,
        action_dim=env.n_a,
        action_space_type='discrete',
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        reward_gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        actor_lr=3e-4,
        critic_lr=1e-3,
        actor_hidden_size=128,
        critic_hidden_size=128,
        max_grad_norm=0.5,
        max_episodes=100000,
        eval_interval=10,
        save_interval=50,
        eval_episodes=5,
    )
    return ego_config


def train(args):
    base_dir = args.base_dir
    config_dir = args.config_dir
    config = configparser.ConfigParser()
    config.read(config_dir)

    env = gym.make('merge-multi-agent-v0')
    env.config['seed'] = config.getint('ENV_CONFIG', 'seed')
    env.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')
    env.config['duration'] = config.getint('ENV_CONFIG', 'duration')
    env.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')
    env.config['COLLISION_REWARD'] = config.getint('ENV_CONFIG', 'COLLISION_REWARD')
    env.config['HIGH_SPEED_REWARD'] = config.getint('ENV_CONFIG', 'HIGH_SPEED_REWARD')
    env.config['HEADWAY_COST'] = config.getint('ENV_CONFIG', 'HEADWAY_COST')
    env.config['HEADWAY_TIME'] = config.getfloat('ENV_CONFIG', 'HEADWAY_TIME')
    env.config['MERGING_LANE_COST'] = config.getint('ENV_CONFIG', 'MERGING_LANE_COST')
    env.config['traffic_density'] = config.getint('ENV_CONFIG', 'traffic_density')
    env.config['action_masking'] = config.getboolean('MODEL_CONFIG', 'action_masking')

    test_seeds = args.evaluation_seeds

    mappo = build_mappo(config, env, test_seeds)

    ego_config = build_ego_config(config, env)

    try:
        from stable_baselines3 import PPO
        natural_model = PPO.load("PPOmodel889")
        print("Natural vehicle model loaded: PPOmodel889")
    except Exception as e:
        print(f"Warning: Could not load natural vehicle model: {e}")
        natural_model = None

    if args.reward_preset == 'cat_safety_reward':
        ego_reward_fn = CatSafetyReward(
            driving_reward=1.0,
            speed_reward=0.1,
            success_reward=args.success_reward,
            out_of_road_penalty=5.0,
            crash_vehicle_penalty=5.0,
            alpha=args.risk_alpha,
            safe_distance=args.safe_distance,
        )
    elif args.reward_preset == 'cat_safety_merged':
        from ego.reward.reward_factory import RewardFactory
        ego_reward_fn = RewardFactory.cat_safety_merged()
    else:
        ego_reward_fn = CatReward(
            driving_reward=1.0,
            speed_reward=0.1,
            success_reward=10.0,
            out_of_road_penalty=5.0,
            crash_vehicle_penalty=5.0,
        )

    trainer = JointTrainer(
        env=env,
        mappo=mappo,
        ego_config=ego_config,
        natural_vehicle_model=natural_model,
        ego_reward_fn=ego_reward_fn,
    )

    print("=" * 60)
    print("Joint Adversarial Training Configuration")
    print("=" * 60)
    print(f"  Mode:                 {args.mode}")
    print(f"  Adversarial vehicles: 3 (MAPPO)")
    print(f"  Ego vehicle:          1 (EgoPPO + GAE)")
    print(f"  Natural vehicle:      1 (frozen PPO)")
    print(f"  Reward function:      {type(ego_reward_fn).__name__}")
    if isinstance(ego_reward_fn, CatSafetyReward):
        print(f"    risk_alpha:         {ego_reward_fn.alpha}")
        print(f"    safe_distance:      {ego_reward_fn.safe_distance} m")
        print(f"    success_reward:     {ego_reward_fn.terminal.success_reward}")
    print(f"  Training strategy:    {'Adaptive switching' if args.mode == 'adaptive' else 'Phased alternating'}")
    print(f"  Output dir:           {base_dir}")
    print("=" * 60)

    if args.mode == 'adaptive':
        trainer.train_adaptive(
            n_rounds=args.n_rounds,
            adv_min_eps=args.adv_min_eps,
            adv_max_eps=args.adv_max_eps,
            ego_min_eps=args.ego_min_eps,
            ego_max_eps=args.ego_max_eps,
            adv_crash_threshold=args.adv_crash_threshold,
            adv_window=args.adv_window,
            ego_window=args.ego_window,
            ego_improvement_threshold=args.ego_improvement_threshold,
            ppo_update_interval=args.ppo_update_interval,
            eval_interval=10,
            save_interval=50,
            progressive=not args.no_progressive,
        )
    else:
        trainer.train(
            n_adversarial_episodes=100,
            n_ego_episodes=50,
            n_rounds=10,
            eval_interval=10,
            save_interval=50,
        )


def evaluate(args):
    if not os.path.exists(args.model_dir):
        raise Exception(f"Model directory not found: {args.model_dir}")

    config_dir = args.config_dir
    config = configparser.ConfigParser()
    config.read(config_dir)

    env = gym.make('merge-multi-agent-v0')
    env.config['seed'] = config.getint('ENV_CONFIG', 'seed')
    env.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')
    env.config['duration'] = config.getint('ENV_CONFIG', 'duration')
    env.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')
    env.config['traffic_density'] = config.getint('ENV_CONFIG', 'traffic_density')
    env.config['action_masking'] = config.getboolean('MODEL_CONFIG', 'action_masking')

    test_seeds = args.evaluation_seeds

    mappo = build_mappo(config, env, test_seeds)
    mappo_dir = os.path.join(args.model_dir, "mappo_models")
    mappo.load(mappo_dir, train_mode=False)

    ego_config = build_ego_config(config, env)

    try:
        from stable_baselines3 import PPO
        natural_model = PPO.load("PPOmodel889")
    except Exception:
        natural_model = None

    trainer = JointTrainer(
        env=env,
        mappo=mappo,
        ego_config=ego_config,
        natural_vehicle_model=natural_model,
    )

    ego_model_dir = os.path.join(args.model_dir, "ego_models")
    if args.checkpoint:
        ego_checkpoint = os.path.join(ego_model_dir, args.checkpoint)
    else:
        ego_checkpoint = os.path.join(ego_model_dir, "final_model.pt")
    if os.path.exists(ego_checkpoint):
        trainer.ego_ppo.load(ego_checkpoint)
        print(f"Ego model loaded from {ego_checkpoint}")
    else:
        print(f"WARNING: Checkpoint not found: {ego_checkpoint}")

    print("\nEvaluating ego vehicle policy...")
    eval_result = trainer.evaluate_ego(n_episodes=30, render=False)

    print(f"\nEvaluation Results:")
    print(f"  Mean Reward:         {eval_result['mean_reward']:.2f} +/- {eval_result['std_reward']:.2f}")
    print(f"  Mean Length:         {eval_result['mean_length']:.1f}")
    print(f"  Crash Rate:          {eval_result['crash_rate']:.1%}")
    print(f"  Road Completion Rate:{eval_result['road_completion_rate']:.1%}")


if __name__ == "__main__":
    args = parse_args()

    if args.option == 'train':
        train(args)
    elif args.option == 'evaluate':
        evaluate(args)
    else:
        raise ValueError(f"Unknown option: {args.option}. Use 'train' or 'evaluate'.")