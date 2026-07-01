"""
Small-scale CAV-HDV co-evolution experiment.

This reuses the existing JointTrainer for MAPPO adversarial training and EgoPPO
training, then adds a short HDV PPO fine-tuning phase using the learned
aggressive reward. The default settings are intentionally small so the pipeline
can be tested before long runs.
"""

import argparse
import configparser
import csv
import os
import sys
from datetime import datetime
from copy import deepcopy

import torch as th


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

from MAPPO import MAPPO  # noqa: E402
from single_agent.Memory_common import OnPolicyReplayMemory  # noqa: E402
from ego.config import PPOConfig  # noqa: E402
from ego.joint_trainer import JointTrainer  # noqa: E402
from ego.reward.cat_reward import CatSafetyReward  # noqa: E402
from hdv.coevolution_hdv_env_wrapper import CoevolutionHDVEnvWrapper  # noqa: E402
from hdv.reward.aggressive_hdv_reward import AggressiveHDVReward  # noqa: E402
from run_joint_train import build_mappo, build_ego_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Small-scale three-policy co-evolution.")
    parser.add_argument("--config-dir", type=str, default=os.path.join(SCRIPT_DIR, "configs", "configs_ppo.ini"))
    parser.add_argument("--base-dir", type=str, default=os.path.join(SCRIPT_DIR, "coevolution_results"))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--adv-episodes", type=int, default=10)
    parser.add_argument("--ego-episodes", type=int, default=10)
    parser.add_argument("--hdv-timesteps", type=int, default=2000)
    parser.add_argument(
        "--stage-schedule",
        type=str,
        default=None,
        help=(
            "Optional staged penetration schedule: "
            "name:rounds:traffic_density:adv_eps:ego_eps:hdv_steps[:num_cav:num_hdv],"
            "e.g. low:5:1:50:50:5000:1:3,mid:5:1:50:50:5000:2:2,high:5:1:50:50:5000:3:1"
        ),
    )
    parser.add_argument(
        "--fixed-penetration-cycle",
        action="store_true",
        default=False,
        help="Use the default fixed penetration curriculum: 25% -> 50% -> 75% CAV.",
    )
    parser.add_argument(
        "--rounds-per-penetration",
        type=int,
        default=5,
        help="Rounds per penetration stage when --fixed-penetration-cycle is used.",
    )
    parser.add_argument("--hdv-model", type=str, default=os.path.join(SCRIPT_DIR, "hdv_models", "HDV_aggressive_0.zip"))
    parser.add_argument("--hdv-reward", type=str, default=os.path.join(SCRIPT_DIR, "hdv_reward_irl", "run_15_300", "theta_aggressive.json"))
    parser.add_argument("--ego-checkpoint", type=str, default=None)
    parser.add_argument("--mappo-dir", type=str, default=None)
    parser.add_argument("--mappo-checkpoint", type=str, default=None)
    parser.add_argument("--freeze-hdv", action="store_true", default=False)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument(
        "--forget-eval-episodes",
        type=int,
        default=5,
        help="Episodes used to evaluate retained performance on previous penetration stages.",
    )
    parser.add_argument(
        "--forget-eval-interval",
        type=int,
        default=1,
        help="Run forgetting evaluation every N co-evolution rounds. Set <=0 to disable.",
    )
    parser.add_argument(
        "--forget-retention-threshold",
        type=float,
        default=None,
        help="Minimum reward retention ratio on replayed previous penetrations required for switching.",
    )
    parser.add_argument(
        "--forget-crash-increase-threshold",
        type=float,
        default=None,
        help="Maximum crash-rate increase on replayed previous penetrations required for switching.",
    )
    parser.add_argument(
        "--forget-completion-drop-threshold",
        type=float,
        default=None,
        help="Maximum road-completion drop on replayed previous penetrations required for switching.",
    )
    parser.add_argument(
        "--replay-train-interval",
        type=int,
        default=1,
        help="Run replay training every N co-evolution rounds. Set <=0 to disable.",
    )
    parser.add_argument(
        "--replay-train-ego-episodes",
        type=int,
        default=0,
        help="EgoPPO rehearsal episodes per replayed previous penetration stage.",
    )
    parser.add_argument(
        "--replay-train-adv-episodes",
        type=int,
        default=0,
        help="MAPPO rehearsal episodes per replayed previous penetration stage.",
    )
    parser.add_argument(
        "--replay-train-hdv-timesteps",
        type=int,
        default=0,
        help="HDV PPO rehearsal timesteps per replayed previous penetration stage.",
    )
    parser.add_argument(
        "--replay-train-max-stages",
        type=int,
        default=0,
        help="Maximum previous penetration stages replayed each round. <=0 replays all previous stages.",
    )
    parser.add_argument(
        "--replay-train-selection",
        type=str,
        default="lowest-retention",
        choices=["lowest-retention", "recent"],
        help="Replay stage selection strategy: prioritize lowest retained reward or most recent stages.",
    )
    parser.add_argument(
        "--replay-train-priority-eval-episodes",
        type=int,
        default=0,
        help="Episodes used to rank replay stages by retention. <=0 reuses --forget-eval-episodes.",
    )
    parser.add_argument(
        "--adaptive-stage-switch",
        action="store_true",
        default=False,
        help="Switch to the next penetration stage once current-stage performance reaches thresholds.",
    )
    parser.add_argument(
        "--performance-eval-episodes",
        type=int,
        default=5,
        help="Episodes used to evaluate whether the current penetration stage has been learned.",
    )
    parser.add_argument(
        "--performance-reward-threshold",
        type=float,
        default=None,
        help="Mean ego reward threshold for considering the current penetration learned.",
    )
    parser.add_argument(
        "--performance-crash-threshold",
        type=float,
        default=None,
        help="Maximum ego crash rate for considering the current penetration learned.",
    )
    parser.add_argument(
        "--performance-completion-threshold",
        type=float,
        default=None,
        help="Minimum road completion rate for considering the current penetration learned.",
    )
    parser.add_argument(
        "--performance-patience",
        type=int,
        default=1,
        help="Consecutive successful performance evaluations required before switching penetration stages.",
    )
    parser.add_argument("--seed", type=int, default=669)
    return parser.parse_args()


def parse_stage_schedule(args):
    if args.fixed_penetration_cycle:
        rounds = args.rounds_per_penetration
        return [
            _make_stage("cav_25", rounds, 1, args.adv_episodes, args.ego_episodes,
                        args.hdv_timesteps, num_cav=1, num_hdv=3),
            _make_stage("cav_50", rounds, 1, args.adv_episodes, args.ego_episodes,
                        args.hdv_timesteps, num_cav=2, num_hdv=2),
            _make_stage("cav_75", rounds, 1, args.adv_episodes, args.ego_episodes,
                        args.hdv_timesteps, num_cav=3, num_hdv=1),
        ]

    if not args.stage_schedule:
        return [
            _make_stage("default", args.rounds, None, args.adv_episodes,
                        args.ego_episodes, args.hdv_timesteps)
        ]

    stages = []
    for raw_stage in args.stage_schedule.split(","):
        raw_stage = raw_stage.strip()
        if not raw_stage:
            continue
        parts = raw_stage.split(":")
        if len(parts) not in (6, 8):
            raise ValueError(
                "Invalid stage '{}'. Expected name:rounds:traffic_density:adv_eps:ego_eps:hdv_steps[:num_cav:num_hdv]".format(
                    raw_stage
                )
            )
        name, rounds, traffic_density, adv_eps, ego_eps, hdv_steps = parts
        num_cav = int(parts[6]) if len(parts) == 8 else None
        num_hdv = int(parts[7]) if len(parts) == 8 else None
        stages.append(_make_stage(name, int(rounds), int(traffic_density),
                                  int(adv_eps), int(ego_eps), int(hdv_steps),
                                  num_cav=num_cav, num_hdv=num_hdv))
    if not stages:
        raise ValueError("Empty stage schedule.")
    return stages


def _make_stage(name, rounds, traffic_density, adv_episodes, ego_episodes,
                hdv_timesteps, num_cav=None, num_hdv=None):
    penetration = None
    if num_cav is not None and num_hdv is not None and (num_cav + num_hdv) > 0:
        penetration = float(num_cav) / float(num_cav + num_hdv)
    return {
        "name": name,
        "rounds": rounds,
        "traffic_density": traffic_density,
        "adv_episodes": adv_episodes,
        "ego_episodes": ego_episodes,
        "hdv_timesteps": hdv_timesteps,
        "num_cav": num_cav,
        "num_hdv": num_hdv,
        "cav_penetration": penetration,
    }


def apply_penetration_stage(env, mappo, stage):
    """Apply the target CAV/HDV composition to all future environment resets."""
    if stage["traffic_density"] is not None:
        env.config["traffic_density"] = stage["traffic_density"]
        mappo.traffic_density = stage["traffic_density"]

    if stage["num_cav"] is None:
        env.config.pop("num_CAV", None)
    else:
        env.config["num_CAV"] = int(stage["num_cav"])

    if stage["num_hdv"] is None:
        env.config.pop("num_HDV", None)
    else:
        env.config["num_HDV"] = int(stage["num_hdv"])


def reset_mappo_env(env, mappo, clear_memory=True):
    """Reset MAPPO rollout state after a penetration change."""
    if clear_memory:
        mappo.memory = OnPolicyReplayMemory(mappo.memory.capacity)
    mappo.env_state, _, mappo.obs2, mappo.obs3 = env.reset()
    mappo.n_steps = 0
    mappo.episode_done = False
    mappo.n_agents = len(env.controlled_vehicles)


class PenetrationReplayBuffer:
    """Stores learned penetration stages and their baseline evaluation metrics."""

    def __init__(self):
        self.items = []

    def add(self, stage, baseline_metrics, global_round):
        item = {
            "stage": deepcopy(stage),
            "baseline_metrics": dict(baseline_metrics),
            "added_round": global_round,
        }
        self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)


def configure_env(config_path):
    config = configparser.ConfigParser()
    read_files = config.read(config_path)
    if not read_files:
        raise FileNotFoundError("Config file was not found: {}".format(config_path))

    env = gym.make("merge-multi-agent-v0")
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
    return env, config


def load_optional_models(args, trainer, mappo):
    if args.mappo_checkpoint:
        checkpoint = th.load(args.mappo_checkpoint)
        mappo.actor.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            mappo.actor_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        mappo.actor.train()
        print("MAPPO checkpoint loaded: {}".format(args.mappo_checkpoint))
    elif args.mappo_dir:
        loaded = mappo.load(args.mappo_dir, train_mode=True)
        print("MAPPO loaded: {}".format(loaded))
    if args.ego_checkpoint and os.path.exists(args.ego_checkpoint):
        trainer.ego_ppo.load(args.ego_checkpoint)
        print("Ego checkpoint loaded: {}".format(args.ego_checkpoint))


def main():
    args = parse_args()
    env, config = configure_env(args.config_dir)
    test_seeds = ",".join(str(i) for i in range(0, 600, 20))

    mappo = build_mappo(config, env, test_seeds)
    ego_config = build_ego_config(config, env)
    ego_config.n_steps = 512
    ego_config.batch_size = 64

    hdv_model = PPO.load(args.hdv_model)

    ego_reward_fn = CatSafetyReward(
        driving_reward=1.0,
        speed_reward=0.1,
        success_reward=30.0,
        out_of_road_penalty=5.0,
        crash_vehicle_penalty=5.0,
        alpha=1.0,
        safe_distance=25.0,
    )

    trainer = JointTrainer(
        env=env,
        mappo=mappo,
        ego_config=ego_config,
        natural_vehicle_model=hdv_model,
        ego_reward_fn=ego_reward_fn,
    )
    load_optional_models(args, trainer, mappo)
    stages = parse_stage_schedule(args)

    output_dir = os.path.join(args.base_dir, datetime.utcnow().strftime("%b_%d_%H_%M_%S"))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "hdv_models"), exist_ok=True)
    trainer.output_dir = output_dir
    os.makedirs(os.path.join(output_dir, "mappo_models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ego_models"), exist_ok=True)
    log_path = os.path.join(output_dir, "coevolution_log.csv")
    log_fieldnames = [
        "stage", "stage_round", "round", "target_num_cav", "target_num_hdv",
        "target_cav_penetration", "actual_num_cav", "actual_num_hdv",
        "actual_cav_penetration", "traffic_density", "phase", "episode", "total_adversarial_episodes",
        "total_ego_episodes", "adversarial_reward", "ego_reward",
        "ego_policy_loss", "ego_value_loss", "ego_crash_rate",
        "ego_road_completion_rate", "ego_avg_speed", "adversarial_crash_rate",
    ]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=log_fieldnames).writeheader()

    forgetting_log_path = os.path.join(output_dir, "forgetting_eval_log.csv")
    forgetting_fieldnames = [
        "round", "current_stage", "current_stage_round", "replay_stage",
        "replay_num_cav", "replay_num_hdv", "replay_cav_penetration",
        "eval_episodes", "baseline_round", "baseline_reward",
        "current_reward", "reward_drop", "reward_retention_ratio",
        "baseline_crash_rate", "current_crash_rate", "crash_rate_increase",
        "baseline_completion", "current_completion", "completion_drop",
        "mean_length", "reward_std",
    ]
    with open(forgetting_log_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=forgetting_fieldnames).writeheader()

    performance_log_path = os.path.join(output_dir, "performance_eval_log.csv")
    performance_fieldnames = [
        "round", "stage", "stage_round", "target_num_cav", "target_num_hdv",
        "target_cav_penetration", "eval_episodes", "mean_reward",
        "std_reward", "crash_rate", "road_completion_rate", "mean_length",
        "reward_threshold", "crash_threshold", "completion_threshold",
        "reward_pass", "crash_pass", "completion_pass", "stage_learned",
    ]
    with open(performance_log_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=performance_fieldnames).writeheader()

    replay_train_log_path = os.path.join(output_dir, "replay_train_log.csv")
    replay_train_fieldnames = [
        "replay_context", "current_stage", "current_stage_round",
        "replay_stage", "replay_num_cav", "replay_num_hdv",
        "replay_cav_penetration", "replay_selection_strategy",
        "replay_priority_rank", "replay_priority_retention_ratio",
        "replay_priority_current_reward", "replay_priority_baseline_reward",
        "replay_priority_crash_rate_increase",
        "replay_priority_completion_drop",
    ] + log_fieldnames
    with open(replay_train_log_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=replay_train_fieldnames).writeheader()

    print("=" * 70)
    print("SMALL-SCALE CAV-HDV CO-EVOLUTION")
    print("=" * 70)
    print("Stages:          {}".format(stages))
    print("Total rounds:    {}".format(sum(stage["rounds"] for stage in stages)))
    print("HDV model:       {}".format(args.hdv_model))
    print("HDV reward:      {}".format(args.hdv_reward))
    print("Output dir:      {}".format(output_dir))
    print("=" * 70)

    global_round = 0
    replay_buffer = PenetrationReplayBuffer()
    for stage in stages:
        apply_penetration_stage(env, mappo, stage)
        reset_mappo_env(env, mappo, clear_memory=True)
        actual_num_cav, actual_num_hdv, actual_penetration = current_penetration(env)
        print("\n" + "=" * 70)
        print("STAGE {} | rounds={} | traffic_density={} | adv={} | ego={} | hdv_steps={}".format(
            stage["name"],
            stage["rounds"],
            env.config.get("traffic_density"),
            stage["adv_episodes"],
            stage["ego_episodes"],
            0 if args.freeze_hdv else stage["hdv_timesteps"],
        ))
        print("TARGET PENETRATION | num_CAV={} | num_HDV={} | CAV penetration={}".format(
            stage["num_cav"],
            stage["num_hdv"],
            stage["cav_penetration"],
        ))
        print("ACTUAL PENETRATION | num_CAV={} | num_HDV={} | CAV penetration={}".format(
            actual_num_cav,
            actual_num_hdv,
            actual_penetration,
        ))
        print("=" * 70)

        consecutive_passes = 0
        for stage_round in range(stage["rounds"]):
            global_round += 1
            print("\n========== Stage {} Round {} / {} | Global Round {} ==========".format(
                stage["name"], stage_round + 1, stage["rounds"], global_round
            ))

            trainer._round = global_round
            trainer._print_phase_header("ADVERSARIAL", stage["adv_episodes"])
            adv_info = trainer._train_adversarial_phase(
            n_episodes=stage["adv_episodes"],
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            log_path=log_path,
            fieldnames=log_fieldnames,
            )

            trainer._print_phase_header("EGO", stage["ego_episodes"])
            ego_info = trainer._train_ego_phase(
            n_episodes=stage["ego_episodes"],
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            log_path=log_path,
            fieldnames=log_fieldnames,
            )

            if not args.freeze_hdv and stage["hdv_timesteps"] > 0:
                print("\n{} HDV FINE-TUNING {}".format("#" * 20, "#" * 20))
                hdv_reward = AggressiveHDVReward(weights_path=args.hdv_reward)
                hdv_env = CoevolutionHDVEnvWrapper(
                    env=env,
                    mappo=mappo,
                    ego_ppo=trainer.ego_ppo,
                    reward_fn=hdv_reward,
                    hdv_policy_model=hdv_model,
                    deterministic_seed=args.seed + 1000 * global_round,
                )
                hdv_model.set_env(hdv_env)
                hdv_model.learn(total_timesteps=stage["hdv_timesteps"], reset_num_timesteps=False)
                trainer.natural_vehicle_model = hdv_model
                trainer.ego_wrapper.natural_vehicle_model = hdv_model
                reset_mappo_env(env, mappo, clear_memory=True)

                hdv_save_path = os.path.join(
                    output_dir,
                    "hdv_models",
                    "{}_HDV_round_{}".format(stage["name"], stage_round + 1),
                )
                hdv_model.save(hdv_save_path)
                print("HDV model saved to: {}.zip".format(hdv_save_path))

            replay_training_enabled = (
                args.replay_train_interval > 0
                and len(replay_buffer) > 0
                and (
                    args.replay_train_ego_episodes > 0
                    or args.replay_train_adv_episodes > 0
                    or args.replay_train_hdv_timesteps > 0
                )
            )
            if replay_training_enabled and global_round % args.replay_train_interval == 0:
                hdv_model = run_replay_training(
                    trainer=trainer,
                    env=env,
                    mappo=mappo,
                    hdv_model=hdv_model,
                    replay_buffer=replay_buffer,
                    current_stage=stage,
                    current_stage_round=stage_round + 1,
                    global_round=global_round,
                    args=args,
                    replay_log_path=replay_train_log_path,
                    replay_fieldnames=replay_train_fieldnames,
                    output_dir=output_dir,
                )
                apply_penetration_stage(env, mappo, stage)
                reset_mappo_env(env, mappo, clear_memory=True)

            annotate_latest_log_rows(
                log_path=log_path,
                stage=stage,
                stage_round=stage_round + 1,
                actual_num_cav=actual_num_cav,
                actual_num_hdv=actual_num_hdv,
                actual_penetration=actual_penetration,
            )

            forget_gate_enabled = any(
                value is not None
                for value in (
                    args.forget_retention_threshold,
                    args.forget_crash_increase_threshold,
                    args.forget_completion_drop_threshold,
                )
            )

            if not forget_gate_enabled and args.forget_eval_interval > 0 and len(replay_buffer) > 0 and \
                    (global_round % args.forget_eval_interval == 0):
                evaluate_forgetting_replay(
                    trainer=trainer,
                    env=env,
                    mappo=mappo,
                    replay_buffer=replay_buffer,
                    current_stage=stage,
                    current_stage_round=stage_round + 1,
                    global_round=global_round,
                    eval_episodes=args.forget_eval_episodes,
                    log_path=forgetting_log_path,
                    fieldnames=forgetting_fieldnames,
                )
                apply_penetration_stage(env, mappo, stage)
                reset_mappo_env(env, mappo, clear_memory=True)

            stage_learned = False
            forgetting_pass = True
            if args.adaptive_stage_switch:
                performance_metrics = evaluate_current_stage_performance(
                    trainer=trainer,
                    env=env,
                    mappo=mappo,
                    stage=stage,
                    stage_round=stage_round + 1,
                    global_round=global_round,
                    eval_episodes=args.performance_eval_episodes,
                    reward_threshold=args.performance_reward_threshold,
                    crash_threshold=args.performance_crash_threshold,
                    completion_threshold=args.performance_completion_threshold,
                    log_path=performance_log_path,
                    fieldnames=performance_fieldnames,
                )
                stage_learned = performance_metrics["stage_learned"]
                if len(replay_buffer) > 0 and forget_gate_enabled:
                    forgetting_metrics = evaluate_forgetting_replay(
                        trainer=trainer,
                        env=env,
                        mappo=mappo,
                        replay_buffer=replay_buffer,
                        current_stage=stage,
                        current_stage_round=stage_round + 1,
                        global_round=global_round,
                        eval_episodes=args.forget_eval_episodes,
                        log_path=forgetting_log_path,
                        fieldnames=forgetting_fieldnames,
                    )
                    forgetting_pass = forgetting_replay_passes(
                        forgetting_metrics,
                        retention_threshold=args.forget_retention_threshold,
                        crash_increase_threshold=args.forget_crash_increase_threshold,
                        completion_drop_threshold=args.forget_completion_drop_threshold,
                    )
                    print("  [FORGET-GATE] pass={} | min_retention={:.3f} max_crash_inc={:.3f} max_completion_drop={:.3f}".format(
                        forgetting_pass,
                        forgetting_metrics.get("min_reward_retention_ratio", 0.0),
                        forgetting_metrics.get("max_crash_rate_increase", 0.0),
                        forgetting_metrics.get("max_completion_drop", 0.0),
                    ))
                if stage_learned:
                    consecutive_passes += 1
                else:
                    consecutive_passes = 0
                stage_learned = (
                    consecutive_passes >= max(args.performance_patience, 1)
                    and forgetting_pass
                )
                apply_penetration_stage(env, mappo, stage)
                reset_mappo_env(env, mappo, clear_memory=True)

            trainer._print_round_summary(adv_info, ego_info)
            if stage_learned:
                print("Stage {} learned at round {} after {} consecutive successful evaluations. Switching to next penetration.".format(
                    stage["name"], stage_round + 1, consecutive_passes
                ))
                break

        baseline_metrics = evaluate_stage_baseline(
            trainer=trainer,
            env=env,
            mappo=mappo,
            stage=stage,
            eval_episodes=args.forget_eval_episodes,
        )
        replay_buffer.add(stage, baseline_metrics, global_round)
        apply_penetration_stage(env, mappo, stage)
        reset_mappo_env(env, mappo, clear_memory=True)
        print("Replay buffer added stage {} | baseline reward={:.2f} crash={:.3f} completion={:.3f}".format(
            stage["name"],
            baseline_metrics.get("mean_reward", 0.0),
            baseline_metrics.get("crash_rate", 0.0),
            baseline_metrics.get("road_completion_rate", 0.0),
        ))

    trainer._save_final_models()
    final_hdv_path = os.path.join(output_dir, "hdv_models", "HDV_final")
    hdv_model.save(final_hdv_path)
    print("Final HDV model saved to: {}.zip".format(final_hdv_path))
    print("Co-evolution complete: {}".format(output_dir))


def current_penetration(env):
    num_cav = len(getattr(env, "controlled_vehicles", []))
    hdvs = getattr(env, "hdv_vehicles", None)
    num_hdv = len(hdvs) if hdvs is not None else max(len(env.road.vehicles) - num_cav - 1, 0)
    total = num_cav + num_hdv
    penetration = float(num_cav) / float(total) if total > 0 else None
    return num_cav, num_hdv, penetration


def evaluate_stage_baseline(trainer, env, mappo, stage, eval_episodes):
    apply_penetration_stage(env, mappo, stage)
    reset_mappo_env(env, mappo, clear_memory=True)
    return trainer.evaluate_ego(n_episodes=eval_episodes, render=False)


def evaluate_current_stage_performance(trainer, env, mappo, stage, stage_round,
                                       global_round, eval_episodes,
                                       reward_threshold, crash_threshold,
                                       completion_threshold, log_path,
                                       fieldnames):
    apply_penetration_stage(env, mappo, stage)
    reset_mappo_env(env, mappo, clear_memory=True)
    metrics = trainer.evaluate_ego(n_episodes=eval_episodes, render=False)

    mean_reward = float(metrics.get("mean_reward", 0.0))
    crash_rate = float(metrics.get("crash_rate", 0.0))
    completion = float(metrics.get("road_completion_rate", 0.0))

    reward_pass = True if reward_threshold is None else mean_reward >= reward_threshold
    crash_pass = True if crash_threshold is None else crash_rate <= crash_threshold
    completion_pass = True if completion_threshold is None else completion >= completion_threshold

    has_threshold = any(
        value is not None
        for value in (reward_threshold, crash_threshold, completion_threshold)
    )
    stage_learned = bool(has_threshold and reward_pass and crash_pass and completion_pass)

    row = {
        "round": global_round,
        "stage": stage["name"],
        "stage_round": stage_round,
        "target_num_cav": stage["num_cav"],
        "target_num_hdv": stage["num_hdv"],
        "target_cav_penetration": stage["cav_penetration"],
        "eval_episodes": eval_episodes,
        "mean_reward": mean_reward,
        "std_reward": metrics.get("std_reward", 0.0),
        "crash_rate": crash_rate,
        "road_completion_rate": completion,
        "mean_length": metrics.get("mean_length", 0.0),
        "reward_threshold": reward_threshold,
        "crash_threshold": crash_threshold,
        "completion_threshold": completion_threshold,
        "reward_pass": reward_pass,
        "crash_pass": crash_pass,
        "completion_pass": completion_pass,
        "stage_learned": stage_learned,
    }
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

    print("  [PERF] {} R{} | reward={:.2f} crash={:.3f} completion={:.3f} | learned={}".format(
        stage["name"], stage_round, mean_reward, crash_rate, completion, stage_learned
    ))
    metrics.update(row)
    return metrics


def compute_replay_priority_metrics(trainer, env, mappo, item, eval_episodes):
    replay_stage = item["stage"]
    baseline = item["baseline_metrics"]

    apply_penetration_stage(env, mappo, replay_stage)
    reset_mappo_env(env, mappo, clear_memory=True)
    metrics = trainer.evaluate_ego(n_episodes=eval_episodes, render=False)

    baseline_reward = float(baseline.get("mean_reward", 0.0))
    current_reward = float(metrics.get("mean_reward", 0.0))
    retention_ratio = current_reward / baseline_reward if abs(baseline_reward) > 1e-8 else 0.0

    baseline_crash = float(baseline.get("crash_rate", 0.0))
    current_crash = float(metrics.get("crash_rate", 0.0))
    baseline_completion = float(baseline.get("road_completion_rate", 0.0))
    current_completion = float(metrics.get("road_completion_rate", 0.0))

    return {
        "baseline_reward": baseline_reward,
        "current_reward": current_reward,
        "retention_ratio": retention_ratio,
        "crash_rate_increase": current_crash - baseline_crash,
        "completion_drop": baseline_completion - current_completion,
        "current_crash_rate": current_crash,
        "current_completion": current_completion,
    }


def select_replay_items(replay_buffer, max_stages, selection_strategy="recent",
                        trainer=None, env=None, mappo=None, eval_episodes=1):
    items = list(replay_buffer)
    ranked_items = []

    if selection_strategy == "lowest-retention" and trainer is not None:
        eval_episodes = max(int(eval_episodes), 1)
        print("  [REPLAY-SELECT] ranking {} stages by reward retention (eval_episodes={})".format(
            len(items), eval_episodes
        ))
        for item in items:
            priority = compute_replay_priority_metrics(
                trainer=trainer,
                env=env,
                mappo=mappo,
                item=item,
                eval_episodes=eval_episodes,
            )
            ranked_items.append({
                "item": item,
                "priority": priority,
            })
            print("    candidate={} | retention={:.3f} reward={:.2f}/{:.2f} crash_inc={:.3f} completion_drop={:.3f}".format(
                item["stage"]["name"],
                priority["retention_ratio"],
                priority["current_reward"],
                priority["baseline_reward"],
                priority["crash_rate_increase"],
                priority["completion_drop"],
            ))

        ranked_items.sort(
            key=lambda x: (
                x["priority"]["retention_ratio"],
                -x["priority"]["crash_rate_increase"],
                -x["priority"]["completion_drop"],
            )
        )
    else:
        if max_stages is not None and max_stages > 0:
            items = items[-max_stages:]
        ranked_items = [
            {
                "item": item,
                "priority": {
                    "baseline_reward": "",
                    "current_reward": "",
                    "retention_ratio": "",
                    "crash_rate_increase": "",
                    "completion_drop": "",
                },
            }
            for item in items
        ]

    if max_stages is not None and max_stages > 0:
        ranked_items = ranked_items[:max_stages]

    for rank, ranked in enumerate(ranked_items, start=1):
        ranked["priority"]["rank"] = rank
    return ranked_items


def run_replay_training(trainer, env, mappo, hdv_model, replay_buffer,
                        current_stage, current_stage_round, global_round,
                        args, replay_log_path, replay_fieldnames, output_dir):
    priority_eval_episodes = (
        args.replay_train_priority_eval_episodes
        if args.replay_train_priority_eval_episodes > 0
        else args.forget_eval_episodes
    )
    replay_items = select_replay_items(
        replay_buffer=replay_buffer,
        max_stages=args.replay_train_max_stages,
        selection_strategy=args.replay_train_selection,
        trainer=trainer,
        env=env,
        mappo=mappo,
        eval_episodes=priority_eval_episodes,
    )
    if not replay_items:
        return hdv_model

    print("\n{} REPLAY TRAINING | replay stages={} {}".format(
        "#" * 18, len(replay_items), "#" * 18
    ))

    for replay_idx, ranked_item in enumerate(replay_items):
        item = ranked_item["item"]
        priority = ranked_item["priority"]
        replay_stage = item["stage"]
        print("  [REPLAY] current={} R{} -> replay={} | rank={} retention={} | adv_eps={} ego_eps={} hdv_steps={}".format(
            current_stage["name"],
            current_stage_round,
            replay_stage["name"],
            priority.get("rank", ""),
            priority.get("retention_ratio", ""),
            args.replay_train_adv_episodes,
            args.replay_train_ego_episodes,
            0 if args.freeze_hdv else args.replay_train_hdv_timesteps,
        ))

        apply_penetration_stage(env, mappo, replay_stage)
        reset_mappo_env(env, mappo, clear_memory=True)
        trainer._round = global_round

        if args.replay_train_adv_episodes > 0:
            trainer._print_phase_header("REPLAY-ADVERSARIAL", args.replay_train_adv_episodes)
            trainer._train_adversarial_phase(
                n_episodes=args.replay_train_adv_episodes,
                eval_interval=max(1, min(args.eval_interval, args.replay_train_adv_episodes)),
                save_interval=args.save_interval,
                log_path=replay_log_path,
                fieldnames=replay_fieldnames,
            )

        if args.replay_train_ego_episodes > 0:
            trainer._print_phase_header("REPLAY-EGO", args.replay_train_ego_episodes)
            trainer._train_ego_phase(
                n_episodes=args.replay_train_ego_episodes,
                eval_interval=max(1, min(args.eval_interval, args.replay_train_ego_episodes)),
                save_interval=args.save_interval,
                log_path=replay_log_path,
                fieldnames=replay_fieldnames,
            )

        if (
            not args.freeze_hdv
            and args.replay_train_hdv_timesteps > 0
        ):
            print("\n{} REPLAY HDV FINE-TUNING {}".format("#" * 16, "#" * 16))
            hdv_reward = AggressiveHDVReward(weights_path=args.hdv_reward)
            hdv_env = CoevolutionHDVEnvWrapper(
                env=env,
                mappo=mappo,
                ego_ppo=trainer.ego_ppo,
                reward_fn=hdv_reward,
                hdv_policy_model=hdv_model,
                deterministic_seed=args.seed + 500000 + 1000 * global_round + replay_idx,
            )
            hdv_model.set_env(hdv_env)
            hdv_model.learn(
                total_timesteps=args.replay_train_hdv_timesteps,
                reset_num_timesteps=False,
            )
            trainer.natural_vehicle_model = hdv_model
            trainer.ego_wrapper.natural_vehicle_model = hdv_model
            reset_mappo_env(env, mappo, clear_memory=True)

            hdv_save_path = os.path.join(
                output_dir,
                "hdv_models",
                "replay_{}_round_{}".format(replay_stage["name"], global_round),
            )
            hdv_model.save(hdv_save_path)
            print("  [REPLAY] HDV model saved to: {}.zip".format(hdv_save_path))

        annotate_latest_replay_train_rows(
            log_path=replay_log_path,
            current_stage=current_stage,
            current_stage_round=current_stage_round,
            replay_stage=replay_stage,
            selection_strategy=args.replay_train_selection,
            priority=priority,
        )

    return hdv_model


def evaluate_forgetting_replay(trainer, env, mappo, replay_buffer,
                               current_stage, current_stage_round,
                               global_round, eval_episodes, log_path,
                               fieldnames):
    print("\n{} FORGETTING EVALUATION | replay stages={} {}".format(
        "#" * 16, len(replay_buffer), "#" * 16
    ))

    rows = []
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for item in replay_buffer:
            replay_stage = item["stage"]
            baseline = item["baseline_metrics"]

            apply_penetration_stage(env, mappo, replay_stage)
            reset_mappo_env(env, mappo, clear_memory=True)
            metrics = trainer.evaluate_ego(n_episodes=eval_episodes, render=False)

            baseline_reward = float(baseline.get("mean_reward", 0.0))
            current_reward = float(metrics.get("mean_reward", 0.0))
            reward_drop = baseline_reward - current_reward
            reward_retention_ratio = (
                current_reward / baseline_reward if abs(baseline_reward) > 1e-8 else 0.0
            )

            baseline_crash = float(baseline.get("crash_rate", 0.0))
            current_crash = float(metrics.get("crash_rate", 0.0))
            baseline_completion = float(baseline.get("road_completion_rate", 0.0))
            current_completion = float(metrics.get("road_completion_rate", 0.0))

            row = {
                "round": global_round,
                "current_stage": current_stage["name"],
                "current_stage_round": current_stage_round,
                "replay_stage": replay_stage["name"],
                "replay_num_cav": replay_stage["num_cav"],
                "replay_num_hdv": replay_stage["num_hdv"],
                "replay_cav_penetration": replay_stage["cav_penetration"],
                "eval_episodes": eval_episodes,
                "baseline_round": item["added_round"],
                "baseline_reward": baseline_reward,
                "current_reward": current_reward,
                "reward_drop": reward_drop,
                "reward_retention_ratio": reward_retention_ratio,
                "baseline_crash_rate": baseline_crash,
                "current_crash_rate": current_crash,
                "crash_rate_increase": current_crash - baseline_crash,
                "baseline_completion": baseline_completion,
                "current_completion": current_completion,
                "completion_drop": baseline_completion - current_completion,
                "mean_length": metrics.get("mean_length", 0.0),
                "reward_std": metrics.get("std_reward", 0.0),
            }
            rows.append(row)
            writer.writerow(row)

            print("  Replay {} | reward {:.2f}->{:.2f} drop={:.2f} retain={:.2f} | crash {:.3f}->{:.3f}".format(
                replay_stage["name"],
                baseline_reward,
                current_reward,
                reward_drop,
                reward_retention_ratio,
                baseline_crash,
                current_crash,
            ))

    if not rows:
        return {
            "min_reward_retention_ratio": 1.0,
            "max_crash_rate_increase": 0.0,
            "max_completion_drop": 0.0,
            "rows": rows,
        }
    return {
        "min_reward_retention_ratio": min(row["reward_retention_ratio"] for row in rows),
        "max_crash_rate_increase": max(row["crash_rate_increase"] for row in rows),
        "max_completion_drop": max(row["completion_drop"] for row in rows),
        "rows": rows,
    }


def forgetting_replay_passes(metrics, retention_threshold=None,
                             crash_increase_threshold=None,
                             completion_drop_threshold=None):
    retention_pass = (
        True if retention_threshold is None
        else metrics.get("min_reward_retention_ratio", 0.0) >= retention_threshold
    )
    crash_pass = (
        True if crash_increase_threshold is None
        else metrics.get("max_crash_rate_increase", 0.0) <= crash_increase_threshold
    )
    completion_pass = (
        True if completion_drop_threshold is None
        else metrics.get("max_completion_drop", 0.0) <= completion_drop_threshold
    )
    return retention_pass and crash_pass and completion_pass


def annotate_latest_log_rows(log_path, stage, stage_round,
                             actual_num_cav=None, actual_num_hdv=None,
                             actual_penetration=None):
    # Existing JointTrainer log writer knows nothing about stage scheduling.
    # Fill empty stage fields after each round while keeping the trainer reusable.
    import pandas as pd

    if not os.path.exists(log_path):
        return
    df = pd.read_csv(log_path)
    required = [
        ("stage", ""),
        ("stage_round", ""),
        ("target_num_cav", ""),
        ("target_num_hdv", ""),
        ("target_cav_penetration", ""),
        ("actual_num_cav", ""),
        ("actual_num_hdv", ""),
        ("actual_cav_penetration", ""),
        ("traffic_density", ""),
    ]
    for col, default in reversed(required):
        if col not in df.columns:
            df.insert(0, col, default)

    empty_stage = df["stage"].isna() | (df["stage"].astype(str) == "")
    df.loc[empty_stage, "stage"] = stage["name"]
    df.loc[empty_stage, "stage_round"] = stage_round
    df.loc[empty_stage, "target_num_cav"] = stage["num_cav"]
    df.loc[empty_stage, "target_num_hdv"] = stage["num_hdv"]
    df.loc[empty_stage, "target_cav_penetration"] = stage["cav_penetration"]
    df.loc[empty_stage, "actual_num_cav"] = actual_num_cav
    df.loc[empty_stage, "actual_num_hdv"] = actual_num_hdv
    df.loc[empty_stage, "actual_cav_penetration"] = actual_penetration
    df.loc[empty_stage, "traffic_density"] = stage["traffic_density"]
    df.to_csv(log_path, index=False, encoding="utf-8")


def annotate_latest_replay_train_rows(log_path, current_stage, current_stage_round,
                                      replay_stage, selection_strategy="recent",
                                      priority=None):
    import pandas as pd

    if not os.path.exists(log_path):
        return
    priority = priority or {}
    df = pd.read_csv(log_path)
    required = [
        ("replay_context", ""),
        ("current_stage", ""),
        ("current_stage_round", ""),
        ("replay_stage", ""),
        ("replay_num_cav", ""),
        ("replay_num_hdv", ""),
        ("replay_cav_penetration", ""),
        ("replay_selection_strategy", ""),
        ("replay_priority_rank", ""),
        ("replay_priority_retention_ratio", ""),
        ("replay_priority_current_reward", ""),
        ("replay_priority_baseline_reward", ""),
        ("replay_priority_crash_rate_increase", ""),
        ("replay_priority_completion_drop", ""),
        ("stage", ""),
        ("stage_round", ""),
        ("target_num_cav", ""),
        ("target_num_hdv", ""),
        ("target_cav_penetration", ""),
        ("actual_num_cav", ""),
        ("actual_num_hdv", ""),
        ("actual_cav_penetration", ""),
        ("traffic_density", ""),
    ]
    for col, default in reversed(required):
        if col not in df.columns:
            df.insert(0, col, default)

    empty_replay = df["replay_context"].isna() | (df["replay_context"].astype(str) == "")
    actual_num_cav = replay_stage["num_cav"]
    actual_num_hdv = replay_stage["num_hdv"]
    actual_penetration = replay_stage["cav_penetration"]

    df.loc[empty_replay, "replay_context"] = "replay_train"
    df.loc[empty_replay, "current_stage"] = current_stage["name"]
    df.loc[empty_replay, "current_stage_round"] = current_stage_round
    df.loc[empty_replay, "replay_stage"] = replay_stage["name"]
    df.loc[empty_replay, "replay_num_cav"] = replay_stage["num_cav"]
    df.loc[empty_replay, "replay_num_hdv"] = replay_stage["num_hdv"]
    df.loc[empty_replay, "replay_cav_penetration"] = replay_stage["cav_penetration"]
    df.loc[empty_replay, "replay_selection_strategy"] = selection_strategy
    df.loc[empty_replay, "replay_priority_rank"] = priority.get("rank", "")
    df.loc[empty_replay, "replay_priority_retention_ratio"] = priority.get("retention_ratio", "")
    df.loc[empty_replay, "replay_priority_current_reward"] = priority.get("current_reward", "")
    df.loc[empty_replay, "replay_priority_baseline_reward"] = priority.get("baseline_reward", "")
    df.loc[empty_replay, "replay_priority_crash_rate_increase"] = priority.get("crash_rate_increase", "")
    df.loc[empty_replay, "replay_priority_completion_drop"] = priority.get("completion_drop", "")
    df.loc[empty_replay, "stage"] = replay_stage["name"]
    df.loc[empty_replay, "stage_round"] = current_stage_round
    df.loc[empty_replay, "target_num_cav"] = replay_stage["num_cav"]
    df.loc[empty_replay, "target_num_hdv"] = replay_stage["num_hdv"]
    df.loc[empty_replay, "target_cav_penetration"] = replay_stage["cav_penetration"]
    df.loc[empty_replay, "actual_num_cav"] = actual_num_cav
    df.loc[empty_replay, "actual_num_hdv"] = actual_num_hdv
    df.loc[empty_replay, "actual_cav_penetration"] = actual_penetration
    df.loc[empty_replay, "traffic_density"] = replay_stage["traffic_density"]
    df.to_csv(log_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
