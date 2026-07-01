#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = "模型/carla_evolution/configs/carla_0915.yaml"
DEFAULT_REWARD = "模型/carla_evolution/results/hdv_irl/theta_aggressive.json"
DEFAULT_OUTPUT_DIR = "模型/carla_evolution/results/hdv_models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/fine-tune aggressive CARLA HDV PPO with learned IRL reward.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--backend", default="mock", choices=["mock", "carla"])
    parser.add_argument("--reward-weights", default=DEFAULT_REWARD)
    parser.add_argument("--hdv-model", default=None, help="Optional neutral HDV PPO model to continue from.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save-name", default="aggressive_hdv_ppo")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--num-cav", type=int, default=3)
    parser.add_argument("--num-hdv", type=int, default=3)
    parser.add_argument("--num-background", type=int, default=0)
    parser.add_argument("--target-hdv-index", type=int, default=0)
    parser.add_argument("--ego-action", type=int, default=1)
    parser.add_argument("--adv-action", type=int, default=1)
    parser.add_argument("--other-hdv-action", type=int, default=1)
    parser.add_argument("--seed", type=int, default=889)
    parser.add_argument("--max-steps", type=int, default=700, help="Max environment steps per HDV episode.")
    parser.add_argument("--log-interval-episodes", type=int, default=100)
    parser.add_argument("--log-file", default=None, help="CSV episode log path. Defaults to output-dir/save-name_train_log.csv.")
    parser.add_argument("--adv-model-dir", default=None, help="Current MAPPO model directory for co-evolution HDV training.")
    parser.add_argument("--adv-step", type=int, default=None, help="Current MAPPO checkpoint step.")
    parser.add_argument("--ego-checkpoint", default=None, help="Current EgoPPO checkpoint for co-evolution HDV training.")
    parser.add_argument("--deterministic-cav", action="store_true", default=False)
    parser.add_argument("--torch-seed", type=int, default=669)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required for HDV PPO training.") from exc
    from carla_evolution.hdv.training.sb3_hdv_env import CarlaHDVSB3Env
    from carla_evolution.agents.ego_ppo import EgoPPOAdapter
    from carla_evolution.agents.mappo import MAPPOAgent

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file) if args.log_file else output_dir / f"{args.save_name}_train_log.csv"

    class HDVEpisodeLogger(BaseCallback):
        def __init__(self, path: Path, interval_episodes: int = 100):
            super().__init__(verbose=0)
            self.path = Path(path)
            self.interval_episodes = max(int(interval_episodes), 1)
            self.episode_count = 0
            self.current_reward = 0.0
            self.current_length = 0
            self.current_speeds = []
            self.current_crash = False
            self.current_completion = 0.0
            self.window = []
            self.start_time = time.time()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "episode",
                        "timesteps",
                        "episode_reward",
                        "episode_length",
                        "avg_speed",
                        "crash",
                        "route_completion",
                        "elapsed_sec",
                    ],
                )
                writer.writeheader()

        def _on_step(self) -> bool:
            rewards = self.locals.get("rewards", [0.0])
            dones = self.locals.get("dones", [False])
            infos = self.locals.get("infos", [{}])
            reward = float(rewards[0]) if len(rewards) else 0.0
            done = bool(dones[0]) if len(dones) else False
            info = infos[0] if len(infos) else {}

            self.current_reward += reward
            self.current_length += 1
            self.current_speeds.append(float(info.get("hdv_speed", info.get("average_speed", 0.0)) or 0.0))
            self.current_crash = bool(self.current_crash or info.get("hdv_crash", False))
            self.current_completion = float(
                info.get("hdv_route_completion", info.get("route_completion", self.current_completion)) or 0.0
            )

            if done:
                self._finish_episode()
            return True

        def _finish_episode(self):
            self.episode_count += 1
            elapsed = time.time() - self.start_time
            row = {
                "episode": self.episode_count,
                "timesteps": int(self.num_timesteps),
                "episode_reward": float(self.current_reward),
                "episode_length": int(self.current_length),
                "avg_speed": float(sum(self.current_speeds) / max(len(self.current_speeds), 1)),
                "crash": int(self.current_crash),
                "route_completion": float(self.current_completion),
                "elapsed_sec": float(elapsed),
            }
            with self.path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writerow(row)
            self.window.append(row)
            if len(self.window) >= self.interval_episodes:
                self._print_window()
                self.window.clear()
            self.current_reward = 0.0
            self.current_length = 0
            self.current_speeds = []
            self.current_crash = False
            self.current_completion = 0.0

        def _print_window(self):
            n = len(self.window)
            avg_reward = sum(item["episode_reward"] for item in self.window) / n
            avg_speed = sum(item["avg_speed"] for item in self.window) / n
            crash_rate = sum(item["crash"] for item in self.window) / n
            completion = sum(item["route_completion"] for item in self.window) / n
            elapsed = time.time() - self.start_time
            print(
                "[HDV] episodes={:d} timesteps={:d} reward={:.2f} "
                "speed={:.2f} crash_rate={:.3f} completion={:.3f} elapsed={:.1f}s log={}".format(
                    self.episode_count,
                    int(self.num_timesteps),
                    avg_reward,
                    avg_speed,
                    crash_rate,
                    completion,
                    elapsed,
                    self.path,
                ),
                flush=True,
            )

    env = CarlaHDVSB3Env(
        config_path=args.config,
        backend=args.backend,
        reward_weights=args.reward_weights,
        num_cav=args.num_cav,
        num_hdv=args.num_hdv,
        num_background=args.num_background,
        target_hdv_index=args.target_hdv_index,
        ego_action=args.ego_action,
        adv_action=args.adv_action,
        other_hdv_action=args.other_hdv_action,
        seed=args.seed,
        max_episode_steps=args.max_steps,
        deterministic_cav=args.deterministic_cav,
    )

    mappo_policy = None
    if args.adv_model_dir:
        if args.adv_step is None:
            raise ValueError("--adv-step is required when --adv-model-dir is set.")
        mappo_policy = MAPPOAgent.from_config(
            {
                "reward_type": "agents_rewards",
                "use_cuda": not args.no_cuda,
                "torch_seed": args.torch_seed,
            },
            state_dim=env.env.n_s,
            action_dim=env.env.n_a,
        )
        mappo_policy.load(args.adv_model_dir, int(args.adv_step), train_mode=False)
        print(f"Loaded MAPPO policy for HDV co-evolution: {args.adv_model_dir} step={args.adv_step}", flush=True)

    ego_policy = None
    if args.ego_checkpoint:
        ego_policy = EgoPPOAdapter.from_config(
            {"use_cuda": not args.no_cuda, "seed": args.torch_seed},
            state_dim=env.env.n_s,
            action_dim=env.env.n_a,
        )
        ego_policy.load(args.ego_checkpoint)
        print(f"Loaded EgoPPO policy for HDV co-evolution: {args.ego_checkpoint}", flush=True)

    env.mappo_policy = mappo_policy
    env.ego_policy = ego_policy

    if args.hdv_model:
        model = PPO.load(args.hdv_model, env=env)
    else:
        model = PPO("MlpPolicy", env, verbose=1, seed=args.seed)
    env.other_hdv_policy = model
    logger = HDVEpisodeLogger(log_path, interval_episodes=args.log_interval_episodes)
    print(f"HDV training log: {log_path}", flush=True)
    model.learn(total_timesteps=args.timesteps, callback=logger)
    save_path = output_dir / args.save_name
    model.save(str(save_path))
    env.close()
    print(f"Saved aggressive CARLA HDV model to: {save_path}.zip")


if __name__ == "__main__":
    main()
