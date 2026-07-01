import numpy as np
import torch as th
import os
import csv
import codecs
from datetime import datetime

from ego.config import PPOConfig
from ego.ppo import EgoPPO


class EgoTrainer:
    """
    主车 PPO 训练器

    负责:
        1. 训练循环: rollout 收集 -> GAE 计算 -> PPO 更新
        2. 评估: 定期在测试环境评估策略性能
        3. 日志: 记录训练指标到 CSV
        4. 模型保存/加载
    """

    def __init__(self, env, eval_env=None, config=None, config_path=None, **overrides):
        if config is None:
            config = PPOConfig(config_path=config_path, **overrides)
        self.config = config

        self.env = env
        self.eval_env = eval_env if eval_env is not None else env

        self.ppo = EgoPPO(config)

        self.output_dir = None
        self.logs = []

    def setup_output_dir(self, base_dir="./ego_results/"):
        now = datetime.utcnow().strftime("%b_%d_%H_%M_%S")
        self.output_dir = os.path.join(base_dir, now)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "models"), exist_ok=True)
        return self.output_dir

    def collect_rollout(self, env, n_steps=None):
        if n_steps is None:
            n_steps = self.config.n_steps

        self.ppo.train_mode()
        state = env.reset()

        if isinstance(state, tuple):
            state = state[0]

        episode_rewards = []
        episode_lengths = []

        for step in range(n_steps):
            action, log_prob, value = self.ppo.select_action(state)

            result = env.step(action)
            if len(result) == 4:
                next_state, reward, done, info = result
            elif len(result) == 5:
                next_state, reward, done, info, _ = result
            else:
                next_state, reward, done, info, *_ = result

            if isinstance(next_state, tuple):
                next_state = next_state[0]

            done_flag = float(done)
            if isinstance(done, (list, tuple, np.ndarray)):
                done_flag = float(done[0] if len(done) > 0 else done)

            self.ppo.step(state, action, log_prob, value, reward, done_flag, next_state)

            state = next_state

            if done:
                ep_reward, ep_length = self.ppo.end_episode()
                episode_rewards.append(ep_reward)
                episode_lengths.append(ep_length)
                state = env.reset()
                if isinstance(state, tuple):
                    state = state[0]

        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "n_episodes": len(episode_rewards),
        }

    def evaluate(self, env=None, n_episodes=None, render=False):
        if env is None:
            env = self.eval_env
        if n_episodes is None:
            n_episodes = self.config.eval_episodes

        self.ppo.eval_mode()
        all_rewards = []
        all_lengths = []
        all_crashes = []

        for ep in range(n_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]

            ep_reward = 0.0
            ep_length = 0
            crashed = False

            done = False
            while not done:
                action, _, _ = self.ppo.select_action(state, deterministic=True)

                result = env.step(action)
                if len(result) == 4:
                    next_state, reward, done, info = result
                elif len(result) == 5:
                    next_state, reward, done, info, _ = result
                else:
                    next_state, reward, done, info, *_ = result

                if isinstance(next_state, tuple):
                    next_state = next_state[0]

                ep_reward += reward
                ep_length += 1

                if isinstance(info, dict):
                    crashed = crashed or info.get('crashed', False)

                state = next_state

                if render:
                    env.render()

            all_rewards.append(ep_reward)
            all_lengths.append(ep_length)
            all_crashes.append(float(crashed))

        return {
            "mean_reward": np.mean(all_rewards),
            "std_reward": np.std(all_rewards),
            "mean_length": np.mean(all_lengths),
            "crash_rate": np.mean(all_crashes),
            "rewards": all_rewards,
            "lengths": all_lengths,
        }

    def train(self, max_episodes=None, output_dir=None):
        if max_episodes is None:
            max_episodes = self.config.max_episodes

        if output_dir is None:
            output_dir = self.setup_output_dir()
        self.output_dir = output_dir

        model_dir = os.path.join(output_dir, "models")
        log_path = os.path.join(output_dir, "train_log.csv")

        fieldnames = [
            "episode", "total_steps", "ep_reward_mean", "ep_length_mean",
            "policy_loss", "value_loss", "entropy",
            "eval_reward_mean", "eval_reward_std", "eval_crash_rate"
        ]

        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

        while self.ppo.n_episodes < max_episodes:
            rollout_info = self.collect_rollout(self.env)

            update_info = self.ppo.update()

            log_entry = {
                "episode": self.ppo.n_episodes,
                "total_steps": self.ppo.n_steps,
                "ep_reward_mean": np.mean(rollout_info["episode_rewards"]) if rollout_info["episode_rewards"] else 0.0,
                "ep_length_mean": np.mean(rollout_info["episode_lengths"]) if rollout_info["episode_lengths"] else 0.0,
                "policy_loss": update_info.get("policy_loss", 0.0),
                "value_loss": update_info.get("value_loss", 0.0),
                "entropy": update_info.get("entropy", 0.0),
                "eval_reward_mean": 0.0,
                "eval_reward_std": 0.0,
                "eval_crash_rate": 0.0,
            }

            if self.ppo.n_episodes % self.config.eval_interval == 0:
                eval_info = self.evaluate()
                log_entry["eval_reward_mean"] = eval_info["mean_reward"]
                log_entry["eval_reward_std"] = eval_info["std_reward"]
                log_entry["eval_crash_rate"] = eval_info["crash_rate"]

                print(f"[Ep {self.ppo.n_episodes:5d}] "
                      f"Train R: {log_entry['ep_reward_mean']:7.2f} | "
                      f"Eval R: {eval_info['mean_reward']:7.2f} ± {eval_info['std_reward']:5.2f} | "
                      f"Crash: {eval_info['crash_rate']:.3f} | "
                      f"P_Loss: {update_info.get('policy_loss', 0):.4f} | "
                      f"V_Loss: {update_info.get('value_loss', 0):.4f}")

            if self.ppo.n_episodes % self.config.save_interval == 0:
                save_path = os.path.join(model_dir, f"checkpoint-{self.ppo.n_episodes}.pt")
                self.ppo.save(save_path)

            with open(log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(log_entry)

        final_path = os.path.join(model_dir, "final_model.pt")
        self.ppo.save(final_path)
        print(f"Training finished. Model saved to {final_path}")

        return self.ppo