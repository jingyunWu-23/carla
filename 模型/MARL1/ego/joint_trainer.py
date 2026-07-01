import numpy as np
import torch as th
import os
import csv
import codecs
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from copy import deepcopy

from ego.config import PPOConfig
from ego.ppo import EgoPPO
from ego.ego_env_wrapper import EgoEnvWrapper
from ego.reward.cat_reward import CatReward, CatSafetyReward


class JointTrainer:
    """
    Phased Alternating Joint Trainer

    训练流程:
        ┌──────────────────────────────────────────────┐
        │  Phase 1: 对抗车训练 (MAPPO)                    │
        │  - MAPPO 的 3 辆对抗车正常训练                    │
        │  - 主车使用 EgoPPO 当前策略 (仅推理, 不更新)       │
        │  - 自然车使用冻结 PPO 模型                        │
        │  - 持续 N_adversarial 个 episode                │
        ├──────────────────────────────────────────────┤
        │  Phase 2: 主车训练 (EgoPPO)                     │
        │  - 冻结 MAPPO 对抗车 (仅推理, 不更新)              │
        │  - 主车使用 EgoPPO 正常训练 (收集 rollout + 更新)  │
        │  - 自然车使用冻结 PPO 模型                        │
        │  - 持续 N_ego 个 episode                        │
        └──────────────────────────────────────────────┘
        → 循环直到达到总训练轮数

    Usage:
        trainer = JointTrainer(env, mappo, ego_config)
        trainer.train(
            n_adversarial_episodes=100,
            n_ego_episodes=50,
            n_rounds=10,
        )
    """

    def __init__(
        self,
        env,
        mappo: Any,
        ego_config: Optional[PPOConfig] = None,
        ego_config_path: Optional[str] = None,
        natural_vehicle_model: Any = None,
        ego_reward_fn: Optional[CatReward] = None,
        **ego_overrides,
    ):
        self.env = env
        self.mappo = mappo
        self.natural_vehicle_model = natural_vehicle_model

        if ego_config is None:
            ego_config = PPOConfig(config_path=ego_config_path, **ego_overrides)
        self.ego_config = ego_config

        self.ego_reward_fn = ego_reward_fn or CatSafetyReward()

        self.ego_wrapper = EgoEnvWrapper(
            env=env,
            mappo=mappo,
            reward_fn=self.ego_reward_fn,
            natural_vehicle_model=natural_vehicle_model,
        )

        self.ego_ppo = EgoPPO(ego_config)

        self.output_dir = None
        self.logs = []

        self._phase = None
        self._round = 0
        self._total_adversarial_episodes = 0
        self._total_ego_episodes = 0

        self._max_steps = int(
            env.config.get("duration", 20) * env.config.get("policy_frequency", 5) * 0.4
        )

    def setup_output_dir(self, base_dir="./joint_results/"):
        now = datetime.utcnow().strftime("%b_%d_%H_%M_%S")
        self.output_dir = os.path.join(base_dir, now)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "mappo_models"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "ego_models"), exist_ok=True)
        return self.output_dir

    def train(
        self,
        n_adversarial_episodes: int = 100,
        n_ego_episodes: int = 50,
        n_rounds: int = 10,
        output_dir: Optional[str] = None,
        eval_interval: int = 10,
        save_interval: int = 50,
    ):
        """
        主训练循环

        Args:
            n_adversarial_episodes: 每轮对抗车训练的 episode 数
            n_ego_episodes: 每轮主车训练的 episode 数
            n_rounds: 总交替轮数
            output_dir: 输出目录
            eval_interval: 评估间隔 (episodes)
            save_interval: 模型保存间隔 (episodes)
        """
        if output_dir is None:
            output_dir = self.setup_output_dir()
        self.output_dir = output_dir

        log_path = os.path.join(output_dir, "joint_train_log.csv")
        fieldnames = [
            "round", "phase", "episode", "total_adversarial_episodes",
            "total_ego_episodes", "adversarial_reward", "ego_reward",
            "ego_policy_loss", "ego_value_loss",
            "ego_crash_rate", "ego_road_completion_rate",
            "ego_avg_speed", "adversarial_crash_rate",
        ]

        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

        print("=" * 70)
        print("  JOINT ADVERSARIAL TRAINING - Phased Alternating")
        print("=" * 70)
        print(f"  Rounds:              {n_rounds}")
        print(f"  Adversarial eps/rd:  {n_adversarial_episodes}")
        print(f"  Ego eps/rd:          {n_ego_episodes}")
        print(f"  Total episodes:      {n_rounds * (n_adversarial_episodes + n_ego_episodes)}")
        print(f"  Output dir:          {output_dir}")
        print("=" * 70)

        for round_idx in range(n_rounds):
            self._round = round_idx + 1

            self._print_phase_header("ADVERSARIAL", n_adversarial_episodes)
            adv_info = self._train_adversarial_phase(
                n_episodes=n_adversarial_episodes,
                eval_interval=eval_interval,
                save_interval=save_interval,
                log_path=log_path,
                fieldnames=fieldnames,
            )

            self._print_phase_header("EGO", n_ego_episodes)
            ego_info = self._train_ego_phase(
                n_episodes=n_ego_episodes,
                eval_interval=eval_interval,
                save_interval=save_interval,
                log_path=log_path,
                fieldnames=fieldnames,
            )

            self._print_round_summary(adv_info, ego_info)

        self._save_final_models()
        print(f"\n{'=' * 70}")
        print(f"  TRAINING COMPLETE")
        print(f"  Results saved to: {output_dir}")
        print(f"{'=' * 70}")

    def train_adaptive(
        self,
        n_rounds: int = 100,
        adv_min_eps: int = 150,
        adv_max_eps: int = 300,
        ego_min_eps: int = 300,
        ego_max_eps: int = 500,
        adv_crash_threshold: float = 0.15,
        adv_window: int = 50,
        ego_window: int = 100,
        ego_improvement_threshold: float = 0.01,
        ppo_update_interval: int = 256,
        output_dir: Optional[str] = None,
        eval_interval: int = 10,
        save_interval: int = 50,
        progressive: bool = True,
    ):
        """
        自适应交替训练主循环

        切换逻辑:
            - 对抗阶段: 训练 MAPPO, 追踪主车安全指标
              当主车安全指标不再显著提升 → 切换到主车阶段
            - 主车阶段: 训练 EgoPPO (频繁更新), 追踪对抗车碰撞率
              当对抗车碰撞率低于阈值 → 切换到对抗阶段

        Args:
            n_rounds: 总交替轮数 (70)
            adv_min_eps: 对抗阶段最少 episode 数
            adv_max_eps: 对抗阶段最多 episode 数
            ego_min_eps: 主车阶段最少 episode 数
            ego_max_eps: 主车阶段最多 episode 数
            adv_crash_threshold: 对抗车碰撞率阈值 (低于此值触发切换)
            adv_window: 对抗车碰撞率滑动窗口大小
            ego_window: 主车安全指标滑动窗口大小
            ego_improvement_threshold: 主车安全指标提升阈值
            ppo_update_interval: PPO 网络更新间隔 (步数)
            output_dir: 输出目录
            eval_interval: 评估间隔
            save_interval: 模型保存间隔
        """
        if output_dir is None:
            output_dir = self.setup_output_dir()
        self.output_dir = output_dir

        log_path = os.path.join(output_dir, "joint_train_log.csv")
        fieldnames = [
            "round", "phase", "episode", "total_adversarial_episodes",
            "total_ego_episodes", "adversarial_reward", "ego_reward",
            "ego_policy_loss", "ego_value_loss",
            "ego_crash_rate", "ego_road_completion_rate",
            "ego_avg_speed", "adversarial_crash_rate",
        ]

        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

        print("=" * 70)
        print("  ADAPTIVE JOINT ADVERSARIAL TRAINING")
        print("=" * 70)
        print(f"  Rounds:                {n_rounds}")
        if progressive:
            print(f"  Episode allocation:    PROGRESSIVE")
            print(f"    Stage 1 (R1-R10):    ADV 150-300, EGO 300-500")
            print(f"    Stage 2 (R11-R40):   ADV 100-200, EGO 500-800")
            print(f"    Stage 3 (R41-R70):   ADV 80-150,  EGO 800-1000")
        else:
            print(f"  Adversarial phase:     {adv_min_eps}-{adv_max_eps} eps")
            print(f"  Ego phase:             {ego_min_eps}-{ego_max_eps} eps")
        print(f"  Adv crash threshold:   {adv_crash_threshold:.0%}")
        print(f"  Adv crash window:      {adv_window} eps")
        print(f"  Ego safety window:     {ego_window} eps")
        print(f"  Ego improvement th:    {ego_improvement_threshold}")
        print(f"  PPO update interval:   {ppo_update_interval} steps")
        print(f"  Reward function:       {type(self.ego_reward_fn).__name__}")
        if isinstance(self.ego_reward_fn, CatSafetyReward):
            print(f"    risk_alpha:          {self.ego_reward_fn.alpha}")
            print(f"    safe_distance:       {self.ego_reward_fn.safe_distance} m")
            print(f"    success_reward:      {self.ego_reward_fn.terminal.success_reward}")
        print(f"  Output dir:            {output_dir}")
        print("=" * 70)

        phase = "adversarial"

        for round_idx in range(n_rounds):
            self._round = round_idx + 1

            if progressive:
                cur_adv_min, cur_adv_max, cur_ego_min, cur_ego_max = \
                    self._get_progressive_eps(self._round, n_rounds)
            else:
                cur_adv_min, cur_adv_max = adv_min_eps, adv_max_eps
                cur_ego_min, cur_ego_max = ego_min_eps, ego_max_eps

            if phase == "adversarial":
                self._print_phase_header("ADVERSARIAL",
                                         f"{cur_adv_min}-{cur_adv_max}")
                adv_info = self._train_adversarial_phase_adaptive(
                    min_eps=cur_adv_min,
                    max_eps=cur_adv_max,
                    ego_window=ego_window,
                    ego_improvement_threshold=ego_improvement_threshold,
                    eval_interval=eval_interval,
                    save_interval=save_interval,
                    log_path=log_path,
                    fieldnames=fieldnames,
                )
                phase = "ego"
                self._print_round_summary(adv_info, {"episodes": 0})
            else:
                self._print_phase_header("EGO",
                                         f"{cur_ego_min}-{cur_ego_max}")
                ego_info = self._train_ego_phase_adaptive(
                    min_eps=cur_ego_min,
                    max_eps=cur_ego_max,
                    adv_window=adv_window,
                    adv_crash_threshold=adv_crash_threshold,
                    ppo_update_interval=ppo_update_interval,
                    eval_interval=eval_interval,
                    save_interval=save_interval,
                    log_path=log_path,
                    fieldnames=fieldnames,
                )
                phase = "adversarial"
                self._print_round_summary({"episodes": 0}, ego_info)

        self._save_final_models()
        print(f"\n{'=' * 70}")
        print(f"  TRAINING COMPLETE")
        print(f"  Results saved to: {output_dir}")
        print(f"{'=' * 70}")

    def _train_adversarial_phase_adaptive(
        self,
        min_eps: int,
        max_eps: int,
        ego_window: int,
        ego_improvement_threshold: float,
        eval_interval: int,
        save_interval: int,
        log_path: str,
        fieldnames: list,
    ) -> Dict[str, Any]:
        """
        对抗车训练阶段 (自适应退出)

        MAPPO 正常训练, EgoPPO 仅推理。
        追踪主车安全指标, 当不再显著提升时提前退出。
        """
        self._phase = "adversarial"
        self.ego_ppo.eval_mode()

        start_ep = self.mappo.n_episodes
        episode_rewards = []
        crash_flags = []

        ego_safety_history = []

        actual_eps = 0

        while self.mappo.n_episodes < start_ep + max_eps:
            ego_crashed = self._adversarial_interact()
            self._total_adversarial_episodes += 1
            actual_eps += 1

            if self.mappo.n_episodes >= self.mappo.episodes_before_train:
                self.mappo.train()

            if self.mappo.episode_done:
                ep_reward = (self.mappo.episode_rewards[-2]
                             if len(self.mappo.episode_rewards) > 1 else 0)
                episode_rewards.append(ep_reward)
                crash_flags.append(ego_crashed)

                ep_length = self.mappo.epoch_steps[-2] if len(self.mappo.epoch_steps) > 1 else 0
                survival_ratio = ep_length / max(self._max_steps, 1)
                ego_safety = survival_ratio * (1.0 - float(ego_crashed))
                ego_safety_history.append(ego_safety)

                if (self.mappo.n_episodes + 1) % eval_interval == 0:
                    recent = episode_rewards[-eval_interval:]
                    recent_crashes = crash_flags[-eval_interval:]
                    avg_r = np.mean(recent) if recent else 0
                    crash_rate = np.mean(recent_crashes) if recent_crashes else 0

                    print(f"  [ADV] Ep {self.mappo.n_episodes + 1:5d} | "
                          f"Adv Reward: {avg_r:8.2f} | "
                          f"Ego Crash: {crash_rate:.1%}")

                    self._log_entry(
                        log_path, fieldnames,
                        phase="adversarial",
                        episode=self.mappo.n_episodes + 1,
                        adversarial_reward=avg_r,
                        adversarial_crash_rate=crash_rate,
                    )

                if (self.mappo.n_episodes + 1) % save_interval == 0:
                    mappo_dir = os.path.join(self.output_dir, "mappo_models")
                    self.mappo.save(mappo_dir, self.mappo.n_episodes + 1)

                if actual_eps >= min_eps and len(ego_safety_history) >= ego_window:
                    recent_safety = ego_safety_history[-ego_window:]
                    older_safety = ego_safety_history[-2 * ego_window:-ego_window] if len(ego_safety_history) >= 2 * ego_window else [0] * ego_window
                    recent_mean = np.mean(recent_safety)
                    older_mean = np.mean(older_safety) if older_safety else 0
                    improvement = recent_mean - older_mean

                    if improvement < ego_improvement_threshold:
                        print(f"\n  [ADV] Early exit at ep {actual_eps}: "
                              f"ego safety plateaued (improvement={improvement:.4f} < {ego_improvement_threshold})")
                        break

            if actual_eps >= max_eps:
                break

        return {
            "episodes": actual_eps,
            "avg_reward": np.mean(episode_rewards) if episode_rewards else 0,
            "crash_rate": np.mean(crash_flags) if crash_flags else 0,
        }

    def _train_ego_phase_adaptive(
        self,
        min_eps: int,
        max_eps: int,
        adv_window: int,
        adv_crash_threshold: float,
        ppo_update_interval: int,
        eval_interval: int,
        save_interval: int,
        log_path: str,
        fieldnames: list,
    ) -> Dict[str, Any]:
        """
        主车训练阶段 (自适应退出 + 频繁 PPO 更新)

        EgoPPO 正常训练 (每 ppo_update_interval 步更新一次),
        MAPPO 仅推理。
        追踪对抗车碰撞率, 当低于阈值时提前退出。
        """
        self._phase = "ego"
        self.ego_ppo.train_mode()

        start_ep = self.ego_ppo.n_episodes
        episode_rewards = []
        crash_flags = []
        completion_flags = []
        speed_records = []

        adv_crash_window = []

        self.ego_wrapper.mappo = self.mappo
        self.ego_wrapper.natural_vehicle_model = self.natural_vehicle_model

        obs = self.ego_wrapper.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        current_ep_reward = 0.0
        current_ep_length = 0
        current_ep_crashed = False
        steps_since_update = 0
        actual_eps = 0
        update_info = {}

        while actual_eps < max_eps:
            action, log_prob, value = self.ego_ppo.select_action(obs)

            next_obs, reward, done, info = self.ego_wrapper.step(action)

            if isinstance(next_obs, tuple):
                next_obs = next_obs[0]

            ego_vehicle = self._get_ego_vehicle()
            if ego_vehicle is not None and ego_vehicle.crashed:
                current_ep_crashed = True

            done_flag = float(done)

            self.ego_ppo.step(obs, action, log_prob, value, reward, done_flag, next_obs)

            current_ep_reward += reward
            current_ep_length += 1
            steps_since_update += 1
            obs = next_obs

            if steps_since_update >= ppo_update_interval:
                update_info = self.ego_ppo.update()
                steps_since_update = 0
                print(f"  [EGO] Update | "
                      f"P_Loss: {update_info.get('policy_loss', 0):.4f} | "
                      f"V_Loss: {update_info.get('value_loss', 0):.4f}")

            if done:
                episode_rewards.append(current_ep_reward)
                crash_flags.append(float(current_ep_crashed))
                completion_flags.append(current_ep_length / max(self._max_steps, 1))
                speed_records.append(info.get("ego_avg_speed", 0.0))

                adv_crash_window.append(float(current_ep_crashed))
                if len(adv_crash_window) > adv_window:
                    adv_crash_window.pop(0)

                current_ep_reward = 0.0
                current_ep_length = 0
                current_ep_crashed = False

                self.ego_ppo.end_episode()
                self._total_ego_episodes += 1
                actual_eps += 1

                current_ep = start_ep + len(episode_rewards)
                if current_ep % eval_interval == 0:
                    recent = episode_rewards[-eval_interval:]
                    recent_crashes = crash_flags[-eval_interval:]
                    recent_completions = completion_flags[-eval_interval:]
                    recent_speeds = speed_records[-eval_interval:]

                    avg_r = np.mean(recent) if recent else 0
                    crash_rate = np.mean(recent_crashes) if recent_crashes else 0
                    completion_rate = np.mean(recent_completions) if recent_completions else 0
                    avg_speed = np.mean(recent_speeds) if recent_speeds else 0

                    print(f"  [EGO] Ep {current_ep:5d} | "
                          f"Reward: {avg_r:8.2f} | "
                          f"Crash: {crash_rate:.1%} | "
                          f"Road Compl: {completion_rate:.1%} | "
                          f"Avg Speed: {avg_speed:.1f} m/s")

                    self._log_entry(
                        log_path, fieldnames,
                        phase="ego",
                        episode=current_ep,
                        ego_reward=avg_r,
                        ego_policy_loss=update_info.get("policy_loss", 0),
                        ego_value_loss=update_info.get("value_loss", 0),
                        ego_crash_rate=crash_rate,
                        ego_road_completion_rate=completion_rate,
                        ego_avg_speed=avg_speed,
                    )

                if self.ego_ppo.n_episodes % save_interval == 0:
                    ego_dir = os.path.join(self.output_dir, "ego_models")
                    save_path = os.path.join(ego_dir, f"checkpoint-{self.ego_ppo.n_episodes}.pt")
                    self.ego_ppo.save(save_path)

                if actual_eps >= min_eps and len(adv_crash_window) >= adv_window:
                    avg_crash = np.mean(adv_crash_window)
                    if avg_crash < adv_crash_threshold:
                        print(f"\n  [EGO] Early exit at ep {actual_eps}: "
                              f"adv crash rate too low ({avg_crash:.1%} < {adv_crash_threshold:.0%})")
                        break

                obs = self.ego_wrapper.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]

        if steps_since_update > 0 and self.ego_ppo.buffer.size >= self.ego_config.batch_size:
            update_info = self.ego_ppo.update()

        return {
            "episodes": actual_eps,
            "avg_reward": np.mean(episode_rewards) if episode_rewards else 0,
            "crash_rate": np.mean(crash_flags) if crash_flags else 0,
            "road_completion_rate": np.mean(completion_flags) if completion_flags else 0,
            "avg_speed": np.mean(speed_records) if speed_records else 0,
        }

    def _print_phase_header(self, phase_name: str, n_episodes: str):
        print(f"\n{'#' * 70}")
        if phase_name == "ADVERSARIAL":
            print(f"  R{self._round} | PHASE: {phase_name} TRAINING")
            print(f"  {'[TRAINING]':>12} MAPPO  (3 adversarial vehicles)")
            print(f"  {'[FROZEN]':>12} EgoPPO (ego vehicle, inference only)")
            print(f"  {'[FROZEN]':>12} PPO    (natural vehicle)")
        else:
            print(f"  R{self._round} | PHASE: {phase_name} TRAINING")
            print(f"  {'[FROZEN]':>12} MAPPO  (3 adversarial vehicles)")
            print(f"  {'[TRAINING]':>12} EgoPPO (ego vehicle)")
        print(f"  Episodes:              {n_episodes}")
        print(f"{'#' * 70}")

    @staticmethod
    def _get_progressive_eps(round_idx: int, n_rounds: int):
        """
        渐进式 episode 配比

        根据当前轮次动态调整对抗车和主车的训练 episode 数:

        | 轮次      | 对抗车     | 主车       | 逻辑                         |
        |-----------|-----------|-----------|------------------------------|
        | 1-10      | 150-300   | 300-500   | 初期双方充分探索              |
        | 11-40     | 100-200   | 500-800   | 对抗车快速切换, 主车深度收敛   |
        | 41-70     | 80-150    | 800-1000  | 对抗车保持多样性, 主车精细调优  |

        Returns:
            (adv_min, adv_max, ego_min, ego_max)
        """
        ratio = round_idx / max(n_rounds, 1)

        if ratio <= 10 / 70:
            return 150, 300, 300, 500
        elif ratio <= 40 / 70:
            return 100, 200, 500, 800
        else:
            return 80, 150, 800, 1000

    def _print_round_summary(self, adv_info: Dict, ego_info: Dict):
        print(f"\n{'=' * 70}")
        print(f"  ROUND {self._round} SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Adversarial Phase:")
        print(f"    Episodes:       {adv_info['episodes']}")
        print(f"    Avg Reward:     {adv_info.get('avg_reward', 0):.2f}")
        print(f"    Crash Rate:     {adv_info.get('crash_rate', 0):.3f}")
        print(f"  Ego Phase:")
        print(f"    Episodes:       {ego_info['episodes']}")
        print(f"    Avg Reward:     {ego_info.get('avg_reward', 0):.2f}")
        print(f"    Crash Rate:     {ego_info.get('crash_rate', 0):.3f}")
        print(f"    Road Compl. %:  {ego_info.get('road_completion_rate', 0):.1%}")
        print(f"    Avg Speed:      {ego_info.get('avg_speed', 0):.1f} m/s")
        print(f"{'=' * 70}")

    def _train_adversarial_phase(
        self,
        n_episodes: int,
        eval_interval: int,
        save_interval: int,
        log_path: str,
        fieldnames: list,
    ) -> Dict[str, Any]:
        """
        对抗车训练阶段

        MAPPO 正常训练, EgoPPO 仅推理 (eval mode)。
        """
        self._phase = "adversarial"
        self.ego_ppo.eval_mode()

        start_ep = self.mappo.n_episodes
        target_ep = start_ep + n_episodes
        episode_rewards = []
        crash_flags = []

        while self.mappo.n_episodes < target_ep:
            ego_crashed = self._adversarial_interact()
            self._total_adversarial_episodes += 1

            if self.mappo.n_episodes >= self.mappo.episodes_before_train:
                self.mappo.train()

            if self.mappo.episode_done:
                ep_reward = (self.mappo.episode_rewards[-2]
                             if len(self.mappo.episode_rewards) > 1 else 0)
                episode_rewards.append(ep_reward)
                crash_flags.append(ego_crashed)

                if (self.mappo.n_episodes + 1) % eval_interval == 0:
                    recent = episode_rewards[-eval_interval:]
                    recent_crashes = crash_flags[-eval_interval:]
                    avg_r = np.mean(recent) if recent else 0
                    crash_rate = np.mean(recent_crashes) if recent_crashes else 0

                    print(f"  [ADV] Ep {self.mappo.n_episodes + 1:5d} | "
                          f"Adv Reward: {avg_r:8.2f} | "
                          f"Ego Crash: {crash_rate:.1%}")

                    self._log_entry(
                        log_path, fieldnames,
                        phase="adversarial",
                        episode=self.mappo.n_episodes + 1,
                        adversarial_reward=avg_r,
                        adversarial_crash_rate=crash_rate,
                    )

                if (self.mappo.n_episodes + 1) % save_interval == 0:
                    mappo_dir = os.path.join(self.output_dir, "mappo_models")
                    self.mappo.save(mappo_dir, self.mappo.n_episodes + 1)

        return {
            "episodes": n_episodes,
            "avg_reward": np.mean(episode_rewards) if episode_rewards else 0,
            "crash_rate": np.mean(crash_flags) if crash_flags else 0,
        }

    def _adversarial_interact(self):
        """
        对抗车交互步骤

        复用 MAPPO 的交互逻辑, 但将冻结 PPO 模型替换为 EgoPPO。
        """
        if (self.mappo.max_steps is not None) and \
           (self.mappo.n_steps >= self.mappo.max_steps):
            self.mappo.env_state, _, self.mappo.obs2, self.mappo.obs3 = \
                self.mappo.env.reset()
            self.mappo.n_steps = 0

        states = []
        actions = []
        rewards = []
        done = True
        average_speed = 0
        ego_crashed = False

        n_agents = len(self.mappo.env.controlled_vehicles)
        self.mappo.n_agents = n_agents

        for i in range(self.mappo.roll_out_n_steps):
            states.append(self.mappo.env_state)
            try:
                action = self.mappo.exploration_action(self.mappo.env_state, n_agents)
                if action is None:
                    action = [1] * n_agents
            except Exception:
                action = [1] * n_agents

            ego_action = self._get_ego_action(self.mappo.obs2)

            natural_actions = self._get_natural_actions()

            ego_slot = n_agents
            natural_slot = n_agents + 1
            while len(action) < natural_slot:
                action.append(1)
            action[ego_slot] = ego_action
            action = action[:natural_slot] + natural_actions

            next_state, global_reward, done, info, obs2, obs3 = \
                self.mappo.env.step(tuple(action))

            self.mappo.obs2 = obs2
            self.mappo.obs3 = obs3

            ego_vehicle = self._get_ego_vehicle()
            if ego_vehicle is not None and ego_vehicle.crashed:
                ego_crashed = True

            actions.append([
                self.mappo.env.index_to_one_hot(a, self.mappo.action_dim)
                if hasattr(self.mappo.env, 'index_to_one_hot')
                else np.eye(self.mappo.action_dim)[a]
                for a in action[:n_agents]
            ])

            self.mappo.episode_rewards[-1] += global_reward
            self.mappo.epoch_steps[-1] += 1

            if self.mappo.reward_type == "regionalR":
                reward = info["regional_rewards"]
            elif self.mappo.reward_type == "global_R":
                reward = [global_reward] * n_agents
            rewards.append(reward)

            average_speed += info.get("average_speed", 0)
            final_state = next_state
            self.mappo.env_state = next_state
            self.mappo.n_steps += 1

            if done:
                self.mappo.env_state, _, self.mappo.obs2, self.mappo.obs3 = \
                    self.mappo.env.reset()
                break

        if done:
            final_value = [0.0] * n_agents
            self.mappo.n_episodes += 1
            self.mappo.episode_done = True
            self.mappo.episode_rewards.append(0)
            self.mappo.average_speed[-1] = average_speed / max(self.mappo.epoch_steps[-1], 1)
            self.mappo.average_speed.append(0)
            self.mappo.epoch_steps.append(0)
        else:
            self.mappo.episode_done = False
            final_action = self.mappo.action(final_state)
            final_value = self.mappo.value(final_state, final_action)

        if self.mappo.reward_scale > 0:
            rewards = np.array(rewards) / self.mappo.reward_scale

        for agent_id in range(n_agents):
            rewards[:, agent_id] = self.mappo._discount_reward(
                rewards[:, agent_id], final_value[agent_id]
            )

        rewards = rewards.tolist()
        self.mappo.memory.push(states, actions, rewards)

        return ego_crashed

    def _train_ego_phase(
        self,
        n_episodes: int,
        eval_interval: int,
        save_interval: int,
        log_path: str,
        fieldnames: list,
    ) -> Dict[str, Any]:
        """
        主车训练阶段

        EgoPPO 正常训练, MAPPO 仅推理。
        """
        self._phase = "ego"
        self.ego_ppo.train_mode()

        start_ep = self.ego_ppo.n_episodes
        target_ep = start_ep + n_episodes
        episode_rewards = []
        crash_flags = []
        completion_flags = []
        speed_records = []

        while self.ego_ppo.n_episodes < target_ep:
            rollout_info = self._ego_collect_rollout()

            update_info = self.ego_ppo.update()

            self._total_ego_episodes += rollout_info.get("n_episodes", 0)

            print(f"  [EGO] Update | "
                  f"P_Loss: {update_info.get('policy_loss', 0):.4f} | "
                  f"V_Loss: {update_info.get('value_loss', 0):.4f}")

            n_new_eps = rollout_info.get("n_episodes", 0)
            for i in range(n_new_eps):
                episode_rewards.append(rollout_info["episode_rewards"][i])
                crash_flags.append(rollout_info["episode_crashes"][i])
                completion_flags.append(rollout_info["episode_completions"][i])
                speed_records.append(rollout_info.get("episode_speeds", [0.0])[i]
                                     if i < len(rollout_info.get("episode_speeds", [])) else 0.0)

                current_ep = start_ep + len(episode_rewards)
                if current_ep % eval_interval == 0:
                    recent = episode_rewards[-eval_interval:]
                    recent_crashes = crash_flags[-eval_interval:]
                    recent_completions = completion_flags[-eval_interval:]
                    recent_speeds = speed_records[-eval_interval:]

                    avg_r = np.mean(recent) if recent else 0
                    crash_rate = np.mean(recent_crashes) if recent_crashes else 0
                    completion_rate = np.mean(recent_completions) if recent_completions else 0
                    avg_speed = np.mean(recent_speeds) if recent_speeds else 0

                    print(f"  [EGO] Ep {current_ep:5d} | "
                          f"Reward: {avg_r:8.2f} | "
                          f"Crash: {crash_rate:.1%} | "
                          f"Road Compl: {completion_rate:.1%} | "
                          f"Avg Speed: {avg_speed:.1f} m/s")

                    self._log_entry(
                        log_path, fieldnames,
                        phase="ego",
                        episode=current_ep,
                        ego_reward=avg_r,
                        ego_policy_loss=update_info.get("policy_loss", 0),
                        ego_value_loss=update_info.get("value_loss", 0),
                        ego_crash_rate=crash_rate,
                        ego_road_completion_rate=completion_rate,
                        ego_avg_speed=avg_speed,
                    )

            if self.ego_ppo.n_episodes % save_interval == 0:
                ego_dir = os.path.join(self.output_dir, "ego_models")
                save_path = os.path.join(ego_dir, f"checkpoint-{self.ego_ppo.n_episodes}.pt")
                self.ego_ppo.save(save_path)

        return {
            "episodes": n_episodes,
            "avg_reward": np.mean(episode_rewards) if episode_rewards else 0,
            "crash_rate": np.mean(crash_flags) if crash_flags else 0,
            "road_completion_rate": np.mean(completion_flags) if completion_flags else 0,
        }

    def _ego_collect_rollout(self) -> Dict[str, Any]:
        """
        EgoPPO rollout 收集

        在对抗环境中收集主车的经验:
        - 对抗车使用 MAPPO 当前策略 (exploration_action)
        - 主车使用 EgoPPO 策略 (select_action)
        - 自然车使用冻结 PPO 模型
        """
        n_steps = self.ego_config.n_steps

        self.ego_wrapper.mappo = self.mappo
        self.ego_wrapper.natural_vehicle_model = self.natural_vehicle_model

        obs = self.ego_wrapper.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        episode_rewards = []
        episode_lengths = []
        episode_crashes = []
        episode_completions = []
        episode_speeds = []

        current_ep_reward = 0.0
        current_ep_length = 0
        current_ep_crashed = False

        for step in range(n_steps):
            action, log_prob, value = self.ego_ppo.select_action(obs)

            next_obs, reward, done, info = self.ego_wrapper.step(action)

            if isinstance(next_obs, tuple):
                next_obs = next_obs[0]

            ego_vehicle = self._get_ego_vehicle()
            if ego_vehicle is not None and ego_vehicle.crashed:
                current_ep_crashed = True

            done_flag = float(done)

            self.ego_ppo.step(obs, action, log_prob, value, reward, done_flag, next_obs)

            current_ep_reward += reward
            current_ep_length += 1
            obs = next_obs

            if done:
                episode_rewards.append(current_ep_reward)
                episode_lengths.append(current_ep_length)
                episode_crashes.append(float(current_ep_crashed))

                road_completion_ratio = current_ep_length / max(self._max_steps, 1)
                episode_completions.append(road_completion_ratio)
                episode_speeds.append(info.get("ego_avg_speed", 0.0))

                current_ep_reward = 0.0
                current_ep_length = 0
                current_ep_crashed = False

                self.ego_ppo.end_episode()

                obs = self.ego_wrapper.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]

        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "episode_crashes": episode_crashes,
            "episode_completions": episode_completions,
            "episode_speeds": episode_speeds,
            "n_episodes": len(episode_rewards),
        }

    def _get_ego_action(self, obs2) -> int:
        """获取 EgoPPO 为当前观测选择的动作"""
        if obs2 is None:
            return 1

        try:
            obs = np.asarray(obs2).flatten()
            obs_tensor = th.from_numpy(obs).float().unsqueeze(0).to(self.ego_ppo.device)

            with th.no_grad():
                if self.ego_ppo.network.action_space == 'discrete':
                    log_probs = self.ego_ppo.network(obs_tensor)[0]
                    probs = th.exp(log_probs)
                    action = th.multinomial(probs, 1).item()
                else:
                    mean, _ = self.ego_ppo.network(obs_tensor)
                    action = mean.cpu().numpy()[0]
                    action = int(np.clip(action, 0, self.ego_config.action_dim - 1))

            return action
        except Exception:
            return np.random.randint(self.ego_config.action_dim)

    def _get_natural_actions(self):
        """Get one shared-policy action for each HDV."""
        hdv_obs_list = list(getattr(self.env, "obs_hdv_list", []))
        if not hdv_obs_list and self.mappo.obs3 is not None:
            hdv_obs_list = [self.mappo.obs3]
        if not hdv_obs_list:
            hdv_obs_list = [None] * max(len(getattr(self.env, "hdv_vehicles", [])), 1)

        actions = []
        for hdv_obs in hdv_obs_list:
            if self.natural_vehicle_model is not None and hdv_obs is not None:
                try:
                    action, _ = self.natural_vehicle_model.predict(hdv_obs)
                    actions.append(int(action))
                    continue
                except Exception:
                    pass
            actions.append(1)
        return actions

    def _get_ego_vehicle(self):
        ego_vehicle = getattr(self.env, "ego_vehicle", None)
        if ego_vehicle is not None:
            return ego_vehicle
        idx = len(getattr(self.env, "controlled_vehicles", []))
        if len(self.env.road.vehicles) > idx:
            return self.env.road.vehicles[idx]
        return None

    def _log_entry(self, log_path, fieldnames, **kwargs):
        entry = {
            "round": self._round,
            "phase": self._phase,
            "episode": 0,
            "total_adversarial_episodes": self._total_adversarial_episodes,
            "total_ego_episodes": self._total_ego_episodes,
            "adversarial_reward": 0.0,
            "ego_reward": 0.0,
            "ego_policy_loss": 0.0,
            "ego_value_loss": 0.0,
            "ego_crash_rate": 0.0,
            "ego_road_completion_rate": 0.0,
            "ego_avg_speed": 0.0,
            "adversarial_crash_rate": 0.0,
        }
        entry.update(kwargs)

        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(entry)

    def _save_final_models(self):
        mappo_dir = os.path.join(self.output_dir, "mappo_models")
        self.mappo.save(mappo_dir, self.mappo.n_episodes + 999)

        ego_dir = os.path.join(self.output_dir, "ego_models")
        final_path = os.path.join(ego_dir, "final_model.pt")
        self.ego_ppo.save(final_path)

        print(f"Final models saved to {self.output_dir}")

    def evaluate_ego(self, n_episodes: int = 10, render: bool = False) -> Dict[str, Any]:
        """
        评估主车策略 (对抗车使用 MAPPO 当前策略)
        """
        self.ego_ppo.eval_mode()
        self.ego_wrapper.mappo = self.mappo
        self.ego_wrapper.natural_vehicle_model = self.natural_vehicle_model

        all_rewards = []
        all_lengths = []
        crash_count = 0
        completion_ratios = []

        for ep in range(n_episodes):
            obs = self.ego_wrapper.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

            ep_reward = 0.0
            ep_length = 0
            ep_crashed = False
            done = False

            while not done:
                action, _, _ = self.ego_ppo.select_action(obs, deterministic=True)
                next_obs, reward, done, info = self.ego_wrapper.step(action)

                if isinstance(next_obs, tuple):
                    next_obs = next_obs[0]

                ego_vehicle = self._get_ego_vehicle()
                if ego_vehicle is not None and ego_vehicle.crashed:
                    ep_crashed = True

                ep_reward += reward
                ep_length += 1
                obs = next_obs

                if render:
                    self.env.render()

            all_rewards.append(ep_reward)
            all_lengths.append(ep_length)
            completion_ratios.append(ep_length / max(self._max_steps, 1))

            if ep_crashed:
                crash_count += 1

        return {
            "mean_reward": np.mean(all_rewards),
            "std_reward": np.std(all_rewards),
            "mean_length": np.mean(all_lengths),
            "crash_rate": crash_count / n_episodes,
            "road_completion_rate": np.mean(completion_ratios) if completion_ratios else 0,
            "rewards": all_rewards,
            "lengths": all_lengths,
        }
