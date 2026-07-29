import torch as th
from torch import nn
from torch.optim import Adam
import numpy as np
import os

from ego.model import ActorCriticNetwork
from ego.memory import RolloutBuffer
from ego.gae import compute_gae, normalize_advantages


class EgoPPO:
    """
    单智能体 PPO + GAE 算法

    核心流程:
        1. collect_rollout: 与环境交互, 收集 trajectory 存入 RolloutBuffer
        2. update: 从 buffer 采样, 多 epoch 更新 Actor/Critic
        3. select_action: 推理时选择动作

    PPO-Clip 目标函数:
        L^CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]
        其中 r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)
    """

    def __init__(self, config):
        self.config = config
        self.device = th.device(
            "cuda" if config.use_cuda and th.cuda.is_available() else "cpu"
        )

        self.network = ActorCriticNetwork(
            state_dim=config.state_dim,
            hidden_size=config.actor_hidden_size,
            action_dim=config.action_dim,
            action_space=config.action_space_type,
        ).to(self.device)

        self.optimizer = Adam(self.network.parameters(), lr=config.actor_lr)

        self.buffer = RolloutBuffer(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            capacity=config.n_steps,
            action_space=config.action_space_type,
        )

        self.n_steps = 0
        self.n_episodes = 0
        self.episode_reward = 0.0
        self.episode_length = 0

    def select_action(self, state, deterministic=False):
        state_tensor = th.FloatTensor(state).unsqueeze(0).to(self.device)

        with th.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(
                state_tensor, deterministic=deterministic
            )

        action_np = action.cpu().numpy()
        log_prob_np = log_prob.cpu().numpy()
        value_np = value.cpu().numpy()

        if self.config.action_space_type == 'discrete':
            return int(action_np[0]), float(log_prob_np[0]), float(value_np[0, 0])
        else:
            return action_np[0], float(log_prob_np[0]), float(value_np[0, 0])

    def step(self, state, action, log_prob, value, reward, done, next_state):
        self.buffer.add(state, action, log_prob, value, reward, done, next_state)
        self.n_steps += 1
        self.episode_reward += reward
        self.episode_length += 1

    def end_episode(self):
        self.n_episodes += 1
        ep_reward = self.episode_reward
        ep_length = self.episode_length
        self.episode_reward = 0.0
        self.episode_length = 0
        return ep_reward, ep_length

    def update(self):
        if self.buffer.size < self.config.batch_size:
            return {}

        states, actions, old_log_probs, old_values, rewards, dones, next_states, _, _ = \
            self.buffer.get_all()

        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        old_values = old_values.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        next_states = next_states.to(self.device)

        with th.no_grad():
            _, _, _, next_values = self.network.get_action_and_value(next_states)

        advantages, returns = compute_gae(
            rewards=rewards.cpu().numpy(),
            values=old_values.cpu().numpy(),
            dones=dones.cpu().numpy(),
            next_values=next_values.cpu().numpy().squeeze(-1),
            gamma=self.config.reward_gamma,
            gae_lambda=self.config.gae_lambda,
        )

        advantages = th.FloatTensor(advantages).to(self.device)
        returns = th.FloatTensor(returns).to(self.device)

        if self.config.normalize_advantage:
            advantages = normalize_advantages(advantages)

        n_samples = self.buffer.size
        indices = np.arange(n_samples)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.config.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                _, new_log_probs, entropy, new_values = \
                    self.network.get_action_and_value(batch_states, batch_actions)

                new_values = new_values.squeeze(-1)

                ratio = th.exp(new_log_probs - batch_old_log_probs)

                clip_range = self.config.clip_range
                surr1 = ratio * batch_advantages
                surr2 = th.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * batch_advantages
                policy_loss = -th.min(surr1, surr2).mean()

                if self.config.clip_range_vf is not None:
                    clipped_values = old_values[batch_indices] + th.clamp(
                        new_values - old_values[batch_indices],
                        -self.config.clip_range_vf,
                        self.config.clip_range_vf,
                    )
                    value_loss1 = (new_values - batch_returns) ** 2
                    value_loss2 = (clipped_values - batch_returns) ** 2
                    value_loss = 0.5 * th.max(value_loss1, value_loss2).mean()
                else:
                    value_loss = 0.5 * ((new_values - batch_returns) ** 2).mean()

                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.config.vf_coef * value_loss
                    - self.config.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                n_updates += 1

        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "n_updates": n_updates,
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        th.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'n_steps': self.n_steps,
            'n_episodes': self.n_episodes,
            'config': self.config.to_dict(),
        }, path)

    def load(self, path):
        checkpoint = th.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.n_steps = checkpoint.get('n_steps', 0)
        self.n_episodes = checkpoint.get('n_episodes', 0)

    def train_mode(self):
        self.network.train()

    def eval_mode(self):
        self.network.eval()