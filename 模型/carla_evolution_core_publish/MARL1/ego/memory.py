import numpy as np
import torch as th
from collections import namedtuple


RolloutBatch = namedtuple(
    "RolloutBatch",
    ["states", "actions", "log_probs", "values", "rewards", "dones", "next_states"]
)


class RolloutBuffer:
    """
    On-policy 经验缓冲区

    存储完整 rollout 轨迹, 支持 GAE 计算和 mini-batch 采样。
    与现有 OnPolicyReplayMemory 的区别:
        1. 存储 log_prob 和 value, 用于 PPO 的 importance sampling 和 GAE
        2. 存储 next_state 和 done, 用于 GAE 的 bootstrap
        3. 支持 mini-batch 随机采样 (PPO 多 epoch 更新)
    """

    def __init__(self, state_dim, action_dim, capacity, action_space='discrete'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.action_space = action_space

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64) if action_space == 'discrete' else np.zeros((capacity, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)

        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)

        self.position = 0
        self.full = False

    def add(self, state, action, log_prob, value, reward, done, next_state):
        idx = self.position

        self.states[idx] = state
        if self.action_space == 'discrete':
            self.actions[idx] = action
        else:
            self.actions[idx] = action
        self.log_probs[idx] = log_prob
        self.values[idx] = value
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.next_states[idx] = next_state

        self.position = (self.position + 1) % self.capacity
        if self.position == 0:
            self.full = True

    def compute_gae(self, last_value, gamma, gae_lambda):
        """
        计算 GAE 优势估计和 Returns

        GAE 公式:
            δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t = δ_t + γλ * (1 - done_t) * A_{t+1}
            R_t = A_t + V(s_t)
        """
        n = self.size
        gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * mask - self.values[t]
            gae = delta + gamma * gae_lambda * mask * gae

            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

    def get_batch(self, batch_size=None):
        """
        获取全部数据用于 PPO 更新

        返回 tensors, 支持 mini-batch 随机索引采样
        """
        n = self.size
        if batch_size is None or batch_size >= n:
            indices = np.arange(n)
        else:
            indices = np.random.permutation(n)[:batch_size]

        return RolloutBatch(
            states=th.FloatTensor(self.states[indices]),
            actions=th.LongTensor(self.actions[indices]) if self.action_space == 'discrete' else th.FloatTensor(self.actions[indices]),
            log_probs=th.FloatTensor(self.log_probs[indices]),
            values=th.FloatTensor(self.values[indices]),
            rewards=th.FloatTensor(self.rewards[indices]),
            dones=th.FloatTensor(self.dones[indices]),
            next_states=th.FloatTensor(self.next_states[indices]),
        )

    def get_all(self):
        n = self.size
        return (
            th.FloatTensor(self.states[:n]),
            th.LongTensor(self.actions[:n]) if self.action_space == 'discrete' else th.FloatTensor(self.actions[:n]),
            th.FloatTensor(self.log_probs[:n]),
            th.FloatTensor(self.values[:n]),
            th.FloatTensor(self.rewards[:n]),
            th.FloatTensor(self.dones[:n]),
            th.FloatTensor(self.next_states[:n]),
            th.FloatTensor(self.advantages[:n]),
            th.FloatTensor(self.returns[:n]),
        )

    def clear(self):
        self.position = 0
        self.full = False

    @property
    def size(self):
        return self.capacity if self.full else self.position

    def __len__(self):
        return self.size