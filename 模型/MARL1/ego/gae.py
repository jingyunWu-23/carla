import numpy as np
import torch as th


def compute_gae(rewards, values, dones, next_values, gamma, gae_lambda):
    """
    计算 Generalized Advantage Estimation (GAE)

    GAE 通过 λ 参数在 bias-variance 之间做权衡:
        λ=0: 等价于 TD(0), 低方差高偏差
        λ=1: 等价于 Monte Carlo, 高方差低偏差

    公式:
        δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t^GAE = Σ_{k=0}^{∞} (γλ)^k * δ_{t+k}

    递归实现:
        A_t = δ_t + γλ * (1 - done_t) * A_{t+1}

    Args:
        rewards:   [T] 每一步的即时奖励
        values:    [T] 每一步的状态价值 V(s_t)
        dones:     [T] 每一步的终止标志
        next_values: [T] 每一步的下一状态价值 V(s_{t+1})
        gamma:     折扣因子
        gae_lambda: GAE λ 参数

    Returns:
        advantages: [T] GAE 优势估计
        returns:    [T] 目标价值 (advantages + values)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)

    gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae

        advantages[t] = gae
        returns[t] = gae + values[t]

    return advantages, returns


def compute_gae_torch(rewards, values, dones, next_values, gamma, gae_lambda):
    """
    GAE 的 PyTorch 版本, 支持 GPU 计算

    Args:
        rewards:     [T] tensor
        values:      [T] tensor
        dones:       [T] tensor
        next_values: [T] tensor
        gamma:       float
        gae_lambda:  float

    Returns:
        advantages: [T] tensor
        returns:    [T] tensor
    """
    T = len(rewards)
    advantages = th.zeros(T, device=rewards.device)
    returns = th.zeros(T, device=rewards.device)

    gae = th.tensor(0.0, device=rewards.device)
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae

        advantages[t] = gae
        returns[t] = gae + values[t]

    return advantages, returns


def normalize_advantages(advantages):
    """
    对优势函数做标准化: (A - mean) / (std + eps)

    标准化可以降低梯度方差, 加速收敛
    """
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)