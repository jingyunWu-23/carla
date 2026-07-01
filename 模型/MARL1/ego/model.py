import torch as th
from torch import nn
from torch.distributions import Categorical, Normal
import numpy as np


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0.0)


class ActorNetwork(nn.Module):
    """
    Actor 网络: 输出动作分布参数

    离散动作空间: 输出 log_softmax, 采样时用 Categorical 分布
    连续动作空间: 输出 mean + log_std, 采样时用 Gaussian 分布
    """

    def __init__(self, state_dim, hidden_size, action_dim, action_space='discrete'):
        super(ActorNetwork, self).__init__()
        self.action_space = action_space
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        if action_space == 'discrete':
            self.fc_out = nn.Linear(hidden_size, action_dim)
            self.output_act = nn.LogSoftmax(dim=-1)
        elif action_space == 'continuous':
            self.fc_mean = nn.Linear(hidden_size, action_dim)
            self.log_std = nn.Parameter(th.zeros(action_dim))
        else:
            raise ValueError(f"Unknown action_space: {action_space}")

        self.apply(init_weights)

    def forward(self, state):
        out = th.relu(self.fc1(state))
        out = th.relu(self.fc2(out))

        if self.action_space == 'discrete':
            logits = self.fc_out(out)
            action_probs = self.output_act(logits)
            return action_probs
        elif self.action_space == 'continuous':
            mean = self.fc_mean(out)
            std = th.exp(self.log_std.clamp(-20, 2))
            return mean, std

    def get_distribution(self, state):
        if self.action_space == 'discrete':
            action_probs = self.forward(state)
            dist = Categorical(logits=action_probs)
        elif self.action_space == 'continuous':
            mean, std = self.forward(state)
            dist = Normal(mean, std)
        return dist

    def get_action(self, state, deterministic=False):
        dist = self.get_distribution(state)
        if deterministic:
            if self.action_space == 'discrete':
                action = th.argmax(dist.probs, dim=-1)
            else:
                action = dist.mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        if self.action_space == 'continuous':
            log_prob = log_prob.sum(dim=-1)
        return action, log_prob

    def evaluate_actions(self, state, action):
        dist = self.get_distribution(state)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        if self.action_space == 'continuous':
            log_prob = log_prob.sum(dim=-1)
            entropy = entropy.sum(dim=-1)
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Critic 网络: 输出状态价值 V(s)

    标准 PPO 中 Critic 只估计状态价值, 不依赖动作
    """

    def __init__(self, state_dim, hidden_size):
        super(CriticNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)

        self.apply(init_weights)

    def forward(self, state):
        out = th.relu(self.fc1(state))
        out = th.relu(self.fc2(out))
        value = self.fc_out(out)
        return value


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic 共享网络: 共享底层特征提取, 分离 Actor/Critic 输出头

    相比独立网络更高效, 适合状态空间较大的场景
    """

    def __init__(self, state_dim, hidden_size, action_dim, action_space='discrete'):
        super(ActorCriticNetwork, self).__init__()
        self.action_space = action_space
        self.action_dim = action_dim

        self.shared_fc1 = nn.Linear(state_dim, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)

        if action_space == 'discrete':
            self.actor_fc = nn.Linear(hidden_size, action_dim)
            self.actor_act = nn.LogSoftmax(dim=-1)
        elif action_space == 'continuous':
            self.actor_mean = nn.Linear(hidden_size, action_dim)
            self.log_std = nn.Parameter(th.zeros(action_dim))

        self.critic_fc = nn.Linear(hidden_size, 1)

        self.apply(init_weights)

    def forward(self, state):
        shared = th.relu(self.shared_fc1(state))
        shared = th.relu(self.shared_fc2(shared))

        if self.action_space == 'discrete':
            action_probs = self.actor_act(self.actor_fc(shared))
            value = self.critic_fc(shared)
            return action_probs, value
        elif self.action_space == 'continuous':
            mean = self.actor_mean(shared)
            std = th.exp(self.log_std.clamp(-20, 2))
            value = self.critic_fc(shared)
            return (mean, std), value

    def get_distribution(self, state):
        shared = th.relu(self.shared_fc1(state))
        shared = th.relu(self.shared_fc2(shared))

        if self.action_space == 'discrete':
            action_probs = self.actor_act(self.actor_fc(shared))
            dist = Categorical(logits=action_probs)
        elif self.action_space == 'continuous':
            mean = self.actor_mean(shared)
            std = th.exp(self.log_std.clamp(-20, 2))
            dist = Normal(mean, std)
        return dist

    def get_value(self, state):
        shared = th.relu(self.shared_fc1(state))
        shared = th.relu(self.shared_fc2(shared))
        return self.critic_fc(shared)

    def get_action_and_value(self, state, action=None, deterministic=False):
        shared = th.relu(self.shared_fc1(state))
        shared = th.relu(self.shared_fc2(shared))

        if self.action_space == 'discrete':
            action_probs = self.actor_act(self.actor_fc(shared))
            dist = Categorical(logits=action_probs)
        elif self.action_space == 'continuous':
            mean = self.actor_mean(shared)
            std = th.exp(self.log_std.clamp(-20, 2))
            dist = Normal(mean, std)

        value = self.critic_fc(shared)

        if action is None:
            if deterministic:
                if self.action_space == 'discrete':
                    action = th.argmax(dist.probs, dim=-1)
                else:
                    action = dist.mean
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        if self.action_space == 'continuous':
            log_prob = log_prob.sum(dim=-1)
            entropy = entropy.sum(dim=-1)

        return action, log_prob, entropy, value