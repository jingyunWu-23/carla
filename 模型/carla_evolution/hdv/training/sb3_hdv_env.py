from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    import gym
    from gym import spaces

from carla_evolution.envs.factory import make_env
from carla_evolution.hdv.reward import AggressiveHDVReward


class CarlaHDVSB3Env(gym.Env):
    """Single-HDV SB3 wrapper over EvolutionEnv.

    Action order in the underlying scenario is:
    [adv CAV..., ego, HDV..., background...].
    This wrapper trains one shared HDV slot while keeping other vehicles on
    fixed actions.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config_path: Optional[str] = None,
        backend: Optional[str] = None,
        reward_weights: Optional[str] = None,
        num_cav: int = 3,
        num_hdv: int = 3,
        num_background: int = 0,
        target_hdv_index: int = 0,
        ego_action: int = 1,
        adv_action: int = 1,
        other_hdv_action: int = 1,
        seed: int = 0,
        max_episode_steps: Optional[int] = None,
        mappo_policy: Any = None,
        ego_policy: Any = None,
        other_hdv_policy: Any = None,
        deterministic_cav: bool = True,
    ):
        super().__init__()
        self.config_path = config_path
        self.backend = backend
        self.num_cav = int(num_cav)
        self.num_hdv = int(num_hdv)
        self.num_background = int(num_background)
        self.target_hdv_index = int(target_hdv_index)
        self.ego_action = int(ego_action)
        self.adv_action = int(adv_action)
        self.other_hdv_action = int(other_hdv_action)
        self.seed_value = int(seed)
        self.mappo_policy = mappo_policy
        self.ego_policy = ego_policy
        self.other_hdv_policy = other_hdv_policy
        self.deterministic_cav = bool(deterministic_cav)
        self._state = None
        config_override = {"backend": backend} if backend else {}
        if max_episode_steps is not None:
            config_override["max_episode_steps"] = int(max_episode_steps)
        self.env = make_env(
            config=config_override or None,
            config_path=config_path,
            legacy=False,
        )
        self.reward_fn = AggressiveHDVReward(weights_path=reward_weights)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed_value = int(seed)
        self.reward_fn.reset()
        self._state, info = self.env.reset(
            seed=self.seed_value,
            options={
                "num_cav": self.num_cav,
                "num_hdv": self.num_hdv,
                "num_background": self.num_background,
                **(options or {}),
            },
        )
        return self._obs(), info

    def step(self, action):
        scenario = self.env.scenario_manager.state
        actions = self._full_action_vector(int(action))
        self._state, _, terminated, truncated, info = self.env.step(actions)
        scenario = self.env.scenario_manager.state
        hdv = self._target_hdv(scenario)
        reward_result = self.reward_fn.compute(scenario, hdv, action=action, done=terminated or truncated, info=info)
        info = {**info, **reward_result.info}
        if hdv is not None:
            info["hdv_speed"] = float(getattr(hdv, "speed", 0.0))
            info["hdv_crash"] = bool(getattr(hdv, "crashed", False))
            info["hdv_route_s"] = float(getattr(hdv, "route_s", 0.0) or 0.0)
            route_length = float(getattr(scenario, "route_length", 1.0) or 1.0)
            info["hdv_route_completion"] = float(np.clip(info["hdv_route_s"] / max(route_length, 1e-6), 0.0, 1.0))
        else:
            info["hdv_speed"] = 0.0
            info["hdv_crash"] = False
            info["hdv_route_s"] = 0.0
            info["hdv_route_completion"] = 0.0
        for key, value in reward_result.components.items():
            info[f"hdv_reward_{key}"] = value
        return self._obs(), float(reward_result.reward), bool(terminated), bool(truncated), info

    def close(self):
        self.env.close()

    def _obs(self) -> np.ndarray:
        obs_list = list(getattr(self.env, "obs_hdv_list", []))
        if not obs_list:
            return np.zeros(25, dtype=np.float32)
        target_index = int(np.clip(self.target_hdv_index, 0, len(obs_list) - 1))
        return np.asarray(obs_list[target_index], dtype=np.float32).reshape(-1)[:25]

    def _target_hdv(self, scenario):
        hdvs = list(getattr(scenario, "hdv_vehicles", []))
        if not hdvs:
            return None
        target_index = int(np.clip(self.target_hdv_index, 0, len(hdvs) - 1))
        return hdvs[target_index]

    def _full_action_vector(self, target_action: int):
        scenario = self.env.scenario_manager.state
        adv_count = len(getattr(scenario, "adv_cav_vehicles", []))
        hdv_count = len(getattr(scenario, "hdv_vehicles", []))
        background_count = len(getattr(scenario, "background_vehicles", []))
        actions = self._adv_actions(self._state, adv_count)
        actions.append(self._ego_action())
        hdv_observations = list(getattr(self.env, "obs_hdv_list", []))
        for idx in range(hdv_count):
            if idx == self.target_hdv_index:
                actions.append(target_action)
            else:
                obs = hdv_observations[idx] if idx < len(hdv_observations) else np.zeros(25, dtype=np.float32)
                actions.append(self._other_hdv_action(obs))
        actions.extend([1] * background_count)
        return actions

    def _adv_actions(self, state, adv_count: int):
        if self.mappo_policy is None or state is None:
            return [self.adv_action] * adv_count
        try:
            if self.deterministic_cav:
                return list(self.mappo_policy.action(state, adv_count))
            return list(self.mappo_policy.exploration_action(state, adv_count))
        except Exception:
            return [self.adv_action] * adv_count

    def _ego_action(self):
        if self.ego_policy is None:
            return self.ego_action
        try:
            ego_obs = np.asarray(getattr(self.env, "obs2", []), dtype=np.float32).reshape(-1)
            action, _, _ = self.ego_policy.select_action(ego_obs, deterministic=self.deterministic_cav)
            return int(action)
        except Exception:
            return self.ego_action

    def _other_hdv_action(self, obs):
        if self.other_hdv_policy is None:
            return self.other_hdv_action
        try:
            action, _ = self.other_hdv_policy.predict(np.asarray(obs, dtype=np.float32).reshape(-1)[:25], deterministic=True)
            return int(action)
        except Exception:
            return self.other_hdv_action
