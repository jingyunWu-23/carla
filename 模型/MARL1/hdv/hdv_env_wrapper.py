from typing import Any, Dict, Optional, Tuple

import gym
import numpy as np

from hdv.reward.aggressive_hdv_reward import AggressiveHDVReward


class HDVEnvWrapper(gym.Env):
    """
    Single-agent SB3 wrapper for fine-tuning the first HDV.

    The wrapped policy receives obs3 and outputs the HDV discrete action. Other
    externally controlled slots use fixed actions by default; later they can be
    replaced by frozen ego/adversarial policies without changing the HDV PPO API.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        env: gym.Env,
        reward_fn: Optional[AggressiveHDVReward] = None,
        hdv_vehicle_index: int = 4,
        ego_action: int = 1,
        other_action: int = 1,
        hdv_policy_model: Any = None,
        deterministic_seed: Optional[int] = None,
    ):
        super().__init__()
        self.env = env
        self.reward_fn = reward_fn or AggressiveHDVReward()
        self.hdv_vehicle_index = hdv_vehicle_index
        self.ego_action = int(ego_action)
        self.other_action = int(other_action)
        self.hdv_policy_model = hdv_policy_model
        self.deterministic_seed = deterministic_seed
        self.episode_count = 0
        self._obs3 = None
        self._obs_hdv_list = []

        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5, 5),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(self.env.n_a)

    def reset(self, **kwargs) -> np.ndarray:
        if self.deterministic_seed is not None and "testing_seeds" not in kwargs:
            kwargs["is_training"] = False
            kwargs["testing_seeds"] = self.deterministic_seed + self.episode_count
            self.episode_count += 1

        result = self.env.reset(**kwargs)
        if len(result) == 4:
            _, _, _, self._obs3 = result
            self._obs_hdv_list = list(getattr(self.env, "obs_hdv_list", []))
        else:
            obs = self.env.observation_type.observe()
            hdv_slot = len(self.env.controlled_vehicles) + 1
            self._obs_hdv_list = list(obs[hdv_slot:])
            self._obs3 = self._obs_hdv_list[0] if self._obs_hdv_list else obs[hdv_slot]

        self.reward_fn.reset()
        return self._extract_hdv_obs()

    def step(self, hdv_action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        n_agents = len(self.env.controlled_vehicles)
        ego_slot = n_agents
        hdv_slot = n_agents + 1
        hdv_actions = self._get_shared_hdv_actions(int(hdv_action))
        full_action = [self.other_action] * hdv_slot
        full_action[ego_slot] = self.ego_action
        full_action = full_action[:hdv_slot] + hdv_actions

        result = self.env.step(tuple(full_action))
        if len(result) >= 6:
            env_state, global_reward, done, info, obs2, self._obs3 = result
            self._obs_hdv_list = list(getattr(self.env, "obs_hdv_list", []))
        else:
            env_state, global_reward, done, info = result
            obs = self.env.observation_type.observe()
            self._obs_hdv_list = list(obs[hdv_slot:])
            self._obs3 = self._obs_hdv_list[0] if self._obs_hdv_list else obs[hdv_slot]

        hdv_vehicle = self._get_hdv_vehicle()
        reward_info = self.reward_fn.compute(
            env=self.env,
            vehicle=hdv_vehicle,
            action=hdv_action,
            done=done,
            info=info,
        )

        info["hdv_reward"] = reward_info.reward
        info["hdv_reward_components"] = reward_info.components
        info.update(reward_info.info)
        info["global_reward"] = global_reward

        return self._extract_hdv_obs(), float(reward_info.reward), bool(done), info

    def render(self, mode="human", **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self):
        return self.env.close()

    def _extract_hdv_obs(self) -> np.ndarray:
        if self._obs3 is None:
            return np.zeros((5, 5), dtype=np.float32)
        return np.asarray(self._obs3, dtype=np.float32).reshape((5, 5))

    def _get_shared_hdv_actions(self, first_action: int):
        hdv_obs_list = self._obs_hdv_list if self._obs_hdv_list else (
            [self._obs3] if self._obs3 is not None else []
        )
        n_hdv = max(len(getattr(self.env, "hdv_vehicles", [])), len(hdv_obs_list), 1)
        actions = [int(first_action)]
        for idx in range(1, n_hdv):
            hdv_obs = hdv_obs_list[idx] if idx < len(hdv_obs_list) else None
            if self.hdv_policy_model is not None and hdv_obs is not None:
                try:
                    action, _ = self.hdv_policy_model.predict(hdv_obs)
                    actions.append(int(action))
                    continue
                except Exception:
                    pass
            actions.append(int(first_action))
        return actions

    def _get_hdv_vehicle(self):
        hdv_vehicles = getattr(self.env, "hdv_vehicles", [])
        if hdv_vehicles:
            return hdv_vehicles[0]
        idx = len(getattr(self.env, "controlled_vehicles", [])) + 1
        if len(self.env.road.vehicles) > idx:
            return self.env.road.vehicles[idx]
        if len(self.env.road.vehicles) > self.hdv_vehicle_index:
            return self.env.road.vehicles[self.hdv_vehicle_index]
        return None
