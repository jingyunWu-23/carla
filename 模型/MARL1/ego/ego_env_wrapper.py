import numpy as np
import gym
from typing import Tuple, Dict, Any, Optional

from ego.reward.base_reward import BaseReward, RewardInfo
from ego.reward.cat_reward import CatReward, CatSafetyReward


class EgoEnvWrapper:
    """
    Single-agent EgoPPO wrapper for the multi-agent MergeEnv.

    Role mapping is dynamic:
        controlled_vehicles[*] -> adversarial CAVs controlled by MAPPO
        env.ego_vehicle        -> ego vehicle controlled by EgoPPO
        env.hdv_vehicles[*]    -> HDVs controlled by a shared HDV policy
    """

    def __init__(
        self,
        env: gym.Env,
        mappo: Any = None,
        reward_fn: Optional[BaseReward] = None,
        natural_vehicle_model: Any = None,
        ego_vehicle_index: int = 3,
        natural_vehicle_index: int = 4,
    ):
        self.env = env
        self.mappo = mappo
        self.reward_fn = reward_fn or CatReward()
        self.natural_vehicle_model = natural_vehicle_model
        self.ego_vehicle_index = ego_vehicle_index
        self.natural_vehicle_index = natural_vehicle_index

        self._obs2 = None
        self._obs3 = None
        self._obs_hdv_list = []
        self._last_ego_obs = None
        self._step_count = 0
        self._episode_speeds = []

    def reset(self, **kwargs) -> np.ndarray:
        """
        閲嶇疆鐜骞惰繑鍥炰富杞﹁娴?
        Returns:
            ego_obs: 涓昏溅鐨勮娴嬪悜閲?(n_s,)
        """
        result = self.env.reset(**kwargs)

        if isinstance(result, tuple):
            if len(result) == 4:
                env_state, action_mask, self._obs2, self._obs3 = result
                self._obs_hdv_list = list(getattr(self.env, "obs_hdv_list", []))
            elif len(result) == 3:
                env_state, action_mask, self._obs2 = result
                self._obs3 = None
                self._obs_hdv_list = []
            elif len(result) == 2:
                env_state, action_mask = result
                self._obs2 = None
                self._obs3 = None
                self._obs_hdv_list = []
            else:
                env_state = result[0]
                self._obs2 = result[2] if len(result) > 2 else None
                self._obs3 = result[3] if len(result) > 3 else None
                self._obs_hdv_list = list(getattr(self.env, "obs_hdv_list", []))
        else:
            env_state = result
            self._obs2 = None
            self._obs3 = None
            self._obs_hdv_list = []

        if self.mappo is not None:
            self.mappo.env_state = env_state
            self.mappo.obs2 = self._obs2
            self.mappo.obs3 = self._obs3

        self.reward_fn.reset()
        self._last_ego_obs = self._extract_ego_obs()
        self._step_count = 0
        self._episode_speeds = []
        return self._last_ego_obs

    def step(self, ego_action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        鎵ц涓€姝ョ幆澧冧氦浜?
        Args:
            ego_action: EgoPPO 涓轰富杞﹂€夋嫨鐨勫姩浣?(int, 0~4)

        Returns:
            ego_obs: 涓昏溅涓嬩竴鏃跺埢瑙傛祴
            ego_reward: cat-main 濂栧姳
            done: 鏄惁缁堟
            info: 鐜淇℃伅瀛楀吀
        """
        self._step_count += 1
        n_agents = len(self.env.controlled_vehicles)
        ego_slot = n_agents
        natural_slot = n_agents + 1

        if self.mappo is not None:
            try:
                mappo_actions = self.mappo.exploration_action(
                    self.mappo.env_state, n_agents
                )
                if mappo_actions is None:
                    mappo_actions = [1] * n_agents
            except Exception:
                mappo_actions = [1] * n_agents
        else:
            mappo_actions = [np.random.randint(self.env.n_a) for _ in range(n_agents)]

        natural_actions = self._get_natural_actions()

        full_action = list(mappo_actions)
        while len(full_action) < natural_slot:
            full_action.append(1)
        full_action[ego_slot] = ego_action
        full_action = full_action[:natural_slot] + natural_actions

        result = self.env.step(tuple(full_action))

        if len(result) >= 5:
            env_state, global_reward, done, info, self._obs2, self._obs3 = result
            self._obs_hdv_list = list(getattr(self.env, "obs_hdv_list", []))
        else:
            env_state, global_reward, done, info = result
            obs1 = self.env.observation_type.observe()
            self._obs2 = obs1[ego_slot]
            self._obs_hdv_list = list(obs1[natural_slot:])
            self._obs3 = self._obs_hdv_list[0] if self._obs_hdv_list else None

        if self.mappo is not None:
            self.mappo.env_state = env_state
            self.mappo.obs2 = self._obs2
            self.mappo.obs3 = self._obs3

        ego_vehicle = self._get_ego_vehicle()
        ego_reward = self._compute_ego_reward(ego_vehicle, done, info)

        if ego_vehicle is not None:
            self._episode_speeds.append(ego_vehicle.speed)

        self._last_ego_obs = self._extract_ego_obs()

        info["ego_reward"] = ego_reward
        info["global_reward"] = global_reward
        info["ego_avg_speed"] = (np.mean(self._episode_speeds)
                                 if self._episode_speeds else 0.0)

        return self._last_ego_obs, ego_reward, done, info

    def _extract_ego_obs(self) -> np.ndarray:
        """浠庡鏅鸿兘浣撹娴嬩腑鎻愬彇涓昏溅瑙傛祴"""
        if self._obs2 is not None:
            obs = np.asarray(self._obs2).flatten()
            return obs
        return np.zeros(self.env.n_s, dtype=np.float32)

    def _get_ego_vehicle(self):
        """鑾峰彇涓昏溅瀵硅薄"""
        ego_vehicle = getattr(self.env, "ego_vehicle", None)
        if ego_vehicle is not None:
            return ego_vehicle
        idx = len(getattr(self.env, "controlled_vehicles", []))
        if len(self.env.road.vehicles) > idx:
            return self.env.road.vehicles[idx]
        if len(self.env.road.vehicles) > self.ego_vehicle_index:
            return self.env.road.vehicles[self.ego_vehicle_index]
        return None

    def _get_natural_actions(self):
        """Get one shared-policy action for each HDV."""
        hdv_obs_list = self._obs_hdv_list if self._obs_hdv_list else (
            [self._obs3] if self._obs3 is not None else []
        )
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

    def _get_natural_action(self) -> int:
        return self._get_natural_actions()[0]

    def _compute_ego_reward(self, ego_vehicle, done: bool, info: Dict) -> float:
        """浣跨敤 cat-main 濂栧姳鍑芥暟璁＄畻涓昏溅濂栧姳"""
        if ego_vehicle is None:
            return 0.0

        result = self.reward_fn.compute(
            env=self.env,
            vehicle=ego_vehicle,
            done=done,
            info=info,
        )
        return result.reward

    def render(self, mode='human', **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self):
        self.env.close()

    @property
    def observation_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.env.n_s,))

    @property
    def action_space(self):
        return gym.spaces.Discrete(self.env.n_a)

    @property
    def n_s(self):
        return self.env.n_s

    @property
    def n_a(self):
        return self.env.n_a

    @property
    def ego_vehicle(self):
        return self._get_ego_vehicle()

    @property
    def ego_obs(self):
        return self._last_ego_obs
