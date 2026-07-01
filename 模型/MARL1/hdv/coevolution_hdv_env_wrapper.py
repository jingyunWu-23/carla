from typing import Any, Dict, Optional, Tuple

import gym
import numpy as np
import torch as th

from hdv.hdv_env_wrapper import HDVEnvWrapper
from hdv.reward.aggressive_hdv_reward import AggressiveHDVReward


class CoevolutionHDVEnvWrapper(HDVEnvWrapper):
    """
    HDV SB3 wrapper for co-evolution.

    Compared with HDVEnvWrapper, this wrapper uses current MAPPO and EgoPPO
    policies to control the CAV opponents and ego vehicle while SB3 trains the
    first HDV action.
    """

    def __init__(
        self,
        env: gym.Env,
        mappo: Any = None,
        ego_ppo: Any = None,
        reward_fn: Optional[AggressiveHDVReward] = None,
        hdv_vehicle_index: int = 4,
        hdv_policy_model: Any = None,
        other_action: int = 1,
        deterministic_seed: Optional[int] = None,
    ):
        super().__init__(
            env=env,
            reward_fn=reward_fn,
            hdv_vehicle_index=hdv_vehicle_index,
            hdv_policy_model=hdv_policy_model,
            ego_action=1,
            other_action=other_action,
            deterministic_seed=deterministic_seed,
        )
        self.mappo = mappo
        self.ego_ppo = ego_ppo
        self._env_state = None
        self._obs2 = None

    def reset(self, **kwargs) -> np.ndarray:
        if self.deterministic_seed is not None and "testing_seeds" not in kwargs:
            kwargs["is_training"] = False
            kwargs["testing_seeds"] = self.deterministic_seed + self.episode_count
            self.episode_count += 1

        result = self.env.reset(**kwargs)
        if len(result) == 4:
            self._env_state, _, self._obs2, self._obs3 = result
            self._obs_hdv_list = list(getattr(self.env, "obs_hdv_list", []))
        else:
            self._env_state = result[0]
            obs = self.env.observation_type.observe()
            n_agents = len(self.env.controlled_vehicles)
            self._obs2 = obs[n_agents]
            self._obs_hdv_list = list(obs[n_agents + 1:])
            self._obs3 = self._obs_hdv_list[0] if self._obs_hdv_list else obs[n_agents + 1]

        if self.mappo is not None:
            self.mappo.env_state = self._env_state
            self.mappo.obs2 = self._obs2
            self.mappo.obs3 = self._obs3

        self.reward_fn.reset()
        return self._extract_hdv_obs()

    def step(self, hdv_action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        full_action = self._build_full_action(int(hdv_action))
        result = self.env.step(tuple(full_action))
        if len(result) >= 6:
            self._env_state, global_reward, done, info, self._obs2, self._obs3 = result
            self._obs_hdv_list = list(getattr(self.env, "obs_hdv_list", []))
        else:
            self._env_state, global_reward, done, info = result
            obs = self.env.observation_type.observe()
            n_agents = len(self.env.controlled_vehicles)
            self._obs2 = obs[n_agents]
            self._obs_hdv_list = list(obs[n_agents + 1:])
            self._obs3 = self._obs_hdv_list[0] if self._obs_hdv_list else obs[n_agents + 1]

        if self.mappo is not None:
            self.mappo.env_state = self._env_state
            self.mappo.obs2 = self._obs2
            self.mappo.obs3 = self._obs3

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

    def _build_full_action(self, hdv_action: int):
        n_agents = len(self.env.controlled_vehicles)

        if self.mappo is not None and self._env_state is not None:
            try:
                actions = list(self.mappo.exploration_action(self._env_state, n_agents))
            except Exception:
                actions = [self.other_action] * n_agents
        else:
            actions = [self.other_action] * n_agents

        ego_slot = n_agents
        hdv_slot = n_agents + 1
        while len(actions) < hdv_slot:
            actions.append(self.other_action)

        actions[ego_slot] = self._get_ego_action()
        actions = actions[:hdv_slot] + self._get_shared_hdv_actions(hdv_action)
        return actions

    def _get_ego_action(self) -> int:
        if self.ego_ppo is None or self._obs2 is None:
            return self.ego_action

        try:
            obs = np.asarray(self._obs2, dtype=np.float32).flatten()
            obs_tensor = th.from_numpy(obs).float().unsqueeze(0).to(self.ego_ppo.device)
            with th.no_grad():
                if self.ego_ppo.network.action_space == "discrete":
                    log_probs = self.ego_ppo.network(obs_tensor)[0]
                    probs = th.exp(log_probs)
                    action = th.multinomial(probs, 1).item()
                else:
                    mean, _ = self.ego_ppo.network(obs_tensor)
                    action = mean.cpu().numpy()[0]
                    action = int(np.clip(action, 0, self.env.n_a - 1))
            return int(action)
        except Exception:
            return self.ego_action
