from typing import Any, Dict, Optional

from ego.reward.base_reward import BaseReward, RewardInfo


class SafetyCost(BaseReward):
    """
    Safety Cost 约束模块

    对应 SafeMetaDriveEnv 的 cost_function():
        - crash_vehicle_cost: 碰撞其他车辆的成本
        - crash_object_cost: 碰撞障碍物的成本
        - out_of_road_cost: 驶出道路的成本

    在 Safety-Critical RL 中, cost 与 reward 分离:
        - reward: 用于策略优化 (PPO 最大化)
        - cost: 用于安全约束 (CMDP 约束满足)

    默认参数与 SafeMetaDriveEnv 一致:
        crash_vehicle_cost = 1.0
        crash_object_cost = 1.0
        out_of_road_cost = 1.0

    使用方式:
        cost_fn = SafetyCost()
        result = cost_fn.compute(env, vehicle)
        cost = result.reward  # 注意: 这里 reward 字段实际存储的是 cost 值
    """

    def __init__(
        self,
        crash_vehicle_cost: float = 1.0,
        crash_object_cost: float = 1.0,
        out_of_road_cost: float = 1.0,
        name: str = "SafetyCost",
    ):
        super().__init__(name=name)
        self.crash_vehicle_cost = crash_vehicle_cost
        self.crash_object_cost = crash_object_cost
        self.out_of_road_cost = out_of_road_cost

        self._episode_cost = 0.0

    def reset(self):
        self._episode_cost = 0.0

    def compute(
        self,
        env: Any,
        vehicle: Any = None,
        action: Optional[Any] = None,
        done: bool = False,
        info: Optional[Dict[str, Any]] = None,
    ) -> RewardInfo:
        if vehicle is None:
            vehicle = env.vehicle

        cost = 0.0
        cost_type = "none"

        if not vehicle.on_road:
            cost = self.out_of_road_cost
            cost_type = "out_of_road"
        elif vehicle.crashed:
            cost = self.crash_vehicle_cost
            cost_type = "crash_vehicle"

        self._episode_cost += cost

        return RewardInfo(
            reward=cost,
            components={"safety_cost": cost},
            info={
                "cost_type": cost_type,
                "episode_cost": self._episode_cost,
            },
        )

    @property
    def episode_cost(self) -> float:
        return self._episode_cost


class CostToRewardWrapper(BaseReward):
    """
    将 Safety Cost 转换为奖励惩罚项

    对应 SafeMetaDriveEnv._post_process_config() 中的 cost_to_reward 逻辑:
        crash_vehicle_penalty += crash_vehicle_cost
        crash_object_penalty += crash_object_cost
        out_of_road_penalty += out_of_road_cost

    当 cost_to_reward=True 时, cost 被合并到 reward 的惩罚项中,
    此时 PPO 直接优化 reward (包含 safety), 而非使用 Lagrangian 方法。

    使用方式:
        # 方式1: 分离 reward 和 cost (推荐, 用于 Lagrangian PPO)
        reward_fn = CatReward()
        cost_fn = SafetyCost()

        # 方式2: 合并到 reward (简单, 用于标准 PPO)
        combined = CostToRewardWrapper(
            base_reward=CatReward(),
            safety_cost=SafetyCost(),
        )
    """

    def __init__(
        self,
        base_reward: BaseReward,
        safety_cost: SafetyCost,
        name: str = "CostToReward",
    ):
        super().__init__(name=name)
        self.base_reward = base_reward
        self.safety_cost = safety_cost

    def reset(self):
        self.base_reward.reset()
        self.safety_cost.reset()

    def compute(
        self,
        env: Any,
        vehicle: Any = None,
        action: Optional[Any] = None,
        done: bool = False,
        info: Optional[Dict[str, Any]] = None,
    ) -> RewardInfo:
        base_result = self.base_reward.compute(env, vehicle, action, done, info)
        cost_result = self.safety_cost.compute(env, vehicle, action, done, info)

        combined_reward = base_result.reward - cost_result.reward

        return RewardInfo(
            reward=combined_reward,
            components={
                **base_result.components,
                "safety_penalty": -cost_result.reward,
            },
            info={
                **base_result.info,
                **cost_result.info,
            },
        )