from typing import Any, Dict, List, Optional, Tuple

from ego.reward.base_reward import BaseReward, RewardInfo
from ego.reward.cat_reward import CatReward, CatDrivingReward, CatSpeedReward, CatTerminalReward, CatSafetyReward
from ego.reward.safety_cost import SafetyCost, CostToRewardWrapper


class WeightedReward(BaseReward):
    """
    加权组合奖励: 将多个奖励组件按权重线性组合

    公式:
        total = Σ w_i * component_i.reward

    使用方式:
        combined = WeightedReward(
            components=[driving_reward, speed_reward, terminal_reward],
            weights=[1.0, 0.1, 1.0],
        )
    """

    def __init__(
        self,
        components: List[BaseReward],
        weights: Optional[List[float]] = None,
        name: str = "WeightedReward",
    ):
        super().__init__(name=name)
        self.components = components
        self.weights = weights or [1.0] * len(components)

        if len(self.weights) != len(self.components):
            raise ValueError(
                f"weights length ({len(self.weights)}) must match "
                f"components length ({len(self.components)})"
            )

    def reset(self):
        for comp in self.components:
            comp.reset()

    def compute(
        self,
        env: Any,
        vehicle: Any = None,
        action: Optional[Any] = None,
        done: bool = False,
        info: Optional[Dict[str, Any]] = None,
    ) -> RewardInfo:
        total = RewardInfo()
        for comp, weight in zip(self.components, self.weights):
            result = comp.compute(env, vehicle, action, done, info)
            weighted = RewardInfo(
                reward=result.reward * weight,
                components={f"{comp.name}": result.reward * weight},
                info=result.info,
            )
            total = total + weighted
        return total


class RewardFactory:
    """
    奖励函数工厂

    提供预配置的奖励函数组合, 方便快速创建不同场景的奖励函数。

    预设配置:
        - cat_default: cat-main 默认配置 (driving + speed + terminal)
        - cat_safety: cat-main + safety cost 分离 (用于 Lagrangian PPO)
        - cat_safety_merged: cat-main + safety cost 合并 (用于标准 PPO)
        - cat_driving_only: 仅驾驶奖励 (用于 ablation study)
    """

    @staticmethod
    def cat_default(**overrides) -> CatReward:
        """
        cat-main 默认奖励配置

        参数与 MetaDriveEnv 默认值一致:
            driving_reward=1.0, speed_reward=0.1,
            success_reward=10.0, out_of_road_penalty=5.0,
            crash_vehicle_penalty=5.0, crash_object_penalty=5.0
        """
        defaults = dict(
            driving_reward=1.0,
            speed_reward=0.1,
            success_reward=10.0,
            out_of_road_penalty=5.0,
            crash_vehicle_penalty=5.0,
            crash_object_penalty=5.0,
            use_lateral_reward=False,
            lane_width=4.0,
            max_speed=30.0,
        )
        defaults.update(overrides)
        return CatReward(**defaults)

    @staticmethod
    def cat_safety(
        reward_overrides: Optional[Dict] = None,
        cost_overrides: Optional[Dict] = None,
    ) -> Tuple[CatReward, SafetyCost]:
        """
        cat-main + Safety Cost 分离配置

        返回 (reward_fn, cost_fn) 元组, 用于 Lagrangian PPO。

        Returns:
            Tuple[CatReward, SafetyCost]: 奖励函数和成本函数
        """
        reward_fn = RewardFactory.cat_default(**(reward_overrides or {}))
        cost_defaults = dict(
            crash_vehicle_cost=1.0,
            crash_object_cost=1.0,
            out_of_road_cost=1.0,
        )
        cost_defaults.update(cost_overrides or {})
        cost_fn = SafetyCost(**cost_defaults)
        return reward_fn, cost_fn

    @staticmethod
    def cat_safety_merged(
        reward_overrides: Optional[Dict] = None,
        cost_overrides: Optional[Dict] = None,
    ) -> CostToRewardWrapper:
        """
        cat-main + Safety Cost 合并配置

        将 cost 直接作为 reward 的惩罚项, 用于标准 PPO。
        """
        reward_fn, cost_fn = RewardFactory.cat_safety(reward_overrides, cost_overrides)
        return CostToRewardWrapper(base_reward=reward_fn, safety_cost=cost_fn)

    @staticmethod
    def cat_safety_reward(**overrides) -> CatSafetyReward:
        """
        CAT-Safety 奖励配置

        在 CAT 基础上增加:
            - 稠密风险惩罚 (alpha=1.0, safe_distance=25.0)
            - 增大终局成功奖励 (success_reward=30.0)
        """
        defaults = dict(
            driving_reward=1.0,
            speed_reward=0.1,
            success_reward=30.0,
            out_of_road_penalty=5.0,
            crash_vehicle_penalty=5.0,
            crash_object_penalty=5.0,
            use_lateral_reward=False,
            lane_width=4.0,
            max_speed=30.0,
            alpha=1.0,
            safe_distance=25.0,
        )
        defaults.update(overrides)
        return CatSafetyReward(**defaults)

    @staticmethod
    def cat_driving_only(**overrides) -> CatDrivingReward:
        """仅驾驶奖励 (用于 ablation study)"""
        defaults = dict(driving_reward=1.0, use_lateral_reward=False, lane_width=4.0)
        defaults.update(overrides)
        return CatDrivingReward(**defaults)

    @staticmethod
    def cat_speed_only(**overrides) -> CatSpeedReward:
        """仅速度奖励 (用于 ablation study)"""
        defaults = dict(speed_reward=0.1, max_speed=30.0)
        defaults.update(overrides)
        return CatSpeedReward(**defaults)

    @staticmethod
    def cat_terminal_only(**overrides) -> CatTerminalReward:
        """仅终端奖励 (用于 ablation study)"""
        defaults = dict(
            success_reward=10.0,
            out_of_road_penalty=5.0,
            crash_vehicle_penalty=5.0,
            crash_object_penalty=5.0,
        )
        defaults.update(overrides)
        return CatTerminalReward(**defaults)


def create_reward(
    preset: str = "cat_default",
    **overrides,
) -> BaseReward:
    """
    便捷函数: 根据预设名称创建奖励函数

    Args:
        preset: 预设名称, 可选:
            - "cat_default": cat-main 默认配置
            - "cat_safety_merged": cat-main + safety cost 合并
        **overrides: 覆盖默认参数

    Returns:
        BaseReward: 奖励函数实例

    Example:
        reward_fn = create_reward("cat_default", driving_reward=2.0)
        reward_fn = create_reward("cat_safety_merged", crash_vehicle_penalty=10.0)
    """
    factory = RewardFactory()

    if preset == "cat_default":
        return factory.cat_default(**overrides)
    elif preset == "cat_safety_merged":
        return factory.cat_safety_merged(**overrides)
    elif preset == "cat_safety_reward":
        return factory.cat_safety_reward(**overrides)
    elif preset == "cat_driving_only":
        return factory.cat_driving_only(**overrides)
    elif preset == "cat_speed_only":
        return factory.cat_speed_only(**overrides)
    elif preset == "cat_terminal_only":
        return factory.cat_terminal_only(**overrides)
    else:
        raise ValueError(
            f"Unknown preset: {preset}. "
            f"Available: cat_default, cat_safety_merged, cat_safety_reward, "
            f"cat_driving_only, cat_speed_only, cat_terminal_only"
        )