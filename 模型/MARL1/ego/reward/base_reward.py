from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class RewardInfo:
    """
    奖励函数返回的结构化信息

    包含:
        reward: 总奖励值
        components: 各子奖励分量的字典, 便于调试和日志记录
        info: 额外的诊断信息 (如 route_completion, lateral_dist 等)
    """

    reward: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)

    def __add__(self, other: "RewardInfo") -> "RewardInfo":
        merged_components = {**self.components, **other.components}
        merged_info = {**self.info, **other.info}
        return RewardInfo(
            reward=self.reward + other.reward,
            components=merged_components,
            info=merged_info,
        )

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __float__(self) -> float:
        return self.reward

    def __repr__(self) -> str:
        comp_str = ", ".join(f"{k}={v:.4f}" for k, v in self.components.items())
        return f"RewardInfo(reward={self.reward:.4f}, components=[{comp_str}])"


class BaseReward(ABC):
    """
    奖励函数抽象基类

    所有奖励组件必须继承此类并实现 compute() 方法。

    设计原则:
        1. 每个奖励组件独立计算, 返回 RewardInfo
        2. 通过 RewardFactory 组合多个组件
        3. 组件之间无耦合, 可自由替换
    """

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def compute(
        self,
        env: Any,
        vehicle: Any = None,
        action: Optional[Any] = None,
        done: bool = False,
        info: Optional[Dict[str, Any]] = None,
    ) -> RewardInfo:
        """
        计算奖励

        Args:
            env: 环境对象 (highway_env 的 AbstractEnv 实例)
            vehicle: 目标车辆对象, None 表示使用 env.vehicle
            action: 当前动作
            done: 是否终止
            info: 环境返回的 info 字典

        Returns:
            RewardInfo: 结构化的奖励信息
        """
        pass

    def reset(self):
        """每个 episode 开始时调用, 重置内部状态"""
        pass