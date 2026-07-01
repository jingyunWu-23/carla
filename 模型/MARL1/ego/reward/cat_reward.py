import numpy as np
from typing import Any, Dict, Optional

from ego.reward.base_reward import BaseReward, RewardInfo


class CatDrivingReward(BaseReward):
    """
    cat-main 驾驶奖励: 沿车道前进的稠密奖励

    对应 MetaDrive 中的:
        reward += driving_reward * (long_now - long_last) * lateral_factor

    适配 highway_env:
        - 使用 lane.local_coordinates() 获取车道局部坐标
        - long_now - long_last: 沿车道方向的位移增量
        - lateral_factor: 横向偏移惩罚因子

    默认参数与 cat-main 的 MetaDriveEnv 一致:
        driving_reward = 1.0
        use_lateral_reward = False
        lane_width = 4.0 (highway_env 默认车道宽度)
    """

    def __init__(
        self,
        driving_reward: float = 1.0,
        use_lateral_reward: bool = False,
        lane_width: float = 4.0,
        name: str = "CatDrivingReward",
    ):
        super().__init__(name=name)
        self.driving_reward = driving_reward
        self.use_lateral_reward = use_lateral_reward
        self.lane_width = lane_width

        self._last_long = None
        self._last_lane_index = None

    def reset(self):
        self._last_long = None
        self._last_lane_index = None

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

        lane = vehicle.lane
        if lane is None:
            return RewardInfo(reward=0.0, components={"driving": 0.0})

        long_now, lateral_now = lane.local_coordinates(vehicle.position)

        if self._last_long is None:
            self._last_long = long_now
            self._last_lane_index = vehicle.lane_index
            return RewardInfo(reward=0.0, components={"driving": 0.0})

        if vehicle.lane_index != self._last_lane_index:
            self._last_long = long_now
            self._last_lane_index = vehicle.lane_index
            return RewardInfo(reward=0.0, components={"driving": 0.0})

        forward_progress = long_now - self._last_long
        self._last_long = long_now

        if self.use_lateral_reward:
            lateral_factor = np.clip(
                1.0 - 2.0 * abs(lateral_now) / self.lane_width, 0.0, 1.0
            )
        else:
            lateral_factor = 1.0

        driving = self.driving_reward * forward_progress * lateral_factor

        return RewardInfo(
            reward=driving,
            components={"driving": driving},
            info={
                "forward_progress": forward_progress,
                "lateral_offset": lateral_now,
                "lateral_factor": lateral_factor,
            },
        )


class CatSpeedReward(BaseReward):
    """
    cat-main 速度奖励: 鼓励车辆维持较高速度

    对应 MetaDrive 中的:
        reward += speed_reward * (vehicle.speed_km_h / vehicle.max_speed_km_h)

    适配 highway_env:
        - highway_env 速度单位为 m/s, 直接使用 speed / max_speed 比例
        - 默认 max_speed = 30 m/s (对应 MDPVehicle.SPEED_MAX)

    默认参数与 cat-main 一致:
        speed_reward = 0.1
    """

    def __init__(
        self,
        speed_reward: float = 0.1,
        max_speed: float = 30.0,
        name: str = "CatSpeedReward",
    ):
        super().__init__(name=name)
        self.speed_reward = speed_reward
        self.max_speed = max_speed

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

        speed_ratio = vehicle.speed / max(self.max_speed, 1e-6)
        speed = self.speed_reward * speed_ratio

        return RewardInfo(
            reward=speed,
            components={"speed": speed},
            info={"speed": vehicle.speed, "speed_ratio": speed_ratio},
        )


class CatTerminalReward(BaseReward):
    """
    cat-main 终端奖励: 到达目的地 / 碰撞 / 驶出道路的稀疏奖励

    对应 MetaDrive 中的:
        if arrive_destination:   reward = +success_reward
        elif out_of_road:        reward = -out_of_road_penalty
        elif crash_vehicle:      reward = -crash_vehicle_penalty
        elif crash_object:       reward = -crash_object_penalty

    适配 highway_env:
        - arrive_destination: 通过 route_completion >= 0.95 判断
        - out_of_road: 通过 vehicle.on_road 判断
        - crash_vehicle: 通过 vehicle.crashed 判断
        - crash_object: highway_env 无此概念, 保留接口

    默认参数与 cat-main 的 MetaDriveEnv 一致:
        success_reward = 10.0
        out_of_road_penalty = 5.0
        crash_vehicle_penalty = 5.0
        crash_object_penalty = 5.0
    """

    def __init__(
        self,
        success_reward: float = 10.0,
        out_of_road_penalty: float = 5.0,
        crash_vehicle_penalty: float = 5.0,
        crash_object_penalty: float = 5.0,
        route_completion_threshold: float = 0.95,
        name: str = "CatTerminalReward",
    ):
        super().__init__(name=name)
        self.success_reward = success_reward
        self.out_of_road_penalty = out_of_road_penalty
        self.crash_vehicle_penalty = crash_vehicle_penalty
        self.crash_object_penalty = crash_object_penalty
        self.route_completion_threshold = route_completion_threshold

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

        if not done:
            return RewardInfo(reward=0.0, components={"terminal": 0.0})

        route_completion = self._get_route_completion(env, vehicle)

        if route_completion is not None and route_completion >= self.route_completion_threshold:
            terminal = self.success_reward
            terminal_type = "success"
        elif not vehicle.on_road:
            terminal = -self.out_of_road_penalty
            terminal_type = "out_of_road"
        elif vehicle.crashed:
            terminal = -self.crash_vehicle_penalty
            terminal_type = "crash_vehicle"
        else:
            terminal = 0.0
            terminal_type = "timeout"

        return RewardInfo(
            reward=terminal,
            components={"terminal": terminal},
            info={
                "terminal_type": terminal_type,
                "route_completion": route_completion,
            },
        )

    def _get_route_completion(self, env: Any, vehicle: Any) -> Optional[float]:
        """
        计算路线完成度

        highway_env 中通过车辆在车道上的纵向位置估算完成度。
        对于 MergeEnv, 终点在 x ≈ 510 (200+80+80+150)。
        """
        try:
            if hasattr(env, 'road') and env.road is not None:
                total_length = sum(getattr(env, 'ends', [200, 80, 80, 150]))
                current_x = vehicle.position[0]
                return min(current_x / max(total_length, 1e-6), 1.0)
        except Exception:
            pass
        return None


class CatReward(BaseReward):
    """
    cat-main 完整奖励函数组合

    将 Driving + Speed + Terminal 三个子奖励组合为一个完整的 cat-main 风格奖励。

    对应 MetaDrive 的 reward_function():
        reward = driving_reward * (long_now - long_last) * lateral_factor
               + speed_reward * (speed / max_speed)
        然后根据终止条件覆盖为 success/out_of_road/crash 奖励

    使用方式:
        reward_fn = CatReward()
        result = reward_fn.compute(env, vehicle, done=done)
        reward = result.reward
    """

    def __init__(
        self,
        driving_reward: float = 1.0,
        speed_reward: float = 0.1,
        success_reward: float = 10.0,
        out_of_road_penalty: float = 5.0,
        crash_vehicle_penalty: float = 5.0,
        crash_object_penalty: float = 5.0,
        use_lateral_reward: bool = False,
        lane_width: float = 4.0,
        max_speed: float = 30.0,
        route_completion_threshold: float = 0.95,
        name: str = "CatReward",
    ):
        super().__init__(name=name)

        self.driving = CatDrivingReward(
            driving_reward=driving_reward,
            use_lateral_reward=use_lateral_reward,
            lane_width=lane_width,
        )
        self.speed = CatSpeedReward(
            speed_reward=speed_reward,
            max_speed=max_speed,
        )
        self.terminal = CatTerminalReward(
            success_reward=success_reward,
            out_of_road_penalty=out_of_road_penalty,
            crash_vehicle_penalty=crash_vehicle_penalty,
            crash_object_penalty=crash_object_penalty,
            route_completion_threshold=route_completion_threshold,
        )

    def reset(self):
        self.driving.reset()
        self.speed.reset()
        self.terminal.reset()

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

        if done:
            terminal_result = self.terminal.compute(env, vehicle, action, done, info)
            return terminal_result

        driving_result = self.driving.compute(env, vehicle, action, done, info)
        speed_result = self.speed.compute(env, vehicle, action, done, info)

        return driving_result + speed_result


class CatSafetyReward(CatReward):
    """
    CAT-Safety 奖励函数: CAT 基础奖励 + 稠密风险惩罚 + 增大终局成功奖励

    公式:
        R = R_driving + R_speed - alpha * f_risk(d) + R_terminal

    其中:
        f_risk(d) = max(0, 1 - d / D_safe)
        d = 主车到最近对抗车的欧氏距离
        D_safe = 安全距离阈值 (默认 25m)
        alpha = 风险惩罚权重 (默认 1.0)

    改进点:
        1. 稠密风险惩罚: 即使没有碰撞, 只要对抗车逼近就给予惩罚
        2. 增大终局成功奖励: 引导主车牺牲短期速度换取长期安全

    使用方式:
        reward_fn = CatSafetyReward(alpha=1.0, safe_distance=25.0, success_reward=30.0)
    """

    def __init__(
        self,
        driving_reward: float = 1.0,
        speed_reward: float = 0.1,
        success_reward: float = 30.0,
        out_of_road_penalty: float = 5.0,
        crash_vehicle_penalty: float = 5.0,
        crash_object_penalty: float = 5.0,
        use_lateral_reward: bool = False,
        lane_width: float = 4.0,
        max_speed: float = 30.0,
        route_completion_threshold: float = 0.95,
        alpha: float = 1.0,
        safe_distance: float = 25.0,
        name: str = "CatSafetyReward",
    ):
        super().__init__(
            driving_reward=driving_reward,
            speed_reward=speed_reward,
            success_reward=success_reward,
            out_of_road_penalty=out_of_road_penalty,
            crash_vehicle_penalty=crash_vehicle_penalty,
            crash_object_penalty=crash_object_penalty,
            use_lateral_reward=use_lateral_reward,
            lane_width=lane_width,
            max_speed=max_speed,
            route_completion_threshold=route_completion_threshold,
            name=name,
        )
        self.alpha = alpha
        self.safe_distance = safe_distance

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

        if done:
            terminal_result = self.terminal.compute(env, vehicle, action, done, info)
            return terminal_result

        driving_result = self.driving.compute(env, vehicle, action, done, info)
        speed_result = self.speed.compute(env, vehicle, action, done, info)
        risk_result = self._compute_risk_penalty(env, vehicle)

        total = driving_result + speed_result + risk_result
        return total

    def _compute_risk_penalty(self, env: Any, ego_vehicle: Any) -> RewardInfo:
        risk = 0.0
        min_distance = float('inf')

        try:
            ego_pos = np.array(ego_vehicle.position)

            for adv_vehicle in env.controlled_vehicles:
                adv_pos = np.array(adv_vehicle.position)
                distance = np.linalg.norm(ego_pos - adv_pos)
                if distance < min_distance:
                    min_distance = distance

            if min_distance < self.safe_distance:
                risk = self.alpha * (1.0 - min_distance / self.safe_distance)
        except Exception:
            pass

        return RewardInfo(
            reward=-risk,
            components={"risk_penalty": -risk},
            info={
                "risk_value": risk,
                "min_adv_distance": min_distance if min_distance != float('inf') else -1,
            },
        )