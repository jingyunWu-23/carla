# CARLA 主车路线感知与静态障碍避让底层控制设计

## 1. 问题背景

当前场景中，主车可能沿右转车道继续行驶，而全局路线实际上要求直行。右转车道在转弯位置附近可能出现路灯、交通标志杆等静态物体，最终导致主车碰撞。

该问题不应仅被定义为：

> 主车检测到路灯后进行紧急避障。

更合理的定义是：

> 主车当前车道的未来走向与全局路线不一致，因此应提前换入左侧直行车道；静态障碍检测只作为补充和紧急兜底。

因此，底层控制需要从“反应式静态障碍避让”升级为“路线一致性优先的预防式换道控制”。

---

## 2. 控制目标

底层控制应实现以下目标：

1. 提前判断当前车道未来是否偏离全局路线。
2. 当当前车道将右转、但全局路线要求直行时，提前选择左侧直行车道。
3. 在换道前检查左侧车道的动态车辆安全性和静态障碍风险。
4. 平滑执行换道，避免激进转向和频繁取消。
5. 换道完成后对齐直行路线并及时释放底层接管。
6. 当左侧车道暂时不安全时，通过减速等待间隙，而不是继续高速冲向障碍。
7. 只有在已无法安全完成换道时，才执行紧急刹车。

---

## 3. 推荐控制优先级

建议底层控制按照以下优先级工作：

```text
全局路线拓扑检查
→ 当前车道是否支持目标路线
→ 若不支持，提前规划路线纠偏换道
→ 检查目标车道动态安全和静态障碍风险
→ 执行平滑换道
→ 对齐目标车道中心线和全局路线
→ 释放底层接管
→ 若无法安全换道，再执行障碍减速或停车
```

不建议继续采用以下逻辑作为主要机制：

```text
检测到路灯
→ 接近障碍
→ 临时寻找逃逸车道
→ 距离过近时换道或刹车
```

---

## 4. 路线一致性检测

### 4.1 当前车道未来航向

通过 CARLA waypoint 获取当前车道未来一段距离后的航向：

```python
current_wp = carla_map.get_waypoint(
    ego_location,
    project_to_road=True,
    lane_type=carla.LaneType.Driving,
)

current_future_wp = get_future_waypoint(
    current_wp,
    distance=25.0,
)
```

### 4.2 全局路线参考航向

从全局 route 中获取主车前方参考 waypoint：

```python
route_wp = get_route_reference_waypoint(
    ego_location,
    lookahead=25.0,
)
```

### 4.3 航向误差

```python
current_route_error = abs(
    normalize_angle_deg(
        current_future_wp.transform.rotation.yaw
        - route_wp.transform.rotation.yaw
    )
)
```

当误差超过阈值时，认为当前车道未来走向可能不符合全局路线：

```python
current_lane_mismatch = (
    current_route_error > route_lane_mismatch_threshold_deg
)
```

建议初始参数：

```yaml
ego_route_future_heading_lookahead: 25.0
ego_route_lane_mismatch_threshold_deg: 15.0
```

---

## 5. 左侧直行车道判断

### 5.1 获取左侧车道

```python
left_wp = current_wp.get_left_lane()
```

左侧车道必须满足：

```python
left_valid = (
    left_wp is not None
    and left_wp.lane_type == carla.LaneType.Driving
)
```

还应检查：

- 左侧车道方向与当前道路方向一致；
- 左侧车道不是对向车道；
- 左侧车道不是路肩、停车带或非机动车道；
- 左侧车道能够继续连接到当前全局路线。

### 5.2 左侧车道路线误差

```python
left_future_wp = get_future_waypoint(
    left_wp,
    distance=25.0,
)

left_route_error = abs(
    normalize_angle_deg(
        left_future_wp.transform.rotation.yaw
        - route_wp.transform.rotation.yaw
    )
)
```

只有当左侧车道明显比当前车道更符合全局路线时，才触发路线纠偏：

```python
route_mismatch = (
    current_route_error > 15.0
    and left_route_error + 10.0 < current_route_error
)
```

建议参数：

```yaml
ego_route_lane_improvement_margin_deg: 10.0
```

---

## 6. 提前换道触发距离

当前静态障碍检测距离为 20 m、接管距离为 18 m，对 15～18 m/s 的主车通常过短。

路线纠偏应使用独立于静态障碍检测的前视距离：

```yaml
ego_route_lane_check_distance: 40.0
ego_route_lane_change_trigger_distance: 35.0
ego_route_lane_change_urgent_distance: 20.0
```

推荐逻辑：

```text
距离车道分流点约 35～40 m
→ 判断当前车道未来右转
→ 全局路线要求直行
→ 提前请求向左换道
```

静态障碍检测仍可保留，但不应成为该场景的主要换道触发器。

---

## 7. 独立路线纠偏状态机

建议不要把路线纠偏完全混入静态障碍 takeover 状态机，而是增加独立状态：

```text
IDLE
ROUTE_MISMATCH_DETECTED
WAITING_FOR_SAFE_GAP
ROUTE_LANE_CHANGING
ALIGNING_TO_ROUTE
ROUTE_RECOVERED
EMERGENCY_STOP
```

### 7.1 IDLE

- 正常执行高层策略动作。
- 持续检查当前车道未来方向与全局路线是否一致。

### 7.2 ROUTE_MISMATCH_DETECTED

触发条件：

- 当前车道未来航向明显偏离全局 route；
- 左侧车道更符合全局 route；
- 左侧车道为合法 Driving lane。

处理：

- 将左侧车道设为候选目标；
- 检查动态安全性和静态障碍风险。

### 7.3 WAITING_FOR_SAFE_GAP

当左侧车道暂时不安全时：

- 不立即放弃路线纠偏；
- 适当降低主车速度；
- 持续寻找左侧前后车辆间隙；
- 避免继续高速接近右转弯道和路灯。

### 7.4 ROUTE_LANE_CHANGING

一旦开始换道：

- 暂时忽略策略的横向动作；
- 固定左侧目标车道；
- 跟踪平滑换道轨迹；
- 普通风险波动不得立即取消换道；
- 只有真实紧急风险才允许取消。

### 7.5 ALIGNING_TO_ROUTE

车辆进入左侧车道后：

- 继续跟踪左侧直行车道中心线；
- 恢复与 route heading 的一致性；
- 等待横向误差和航向误差稳定。

建议完成条件：

```python
aligned = (
    abs(lateral_error) < 0.4
    and abs(route_heading_error_deg) < 5.0
)
```

### 7.6 ROUTE_RECOVERED

- 连续若干步保持路线对齐；
- 清除路线纠偏接管状态；
- 将横向控制重新交给高层策略；
- 逐步恢复巡航速度。

### 7.7 EMERGENCY_STOP

当剩余距离不足以安全换道，或者左侧出现紧急车辆冲突时：

- 放弃强行换道；
- 进入受控减速或停车。

---

## 8. 目标车道选择

目标车道不应只根据静态障碍风险选择，而应综合考虑：

```python
score = (
    w_route * route_alignment_score
    + w_clearance * obstacle_clearance_score
    + w_gap * dynamic_gap_score
    - w_lane_change * lane_change_cost
)
```

推荐初始权重：

```yaml
ego_lane_score_route_weight: 3.0
ego_lane_score_clearance_weight: 2.0
ego_lane_score_dynamic_gap_weight: 2.0
ego_lane_score_change_cost_weight: 0.5
```

---

## 9. 动态换道安全判断

目标车道至少应检查：

- 前车距离；
- 后车距离；
- 前向 TTC；
- 后向 TTC；
- 目标车道静态障碍风险；
- 目标车道是否与全局 route 连通。

伪代码：

```python
left_dynamic_safe = (
    left_front_gap > min_front_gap
    and left_rear_gap > min_rear_gap
    and left_front_ttc > min_front_ttc
    and left_rear_ttc > min_rear_ttc
)
```

建议不要使用单帧判断，应增加连续确认：

```python
if left_dynamic_safe:
    safe_steps += 1
else:
    safe_steps = 0

safe_confirmed = safe_steps >= 3
```

建议参数：

```yaml
ego_lane_change_safe_confirm_steps: 3
ego_lane_change_min_front_gap: 12.0
ego_lane_change_min_rear_gap: 10.0
ego_lane_change_min_front_ttc: 2.0
ego_lane_change_min_rear_ttc: 2.0
```

---

## 10. 换道取消机制

换道一旦开始，不应因为目标车道风险轻微波动就取消。

建议只在以下紧急条件下取消：

```python
cancel_lane_change = (
    target_front_ttc < 1.0
    or target_rear_ttc < 1.0
    or target_lane_obstacle_distance < 5.0
)
```

推荐参数：

```yaml
ego_route_lane_change_cancel_front_ttc: 1.0
ego_route_lane_change_cancel_rear_ttc: 1.0
ego_route_lane_change_cancel_obstacle_distance: 5.0
```

不建议仅使用以下条件直接取消：

```python
target_lane_risk > 0.25
```

因为风险值在换道过程中可能抖动，容易造成：

```text
开始换道
→ 风险轻微上升
→ 取消换道
→ 车辆卡在两车道之间
→ 接近路灯后被迫停车或碰撞
```

---

## 11. 横向控制实现

路线纠偏时，横向控制不应只是将 `lane_delta=-1` 转换成一次强转向。

推荐数据流：

```text
选择左侧目标车道
→ 生成从当前位置到左侧车道中心的平滑轨迹
→ 使用 Pure Pursuit、Stanley 或 MPC 跟踪
→ 进入左侧车道后继续跟踪直行 route
```

建议初始参数：

```yaml
ego_route_lane_change_lookahead: 12.0
ego_route_lane_change_steer_gain: 0.8
ego_route_lane_change_max_steer: 0.4
ego_route_lane_change_duration: 2.5
ego_route_lane_change_max_steer_rate: 0.08
```

与当前静态障碍接管参数相比，路线纠偏的控制应更平滑，避免路口附近出现转向过冲。

---

## 12. 纵向控制配合

### 12.1 左侧车道安全

当左侧车道安全且开始换道时，不应急刹：

```python
target_speed = clamp(
    current_target_speed,
    route_lane_change_min_speed,
    route_lane_change_target_speed,
)
```

建议：

```yaml
ego_route_lane_change_target_speed: 10.0
ego_route_lane_change_min_speed: 6.0
```

### 12.2 左侧车道暂时不安全

主车应适当减速等待间隙：

```python
target_speed = min(
    target_speed,
    route_wait_gap_speed,
)
```

建议：

```yaml
ego_route_wait_gap_speed: 5.0
```

### 12.3 判断是否还来得及换道

可以估算：

```python
remaining_time = conflict_distance / max(ego_speed, 0.1)
required_lane_change_time = 2.5
```

如果：

```python
remaining_time < required_lane_change_time
```

说明已经不具备安全换道时间，应停止强行换道并进入制动。

---

## 13. 静态障碍检测应基于预测行驶走廊

不建议只使用：

```text
障碍距离 < 20 m
且横向距离 < 1.5 m
```

更合理的方法是：

1. 沿当前车道生成未来 waypoint 序列；
2. 构造主车预测行驶走廊；
3. 判断路灯包围盒是否与走廊相交。

伪代码：

```python
future_waypoints = generate_future_lane_centerline(
    current_wp,
    horizon=35.0,
    step=2.0,
)

corridor = build_drivable_corridor(
    waypoints=future_waypoints,
    half_width=vehicle_width / 2 + safety_margin,
)

static_conflict = bbox_intersects_corridor(
    obstacle_bbox,
    corridor,
)
```

左侧目标车道也应构造对应走廊。如果当前右转车道走廊与路灯冲突，而左侧直行车道走廊无冲突，则进一步支持选择左侧车道。

---

## 14. 双触发器设计

建议保留两个独立触发器。

### 14.1 路线不匹配触发器

```python
route_mismatch = (
    current_route_error > 15.0
    and left_route_error + 10.0 < current_route_error
)
```

作用：提前发现右转车道与直行 route 不一致，并在接近路灯前发起换道。

### 14.2 静态障碍冲突触发器

```python
static_conflict = (
    obstacle_intersects_current_lane_corridor
    and obstacle_distance < dynamic_obstacle_lookahead
)
```

作用：当路线判断未能及时纠偏时，提供静态障碍兜底。

最终逻辑：

```python
force_left_lane_change = (
    route_mismatch
    or static_conflict
)
```

优先级应为：

```text
路线不匹配纠偏
> 静态障碍逃逸
> 紧急停车
```

---

## 15. 总体伪代码

```python
def route_aware_low_level_control(state):
    current_wp = get_current_waypoint(state.ego)
    route_wp = get_route_reference_waypoint(
        state.ego,
        lookahead=25.0,
    )

    current_route_error = future_heading_error(
        current_wp,
        route_wp,
        lookahead=25.0,
    )

    left_wp = current_wp.get_left_lane()
    left_valid = is_valid_left_driving_lane(
        current_wp,
        left_wp,
    )

    left_route_error = float("inf")
    if left_valid:
        left_route_error = future_heading_error(
            left_wp,
            route_wp,
            lookahead=25.0,
        )

    route_mismatch = (
        current_route_error > 15.0
        and left_route_error + 10.0 < current_route_error
    )

    current_static_risk = lane_corridor_obstacle_risk(
        current_wp,
        horizon=35.0,
    )

    left_static_risk = 1.0
    if left_valid:
        left_static_risk = lane_corridor_obstacle_risk(
            left_wp,
            horizon=35.0,
        )

    left_dynamic_safe = check_lane_change_safety(
        ego=state.ego,
        target_lane=left_wp,
        confirm_steps=3,
    )

    if route_mismatch:
        if (
            left_valid
            and left_dynamic_safe
            and left_static_risk < 0.20
        ):
            set_route_takeover_state("ROUTE_LANE_CHANGING")

            target_speed = clamp(
                state.target_speed,
                6.0,
                10.0,
            )

            return track_lane_change_trajectory(
                ego=state.ego,
                target_lane=left_wp,
                target_speed=target_speed,
                lookahead=12.0,
                steer_gain=0.8,
                max_steer=0.4,
            )

        set_route_takeover_state("WAITING_FOR_SAFE_GAP")

        target_speed = min(
            state.target_speed,
            5.0,
        )

        return keep_current_lane_with_speed(
            ego=state.ego,
            target_speed=target_speed,
        )

    if current_static_risk > 0.25:
        return obstacle_avoidance_or_stop(state)

    return normal_policy_control(state)
```

---

## 16. 建议配置参数

```yaml
# 路线一致性
ego_route_lane_check_distance: 40.0
ego_route_future_heading_lookahead: 25.0
ego_route_lane_mismatch_threshold_deg: 15.0
ego_route_lane_improvement_margin_deg: 10.0

# 路线纠偏触发
ego_route_lane_change_trigger_distance: 35.0
ego_route_lane_change_urgent_distance: 20.0

# 动态安全
ego_lane_change_safe_confirm_steps: 3
ego_lane_change_min_front_gap: 12.0
ego_lane_change_min_rear_gap: 10.0
ego_lane_change_min_front_ttc: 2.0
ego_lane_change_min_rear_ttc: 2.0

# 换道取消
ego_route_lane_change_cancel_front_ttc: 1.0
ego_route_lane_change_cancel_rear_ttc: 1.0
ego_route_lane_change_cancel_obstacle_distance: 5.0

# 横向控制
ego_route_lane_change_lookahead: 12.0
ego_route_lane_change_steer_gain: 0.8
ego_route_lane_change_max_steer: 0.4
ego_route_lane_change_duration: 2.5
ego_route_lane_change_max_steer_rate: 0.08

# 纵向控制
ego_route_lane_change_target_speed: 10.0
ego_route_lane_change_min_speed: 6.0
ego_route_wait_gap_speed: 5.0

# 目标车道评分
ego_lane_score_route_weight: 3.0
ego_lane_score_clearance_weight: 2.0
ego_lane_score_dynamic_gap_weight: 2.0
ego_lane_score_change_cost_weight: 0.5
```

以上参数仅作为初始值，最终需要结合仿真时间步、主车速度、车辆控制器响应、道路曲率和 CARLA 地图 waypoint 质量进行调参。

---

## 17. 必须增加的诊断日志

每个控制 step 建议记录：

```text
route_mismatch
current_route_error_deg
left_route_error_deg

current_lane_id
left_lane_id
selected_target_lane_id

route_takeover_state
route_takeover_reason

left_dynamic_safe
left_safe_confirm_steps
left_front_gap
left_rear_gap
left_front_ttc
left_rear_ttc

current_static_risk
left_static_risk
nearest_static_obstacle_label
nearest_static_obstacle_distance

target_speed_before_route_control
target_speed_after_route_control

policy_lane_delta
executed_lane_delta
steer
throttle
brake

lane_change_cancel_reason
brake_reason
```

推荐明确区分接管原因：

```text
ROUTE_MISMATCH
STATIC_OBSTACLE_CONFLICT
DYNAMIC_LANE_CONFLICT
EMERGENCY_STOP
```

---

## 18. 与强化学习策略的接口

高层策略仍可输出：

```text
speed_delta
lane_delta
```

但当检测到明确的路线不匹配时，底层控制应暂时覆盖策略横向动作：

```python
if route_takeover_active:
    executed_lane_delta = route_controller_lane_delta
else:
    executed_lane_delta = policy_lane_delta
```

建议向策略观测或 `info` 中返回：

```text
route_mismatch
route_takeover_active
route_takeover_state
target_lane_id
route_heading_error
```

可增加轻微接管惩罚，避免策略长期依赖底层纠偏，但该惩罚不应大到让主车拒绝必要的安全接管。

---

## 19. 最终设计原则

该场景应被定义为：

```text
当前车道未来会右转
+ 全局路线要求直行
+ 左侧车道更符合全局 route
→ 提前执行左侧路线纠偏换道
```

而不是：

```text
右转车道前方出现路灯
→ 接近后再临时避障
```

主车不是“为了躲路灯才换道”，而是“因为当前右转车道不符合直行路线，所以提前换入左侧直行车道”。静态障碍检测只承担补充验证和紧急兜底作用。
