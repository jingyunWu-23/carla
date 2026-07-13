# Ego and Adversarial Vehicle Observations

This project uses the same low-dimensional observation layout for the ego
vehicle and adversarial CAV vehicles. The observation state dimension is
25.

## Definition Files

- `envs/env.py`: defines the environment state dimension with `n_s = 25` and
  creates `LowDimObservationBuilder(state_dim=self.n_s)`.
- `envs/observation.py`: builds the actual 25-dimensional observation vector.
- `training/train.py`: passes `state_dim=env.n_s` into both MAPPO adversarial
  training and EgoPPO ego training.
- `envs/env.py`: stores adversarial observations in the environment return
  value `obs` and ego observations in `obs2`.

## Ego Observation

The ego observation is built by:

```python
self.obs2 = self.observation_builder.build_ego_observation(scenario)
```

`build_ego_observation()` calls `build_vehicle_observation()` on
`scenario.ego_vehicle`, so the ego vehicle receives one 25-dimensional vector.

During ego training, `training/train.py` reads it as:

```python
ego_state = np.asarray(env.obs2, dtype=np.float32).flatten()
```

## Adversarial CAV Observation

The adversarial CAV observations are built by:

```python
obs = self.observation_builder.build_multi_agent_observation(scenario)
```

`build_multi_agent_observation()` iterates over
`scenario.adv_cav_vehicles` and builds one 25-dimensional vector for each
adversarial vehicle. The usual shape is:

```text
(num_cav, 25)
```

During adversarial training, this returned `state` is passed into MAPPO.

## 25-D Observation Layout

The 25 values are constructed in `envs/observation.py` by
`build_vehicle_observation()`:

| Index | Content |
| --- | --- |
| 0 | `route_s / route_length` |
| 1 | `lateral_offset / lane_width` |
| 2 | `speed / 35.0` |
| 3 | `heading_error / pi` |
| 4 | normalized `lane_id` |
| 5 | `scenario.route_completion` |
| 6 | same-lane front gap |
| 7 | same-lane front relative speed |
| 8 | same-lane front TTC |
| 9 | left lane available |
| 10 | left-lane front gap |
| 11 | left-lane rear gap |
| 12 | left-lane front relative speed |
| 13 | left-lane rear relative speed |
| 14 | right lane available |
| 15 | right-lane front gap |
| 16 | right-lane rear gap |
| 17 | right-lane front relative speed |
| 18 | right-lane rear relative speed |
| 19 | front risk |
| 20 | rear risk |
| 21 | left-lane risk |
| 22 | right-lane risk |
| 23 | escape lane available |
| 24 | is surrounded |

## Notes

- Ego and adversarial CAV observations use the same 25-dimensional layout.
- If an observed vehicle is inactive or not physically active, the builder
  returns a zero vector with length 25.
- HDV observations also use the same 25 values, but are reshaped into a
  `5 x 5` matrix by `build_matrix_observation()`.

## 主车和对抗车奖励函数

主车和对抗车的奖励函数都定义在 `envs/reward.py`。

### 主车奖励函数

主车奖励类是 `CatSafetyEgoReward`，入口函数是 `compute()`：

```python
class CatSafetyEgoReward:
    def compute(self, scenario):
        ...
```

所在文件：

```text
envs/reward.py
```

主车非终止状态下的奖励公式是：

```python
reward = driving + speed - risk + lane_bonus + obstacle_bonus - stop_penalty - obstacle_penalty
```

各项含义：

- `driving`：沿 route 向前行驶的进度奖励，由 `_driving_reward()` 计算。
- `speed`：速度奖励，由 `_speed_reward()` 计算。
- `risk`：与对抗车距离相关的风险惩罚，由 `_risk_penalty()` 计算。
- `lane_bonus`：安全换道、前方堵塞改善奖励，同时包含换道成本和频繁换道惩罚，由 `_lane_safety_shaping()` 计算。
- `stop_penalty`：没有前车阻挡时低速或停车惩罚，由 `_stop_penalty()` 计算。
- `obstacle_penalty` / `obstacle_bonus`：静态障碍风险、避障和恢复奖励，由 `_obstacle_shaping()` 计算。

主车终止奖励由 `_terminal_reward()` 计算：

- 偏离道路：负奖励。
- 主车碰撞：负奖励。
- route completion 达到阈值：成功奖励。
- timeout：默认 0。

训练时，主车 PPO 实际读取的是 `info["ego_reward"]`：

```python
ego_reward = float(info.get("ego_reward", global_reward))
ego.step(ego_state, ego_action, ego_log_prob, ego_value, ego_reward, done, next_ego_state)
```

对应位置：

```text
training/train.py
```

### 对抗车奖励函数

对抗车奖励类是 `RiskFieldReward`，入口函数是 `compute_agent_rewards()`：

```python
class RiskFieldReward:
    def compute_agent_rewards(self, scenario) -> tuple:
        ...
```

所在文件：

```text
envs/reward.py
```

每个对抗车的奖励公式是：

```python
reward = (
    10.0 * shaped_delta
    + crash_reward
    + speed_match_weight * speed_match
    + interaction_bonus
    + behavior_bonus
    + diversity_reward
    - close_penalty
    - ttc_penalty
    - duplicate_penalty
)
```

各项含义：

- `shaped_delta`：风险场势能变化 shaping。
- `crash_reward`：如果主车碰撞则给对抗车正奖励；如果对抗车自己发生碰撞则惩罚；碰撞后恢复则给恢复奖励。
- `speed_match`：对抗车速度接近主车速度的奖励。
- `interaction_bonus`：交互模式覆盖奖励。
- `behavior_bonus`：诱发主车速度变化、减速或换道的奖励。
- `diversity_reward`：对抗车交互模式熵奖励，用来鼓励不同攻击模式。
- `close_penalty`：对抗车之间距离过近惩罚。
- `ttc_penalty`：对抗车之间 TTC 过小惩罚。
- `duplicate_penalty`：多个对抗车重复同一交互模式的惩罚。

### 奖励统一出口

奖励统一由 `RewardManager` 计算：

```python
class RewardManager:
    def compute(self, scenario, action) -> RewardResult:
        ego_reward, ego_info = self.ego_reward.compute(scenario)
        agents_rewards, risk_fields, risk_deltas, adv_info = self.risk_field_reward.compute_agent_rewards(scenario)
        ...
```

所在文件：

```text
envs/reward.py
```

`RewardManager` 会把奖励写入 `info`：

```python
"agents_rewards": agents_rewards,
"regional_rewards": agents_rewards,
"ego_reward": ego_reward,
```

对抗车训练时根据 `--reward-type` 选择训练奖励：

- `global_R`：所有对抗车使用同一个 `global_reward`。
- `regionalR`：使用 `info["regional_rewards"]`。
- `agents_rewards`：使用 `info["agents_rewards"]`。

选择逻辑在：

```text
training/train.py
```
