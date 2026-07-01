# CARLA 主车停车问题根因分析与最小修复方案（完整版）

---

# 1. 问题定义（Observation）

## ❗现象

在 CARLA + MAPPO 系统中出现：

- 主车在完成避让后停在路中间
- 无碰撞，但不再恢复速度
- 后续 episode 行为趋于保守或静止

---

# 2. 系统结构（简化MDP定义）

## Ego reward：

R_ego = R_progress + R_speed - R_risk + R_lane

## CAV reward：

R_adv = 10 * ΔU_risk + R_crash

---

# 3. 第一层：Reward 吸引子问题

问题：缺少“必须前进”约束

修复：
if ego_speed < 2.0:
    ego_reward -= 0.5

---

# 4. 第二层：控制器 DECELERATE 问题

问题：target_speed 被压低后无法恢复

修复：
target_speed += k * (cruise_speed - target_speed)

---

# 5. 第三层：geometry gap 误差

错误：
gap = other.x - ego.x

修复：
gap = dot(other.pos - ego.pos, ego.forward_vector)

---

# 6. 第四层：risk field dead zone

问题：
if dist > 40m: reward = 0

修复：
reward *= exp(-dist / 40)

---

# 7. 总结

停车问题 = reward + control + geometry + sparse gradient 的联合吸引子



---

# 8. 🚨 Reward Function Update (DPF-style Correction) —— 与原方案分割

## ❗原始 MAPPO adversarial reward（当前版本）

adv_reward =
  10 * risk_delta * distance_weight
  + crash_reward

risk_delta = previous_risk_potential - current_risk_potential

distance_weight = max(0.1, exp(-distance / 40.0))

ego crash: +100  
adv self crash: -40

---

## ❗存在的结构性问题

### 1. 只优化“局部风险上升”
→ 导致 adv 学习“让 ego 停止”这种稳定策略

### 2. distance_weight 单调衰减
→ 强化“贴近/维持低速围堵”，弱化动态交互

### 3. 缺乏 interaction diversity 约束
→ 容易收敛到单一场景（停车 / blocking）

---

## ✔ 修正后的 reward（DPF-style interaction shaping）

adv_reward =

    10 * (risk_delta * exp(-distance / 40.0))

    + 5 * interaction_variance

    + 3 * ego_behavior_change

    + crash_reward

---

## ✔ 新增项解释

### ① interaction_variance（交互多样性）

鼓励不同交互模式，而不是单一“逼停”

### ② ego_behavior_change（关键）

鼓励 adv 触发 ego 行为变化（速度/换道/制动切换）

---

## ✔ 修正后的系统目标

从：
maximize instantaneous risk increase

转变为：
maximize structured + diverse + dynamic interaction trajectories

