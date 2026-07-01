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

