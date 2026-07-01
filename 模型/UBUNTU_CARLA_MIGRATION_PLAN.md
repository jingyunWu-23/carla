# Ubuntu 20.04 + CARLA 0.9.15 Migration Plan

本文档用于指导当前 CAV-HDV 协同演化项目迁移到 Ubuntu 20.04，并进一步迁移到 CARLA 0.9.15。核心原则是：旧 highway-env 版本保留为 baseline，新 CARLA 版本独立组织，避免把 Windows 环境中的缓存、模型、训练结果和旧依赖一起搬过去。

## 1. 迁移策略

不要直接复制整个当前文件夹。

当前工作目录中包含大量不适合迁移或提交的内容：

- `__pycache__/`
- `.idea/`, `.vs/`
- `coevolution_results/`
- `hdv_eval_results/`
- 大量 `.zip`, `.pt`, `.pth` 模型文件
- `dataset/`
- 临时脚本和绘图脚本
- Windows/IDE 相关文件

推荐做法：

1. 使用 Git 或压缩包只迁移核心源码。
2. 旧版 highway-env 代码保留，用于 baseline 和消融实验。
3. 新建 CARLA 迁移目录或新分支，使用更现代的依赖和 Gymnasium API。
4. 模型、数据集、训练结果单独备份，不放入代码仓库。

## 2. 推荐目录结构

建议迁移后保留两条线：

```text
project/
  highway_baseline/
    MARL1/
    highway_env/
    requirements-highway.txt

  carla_evolution/
    envs/
      carla_evolution_env.py
      observation.py
      action_adapter.py
      reward.py
      metrics.py
      scenario_manager.py
    agents/
      ego_ppo.py
      mappo.py
      hdv_policy.py
    training/
      train_coevolution.py
      curriculum.py
      evaluator.py
      replay.py
    configs/
      carla_0915.yaml
      coevolution.yaml
    scripts/
      run_carla_server.sh
      smoke_test_carla.py
    requirements-carla.txt
```

如果暂时不想大改目录，也至少建议：

```text
MARL1/
highway_env/
carla_evolution/
requirements-highway.txt
requirements-carla.txt
```

## 3. 需要迁移的核心源码

旧版 baseline 至少迁移：

```text
MARL1/
  MAPPO.py
  run_joint_train.py
  run_small_coevolution.py
  evaluate_hdv_baseline.py
  learn_hdv_aggressive_reward.py
  prepare_ngsim_for_irl.py
  train_hdv_aggressive.py
  configs/
  ego/
  hdv/
  single_agent/
  common/

highway_env/
```

可以暂时不迁移：

```text
MARL1/coevolution_results/
MARL1/hdv_eval_results/
MARL1/highway_dqn/
MARL1/__pycache__/
dataset/
*.zip
*.pt
*.pth
*.pkl
.idea/
.vs/
```

如果某些预训练模型必须使用，单独建立：

```text
artifacts/
  pretrained/
    mappo/
    ego/
    hdv/
  rewards/
```

并在 README 中说明下载或拷贝位置。

## 4. Ubuntu 环境选择

CARLA 新版建议：

```text
Ubuntu 20.04
CARLA 0.9.15
Python 3.8
Gymnasium
Stable-Baselines3 2.x
NumPy < 2
```

如果机器 CUDA 是 11.3.x，PyTorch wheel 标签使用：

```text
cu113
```

不是 `cu11358`。

示例：

```bash
conda create -n cav-carla python=3.8 -y
conda activate cav-carla

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
  --extra-index-url https://download.pytorch.org/whl/cu113
```

如果后续 CUDA 可升级，可再考虑 PyTorch 2.x。

## 5. 依赖文件策略

不要直接使用当前 Windows 环境的 `pip freeze`。

当前项目根目录中的 `requirements.txt` 比较旧，并且包含不适合 pip 安装的条目，例如：

```text
python==3.6.13
pytorch==1.7.0
```

推荐拆成两个文件：

### requirements-highway.txt

用于复现当前 highway-env baseline。

```text
numpy==1.19.5
pandas
matplotlib
opencv-python
pygame==2.0.0
gym==0.18.3
stable-baselines3
scipy
tqdm
```

这一路线用于复现实验，不追求现代化。

### requirements-carla.txt

用于新 CARLA 版本。

```text
numpy<2
pandas
matplotlib
scipy
tqdm
pyyaml
tensorboard
gymnasium
shimmy
stable-baselines3==2.0.0
pygame
networkx
opencv-python
carla==0.9.15
```

PyTorch 建议单独安装，不写入 requirements，避免 CUDA 版本冲突。

## 6. CARLA 安装检查

安装 CARLA 0.9.15 后，先不要接入强化学习代码，先做基础验证。

1. 启动 CARLA server：

```bash
cd /path/to/CARLA_0.9.15
./CarlaUE4.sh
```

2. 安装 Python API：

```bash
pip install carla==0.9.15
```

如果 PyPI 包不可用或版本不匹配，使用 CARLA 自带 wheel：

```bash
cd /path/to/CARLA_0.9.15/PythonAPI/carla/dist
pip install carla-0.9.15-*.whl
```

3. 跑通官方示例：

```bash
cd /path/to/CARLA_0.9.15/PythonAPI/examples
python generate_traffic.py
python manual_control.py
```

只有这一步稳定后，再开始接项目代码。

## 7. CARLA 迁移任务清单

### 7.1 新建 CARLA Gymnasium 环境

新环境直接使用 Gymnasium API：

```python
obs, info = env.reset(seed=seed)
obs, reward, terminated, truncated, info = env.step(action)
```

不要在新代码里继续使用旧接口：

```python
obs = env.reset()
obs, reward, done, info = env.step(action)
```

如果旧算法暂时需要 `done`，写 adapter 转换：

```text
done = terminated or truncated
```

### 7.2 车辆角色管理

CARLA 环境中需要明确四类车辆：

```text
ego        主车，由 EgoPPO 控制
adv_cav    对抗 CAV，由 MAPPO 控制
hdv        人类驾驶车辆，由共享 HDV policy 控制
background 可选，由 Traffic Manager 控制
```

每次 reset 时根据 stage 配置生成：

```text
num_cav
num_hdv
traffic_density
spawn_points
route
```

### 7.3 渗透率课程

继续保留当前机制：

```text
25% CAV -> 50% CAV -> 75% CAV
```

CARLA 中对应：

```text
low:  num_cav=1, num_hdv=3
mid:  num_cav=2, num_hdv=2
high: num_cav=3, num_hdv=1
```

如果扩大车辆规模：

```text
low:  num_cav=2, num_hdv=6
mid:  num_cav=4, num_hdv=4
high: num_cav=6, num_hdv=2
```

### 7.4 观测设计

第一版不要上图像观测，先使用低维状态观测：

```text
ego 位置、速度、航向角
周围车辆相对位置
周围车辆相对速度
lane id
到目标点距离
route progress
TTC 或 headway
```

这样可以最大化复用现有 PPO/MAPPO 思路。

### 7.5 动作设计

建议先保留离散动作：

```text
0 keep lane / maintain speed
1 accelerate
2 decelerate
3 lane left
4 lane right
```

由 `action_adapter.py` 转换为 CARLA 控制：

```text
throttle
brake
steer
```

换道可以先用 waypoint target 或局部规划器实现，不建议一开始直接学习连续 steer。

### 7.6 奖励函数

主车奖励：

```text
route progress reward
speed reward
collision penalty
out-of-lane penalty
unsafe headway penalty
success reward
```

对抗车奖励：

```text
risk induction reward
near-ego reward
TTC risk reward
self-collision penalty
```

HDV 奖励：

```text
safe driving reward
smoothness reward
speed keeping reward
collision penalty
frequent lane-change penalty
```

### 7.7 指标统计

必须在 `info` 中提供：

```text
crash
crash_rate
route_completion
road_completion_rate
average_speed
episode_length
success
timeout
```

这样当前的 performance evaluation、forgetting evaluation 和 replay analysis 才能迁移过去。

### 7.8 自适应切换

保留当前逻辑：

```text
mean_reward >= threshold
crash_rate <= threshold
road_completion_rate >= threshold
连续 patience 次通过
```

CARLA 中初期阈值需要重新标定，因为奖励尺度会变化。

### 7.9 遗忘评估和 replay

继续保留当前任务级 replay：

```text
replay buffer = 历史渗透率配置 + baseline metrics
```

不是 transition buffer。

CARLA 中 replay 时：

1. 取历史 stage 配置。
2. 重新生成旧渗透率 CARLA 场景。
3. 评估当前策略是否遗忘。
4. 优先回放 retention 最低的历史阶段。

## 8. 推荐迁移顺序

不要一开始就迁移完整三者协同演化。

推荐顺序：

1. Ubuntu 上跑通旧 highway-env baseline。
2. 安装 CARLA 0.9.15，跑通官方示例。
3. 写 CARLA `reset()`，只生成 ego 车。
4. 写 CARLA `step()`，实现离散动作控制 ego。
5. 接入单主车 EgoPPO smoke test。
6. 加入 HDV，先用 Traffic Manager。
7. 加入自定义 HDV 共享策略。
8. 加入 adv CAV 和 MAPPO。
9. 加入不同渗透率 stage schedule。
10. 加入自适应阶段切换。
11. 加入遗忘评估。
12. 加入最低 retention 优先 replay。
13. 做消融实验。

## 9. 第一阶段 smoke test 标准

CARLA 迁移第一阶段不追求训练效果，只追求接口稳定。

至少通过：

```text
env.reset() 能稳定生成车辆
env.step() 能连续运行 1000 steps
碰撞传感器能记录 crash
route completion 能正常增长
离散动作能控制车辆运动
episode 能正常 terminated/truncated
日志能保存 reward/crash/completion
```

训练参数建议极小：

```text
episodes = 3
max_steps = 200
eval_episodes = 2
```

## 10. 论文表述建议

可以在论文或计划书中写：

```text
为验证所提出协同演化机制在高保真仿真环境中的可迁移性，本文计划将当前基于 highway-env 的轻量化实验框架迁移至 CARLA 0.9.15。迁移过程中，保持主车策略、多对抗智能体策略、HDV 共享策略、自适应渗透率切换、遗忘评估与优先回放机制不变，仅替换底层交通仿真环境，并通过 Gymnasium 风格的环境封装实现算法模块与仿真平台的解耦。
```

## 11. 总结

最合适的迁移方式是：

```text
不要整包复制当前项目
不要继承旧 requirements
不要把模型、数据、结果和源码混在一起

保留旧 highway-env 作为 baseline
新建 CARLA 版本作为现代化实现
使用 Python 3.8 + CARLA 0.9.15 + Gymnasium
先跑通接口，再恢复三者协同演化机制
```

## 12. 当前仓库迁移实施清单

本节记录基于当前仓库结构的实际迁移步骤。当前代码主体位于：

```text
模型/
  MARL1/          # highway-env 版本训练代码，保留为 baseline
  highway_env/    # 旧轻量仿真环境，保留为 baseline
  carla_evolution/  # 新增 CARLA 迁移目标目录
```

当前功能对应关系：

```text
run_mappo.py              纯对抗多智能体训练，MAPPO 控制 adversarial CAV
run_joint_train.py        EgoPPO 主车 + MAPPO 对抗 CAV 交替联合训练
train_hdv_aggressive.py   训练激进驾驶风格 HDV
run_small_coevolution.py  ego / adv CAV / HDV 在不同渗透率下协同演化
hdv/                      HDV wrapper、reward、IRL 特征与共享策略相关代码
ego/                      主车 PPO、GAE、reward 与 joint trainer
```

### 12.1 已完成：结构整理

已新增：

```text
模型/requirements-highway.txt
模型/requirements-carla.txt
模型/carla_evolution/
  envs/
  agents/
  training/
  configs/
  scripts/
```

原则：

1. 不修改 `MARL1/` 和 `highway_env/` 的 baseline 行为。
2. 不复用旧 `requirements.txt` 作为 CARLA 环境依赖。
3. 新 CARLA 代码先通过 adapter 兼容旧算法接口，再逐步替换训练入口。

### 12.2 阶段 1：CARLA 环境 smoke test

目标：只验证 CARLA 环境接口稳定，不追求训练效果。

需要实现：

1. 在 `carla_evolution/envs/scenario_manager.py` 中接入 CARLA client/world。
2. 支持同步模式、固定步长、地图加载和 actor 清理。
3. `reset()` 先只生成 ego 车和 route。
4. `step()` 支持 5 个离散动作：保持、加速、减速、左换道、右换道。
5. collision sensor 能写入 `info["crash"]`。
6. route progress 能写入 `info["route_completion"]`。
7. 连续运行 1000 steps 不崩溃，episode 能正确 `terminated/truncated`。

验收标准：

```text
python scripts/smoke_test_carla.py
```

可以完成 reset、step、碰撞记录、路线进度和 episode 结束。

### 12.3 阶段 2：低维观测与指标对齐

目标：让 CARLA env 输出旧算法可消费的状态和 info。

需要实现：

1. `observation.py` 输出 25 维左右的低维状态，先对齐 `env.n_s=25`。
2. 观测包含位置、速度、航向、相对车距、相对速度、lane id、TTC/headway、route progress。
3. `metrics.py` 输出旧训练逻辑依赖的字段：

```text
crash
crash_rate
route_completion
road_completion_rate
average_speed
episode_length
success
timeout
```

4. `legacy_adapter.py` 将 Gymnasium API 转成旧 `done` 风格，便于逐步复用旧 trainer。

### 12.4 阶段 3：单主车 EgoPPO 接入

目标：先迁移主车训练闭环。

需要实现：

1. 新增 CARLA ego wrapper，复用 `MARL1/ego/ppo.py`、`config.py`、`reward/`。
2. 将 `CatSafetyReward` 需要的车辆状态从 CARLA actor 转为统一车辆状态对象。
3. 跑极小训练：`episodes=3`、`max_steps=200`、`eval_episodes=2`。
4. 保存 reward、crash、completion 日志。

这一阶段不接 MAPPO 和 HDV 自定义策略。

### 12.5 阶段 4：HDV 车辆接入

目标：让 HDV 在 CARLA 中稳定参与交通。

分两步：

1. 先使用 CARLA Traffic Manager 控制 HDV，验证多车场景稳定。
2. 再接入 `hdv/` 下的共享 HDV policy 和 `AggressiveHDVReward`。

需要迁移：

```text
MARL1/hdv/hdv_env_wrapper.py
MARL1/hdv/coevolution_hdv_env_wrapper.py
MARL1/hdv/reward/aggressive_hdv_reward.py
train_hdv_aggressive.py
```

### 12.6 阶段 5：MAPPO 对抗 CAV 接入

目标：让 adversarial CAV 在 CARLA 中由 MAPPO 控制。

需要实现：

1. CARLA env 中维护 `controlled_vehicles`，表示 MAPPO 控制的对抗 CAV。
2. 输出 adversarial agents 的多智能体 observation，shape 对齐旧 `MAPPO.py`。
3. `reward.py` 实现对抗奖励：risk induction、near-ego、TTC risk、self-collision penalty。
4. `info` 中提供：

```text
agents_rewards
regional_rewards
agents_dones
agents_info
```

5. 优先通过 adapter 复用旧 `MAPPO.py`，只在必要时局部改造算法代码。

### 12.7 阶段 6：迁移联合训练 run_joint_train

目标：恢复 EgoPPO + MAPPO 交替联合训练。

需要实现：

1. 新建 CARLA 版训练入口，例如：

```text
carla_evolution/training/train_joint.py
```

2. 保留原训练逻辑：

```text
adversarial phase -> ego phase -> repeat
```

3. 替换环境创建：

```text
gym.make('merge-multi-agent-v0')
```

为 CARLA env factory。

4. CARLA 中重新标定：

```text
reward scale
crash threshold
completion threshold
phase switching threshold
```

### 12.8 阶段 7：迁移三类车辆协同演化 run_small_coevolution

目标：恢复不同 CAV 渗透率下的协同演化。

保留当前课程机制：

```text
25% CAV: num_cav=1, num_hdv=3
50% CAV: num_cav=2, num_hdv=2
75% CAV: num_cav=3, num_hdv=1
```

需要实现：

1. `curriculum.py` 管理渗透率 stage。
2. `scenario_manager.py` 根据 stage reset CARLA 场景。
3. `replay.py` 保存历史 stage 配置和 baseline metrics。
4. `evaluator.py` 支持当前 stage 和历史 stage 评估。

### 12.9 阶段 8：自适应切换、遗忘评估和 replay

目标：恢复论文机制，而不是只跑固定课程。

需要迁移并重新标定：

```text
adaptive-stage-switch
performance-reward-threshold
performance-crash-threshold
performance-completion-threshold
performance-patience
forget-retention-threshold
forget-crash-increase-threshold
forget-completion-drop-threshold
lowest-retention replay
```

CARLA 版本中 replay 仍然是任务级 replay：回放历史渗透率场景，而不是 transition replay buffer。

### 12.10 阶段 9：实验组织

建议最终实验分组：

1. highway-env baseline。
2. CARLA ego only。
3. CARLA ego + HDV。
4. CARLA ego + adversarial CAV。
5. CARLA ego + adversarial CAV + HDV co-evolution。
6. 去掉 adaptive stage switch 的消融。
7. 去掉 forgetting replay 的消融。
8. 不同 CAV 渗透率对比。

每组至少记录：

```text
mean reward
crash rate
route completion
average speed
success rate
episode length
forgetting retention
```

### 12.11 HDV PPO 训练策略修订

当前三类车辆均沿用 PPO/MAPPO 训练路线：

```text
ego      主车策略，EgoPPO
adv CAV  对抗 CAV，多智能体 MAPPO，本质仍是 PPO 类方法
HDV      人类驾驶车辆，共享 PPO policy
```

旧 highway-env 版本中的 `PPOmodel889` 只能作为 baseline 参考，不建议直接在 CARLA 中继续训练，原因是：

1. 观测空间变了：旧模型输入来自 highway-env 的低维观测，CARLA 的位置、速度、车道、航向和相对车辆编码需要重新定义。
2. 动作语义变了：旧离散动作由 highway-env `DiscreteMetaAction` 执行，CARLA 需要经过 throttle、brake、steer、waypoint/lane-change adapter。
3. 状态分布变了：CARLA 的动力学、碰撞、车道几何、仿真步长和交通行为都不同，旧策略即使维度对齐也会产生严重分布偏移。
4. reward 标定变了：激进/类人风格 reward 在 CARLA 中需要重新计算 headway、TTC、smoothness、lane-change penalty 等指标。

因此 HDV 建议采用两阶段训练：

```text
阶段 A：中性 HDV PPO
  目标：先学会在 CARLA 中稳定、守规、低碰撞行驶。
  数据/行为来源：Traffic Manager、规则控制器或人工设计的中性 reward。
  输出：neutral_hdv_ppo.zip

阶段 B：类人/激进 HDV PPO 微调
  目标：在中性模型基础上微调出目标驾驶风格。
  初始化：优先从 neutral_hdv_ppo.zip 开始，而不是从 highway-env 的 PPOmodel889 开始。
  reward：迁移并重标定 AggressiveHDVReward。
  输出：aggressive_hdv_ppo.zip
```

如果 CARLA 中性模型训练不稳定，再考虑从头训练激进 HDV；但默认路线应是“CARLA 中性 PPO -> CARLA 风格微调”，因为它比完全从头训练更稳，也比复用 `PPOmodel889` 更符合新环境分布。

代码层面的约束：

1. 不把 `PPOmodel889` 写死在新训练代码中。
2. 新训练入口只接收可配置的 `--hdv-model`。
3. 如果未提供 `--hdv-model`，先使用 Traffic Manager 或规则策略作为 HDV 行为基线。
4. 待中性 PPO 训练完成后，再把它作为 aggressive HDV 微调的初始化模型。

### 12.12 当前实现进度

已完成第一阶段中的接口落地工作：

```text
carla_evolution/MODULE_DEPENDENCIES.md
carla_evolution/envs/vehicle_state.py
carla_evolution/envs/scenario_manager.py
carla_evolution/envs/observation.py
carla_evolution/envs/reward.py
carla_evolution/envs/metrics.py
carla_evolution/envs/env.py
carla_evolution/envs/legacy_adapter.py
carla_evolution/envs/factory.py
```

当前实现内容：

1. 梳理了 `run_joint_train.py`、`MAPPO.py`、`EgoEnvWrapper`、`HDVEnvWrapper` 对环境的接口依赖。
2. 新增 `VehicleState`，统一车辆状态字段，后续真实仿真 actor 需要先转换成这个状态对象。
3. 新增 mock backend，可生成 ego、adv CAV、HDV 三类车辆并按离散动作推进。
4. 新增低维观测，当前输出维度保持 `n_s=25`，用于兼容旧 PPO/MAPPO 网络输入。
5. 新增 reward 和 metrics，提供 `agents_rewards`、`regional_rewards`、`ego_reward`、`hdv_reward`、`crash`、`route_completion`、`average_speed` 等旧代码依赖字段。
6. 新增 `LegacyTrainingAdapter`，提供旧训练代码使用的返回格式：

```text
reset() -> env_state, action_mask, obs2, obs3
step()  -> env_state, reward, done, info, obs2, obs3
```

已验证：

```bash
PYTHONPATH=模型 python3 模型/carla_evolution/scripts/smoke_test.py
PYTHONPATH=模型 python3 -m compileall -q 模型/carla_evolution
```

下一步实现目标：

1. 将 `ScenarioManager` 的 runtime backend 从仅连接 client/world 扩展到真实 actor spawn。
2. 实现 actor 清理、同步模式、固定步长、地图加载。
3. 将 CARLA actor 状态同步为 `VehicleState`。
4. 用真实 collision sensor 和 route progress 替换 mock 指标。
5. 在真实 backend 通过后，再接入单主车 EgoPPO smoke training。

### 12.13 真实 backend 实现进度

已在 `carla_evolution/envs/scenario_manager.py` 中实现真实 backend 的第一版最小闭环：

1. 连接 CARLA client/world。
2. 支持 `synchronous_mode` 和 `fixed_delta_seconds`。
3. reset 时清理上一轮 actors/sensors。
4. 根据 `num_cav`、`num_hdv`、`num_background` spawn 三类车辆和 ego 车。
5. 为每辆车挂载 `sensor.other.collision`。
6. step 时将离散动作转换后的 control intent 应用为 `carla.VehicleControl`。
7. 每步执行 `world.tick()`。
8. 将 CARLA actor 的 location、velocity、heading、lane 信息同步成 `VehicleState`。
9. 使用 collision sensor 更新 crash/crash_rate。
10. 使用 ego 起点到虚拟 route end 的投影计算 `route_completion`。
11. close/reset 时销毁 sensors 和 actors，并在 close 时恢复原 world settings。

配置文件已更新：

```text
carla_evolution/configs/carla_0915.yaml
```

无 CARLA 环境下仍使用 mock backend 验证接口：

```bash
PYTHONPATH=模型 python3 模型/carla_evolution/scripts/smoke_test.py --backend mock --steps 10
```

真实 CARLA server 启动后，可用以下命令验证 runtime backend：

```bash
cd /path/to/CARLA_0.9.15
./CarlaUE4.sh

cd /home/chenyuanwan/download/co-training/code-migration
PYTHONPATH=模型 python3 模型/carla_evolution/scripts/smoke_test.py   --backend carla   --config 模型/carla_evolution/configs/carla_0915.yaml   --steps 100
```

当前限制：

1. 换道动作暂时用固定 steer 实现，不是 waypoint/local planner 换道。
2. route 仍是从 ego spawn 点沿初始航向构造的直线虚拟 route，还不是 CARLA GlobalRoutePlanner 路线。
3. HDV 和 adv CAV 当前由同一离散动作接口控制，尚未接入 Traffic Manager 或各自 PPO 策略。
4. collision sensor 已接入，但真实碰撞回调需要在 CARLA server 中实测。
5. 阶段 3 EgoPPO 训练还未开始，应等待真实 backend smoke test 通过后再接入。

### 12.14 真实 backend smoke test 修正记录

真实 CARLA smoke test 暴露并修正了两个基础接口问题：

1. `truncated=True` 时 `info["timeout"]` 仍为 `False`。
   - 原因：metrics 生成早于 `scenario.timeout` 更新。
   - 修正：先计算 `terminated/truncated` 并更新 `scenario.timeout`，再生成 metrics。

2. reset 后 `route_completion` 初始值约为 0.44。
   - 原因：route 起点在 actor spawn 后、第一次 `world.tick()` 前保存；CARLA actor 的稳定位置需要 tick 后读取。
   - 修正：将 route reference 保存移动到 `tick_after_spawn` 之后，并保存数值型 route start/vector。

修正后需要在真实 CARLA 中重新验证：

```bash
PYTHONPATH=模型 python3 模型/carla_evolution/scripts/smoke_test.py   --backend carla   --config 模型/carla_evolution/configs/carla_0915.yaml   --steps 100
```

重点检查：

```text
reset info route_completion 应接近 0.0
step info timeout 应在 truncated=True 时为 True
route_completion 应随 ego 运动平滑增长
```

### 12.15 预训练顺序调整

根据当前真实 backend 的验证结果，后续训练顺序调整为：

```text
阶段 A：单主车控制 smoke test
  目标：确认离散动作在 CARLA 中能稳定控制 ego。
  场景：num_cav=0, num_hdv=0, num_background=0。

阶段 B：ego + adv CAV 结构预训练
  目标：先训练主车和多对抗智能体组成的结构。
  场景：ego + num_cav 个 adversarial CAV。
  暂不生成训练型 HDV。

阶段 C：HDV 作为简单 manager/规则车辆加入
  目标：只增加交通扰动，不训练 HDV PPO。
  HDV 行为来源：Traffic Manager 或规则控制。

阶段 D：训练中性 HDV PPO
  目标：在 CARLA 中训练稳定、守规的 HDV。

阶段 E：从中性 HDV PPO 微调类人/激进 HDV
  目标：替代旧 highway-env 的 PPOmodel889 路线。
```

新增单主车控制验证脚本：

```text
carla_evolution/scripts/control_smoke_test.py
```

mock backend 验证：

```bash
PYTHONPATH=模型 python3 模型/carla_evolution/scripts/control_smoke_test.py   --backend mock   --action accelerate   --steps 100
```

真实 CARLA backend 验证：

```bash
PYTHONPATH=模型 python3 模型/carla_evolution/scripts/control_smoke_test.py   --backend carla   --config 模型/carla_evolution/configs/carla_0915.yaml   --action accelerate   --steps 100
```

验收标准：

```text
reset route_completion = 0.0
ego speed 随 accelerate 明显上升
route_completion 从 0 开始增长
crash = False
on_road = True
```

### 12.16 单主车真实控制测试结果

真实 CARLA 中 `accelerate` 动作已验证链路可用：

```text
reset route_completion = 0.0
num_vehicles = 1
initial speed ~= 0.49 m/s
step_100 speed ~= 4.91 m/s
step_100 route_completion ~= 0.0268
crash = False
timeout = True
on_road = True
```

结论：

1. 单主车 spawn、step、VehicleState 同步、route progress 和 timeout 均正常。
2. `accelerate` 能让车辆前进，但原始 throttle=0.6 偏保守，5 秒后速度仍较低。
3. 已将离散动作控制参数配置化，并在 `carla_0915.yaml` 中提高默认加速油门。

新增配置段：

```yaml
control:
  keep_lane_throttle: 0.45
  accelerate_throttle: 1.0
  decelerate_brake: 0.55
  lane_change_throttle: 0.45
  lane_change_steer: 0.35
```

下一次真实测试建议先重复：

```bash
PYTHONPATH=模型 python3 模型/carla_evolution/scripts/control_smoke_test.py   --backend carla   --config 模型/carla_evolution/configs/carla_0915.yaml   --action accelerate   --steps 100
```

如果速度和 route progress 明显改善，再分别测试：

```text
keep_lane
decelerate
lane_left
lane_right
```

### 12.17 ego + adv CAV 多车控制 smoke test

进入阶段 B 前，先验证 ego + 多对抗智能体结构，不生成 HDV：

```text
num_cav >= 1
num_hdv = 0
num_background = 0
动作顺序 = [adv_cav_0, adv_cav_1, ..., ego]
```

新增验证脚本：

```text
carla_evolution/scripts/adv_smoke_test.py
```

mock backend 验证：

```bash
PYTHONPATH=模型 python3 模型/carla_evolution/scripts/adv_smoke_test.py   --backend mock   --num-cav 3   --adv-action keep_lane   --ego-action accelerate   --steps 100
```

真实 CARLA backend 验证：

```bash
PYTHONPATH=模型 python3 模型/carla_evolution/scripts/adv_smoke_test.py   --backend carla   --config 模型/carla_evolution/configs/carla_0915.yaml   --num-cav 3   --adv-action keep_lane   --ego-action accelerate   --steps 100
```

验收标准：

```text
reset obs shape = (num_cav, 25)
reset num_vehicles = num_cav + 1
HDV 不生成
各 adv CAV 和 ego 均有 VehicleState
info 中存在 agents_rewards / regional_rewards / agents_dones
route_completion 从 0 开始增长
crash=False 或碰撞能被正确记录
```

通过后再开始迁移训练入口：

```text
carla_evolution/training/train_joint.py
```

该训练入口先只接入 ego + adv CAV，不接训练型 HDV。

### 12.18 风险场对抗奖励迁移

确认：此前 `carla_evolution/envs/reward.py` 中的对抗奖励只是 smoke-test 用的 proximity reward，不是原 highway-env 的风险场奖励。

已完成修正：

1. 新增 `RiskFieldReward`。
2. 迁移 `highway_env/envs/merge_env_v1.py` 中 `weixian()` 的核心风险场公式。
3. 对每个 adv CAV 计算：

```text
risk_field
risk_delta = previous_risk_field - current_risk_field
agent_reward = 10 * shaped_risk_delta + crash_reward
```

4. 保留原碰撞奖励逻辑：

```text
ego crashed: +100
adv crashed: -40
otherwise: 0
```

5. `reset()` 时清空上一 episode 的风险场历史。
6. `adv_smoke_test.py` 输出新增诊断字段：

```text
risk_fields
risk_deltas
```

mock backend 已验证风险场奖励生效：

```text
agents_rewards != 全 0
risk_fields 有数值
risk_deltas 有数值
```

真实 CARLA 多车 smoke test 暴露的问题：

```text
3 个 adv CAV 和 ego 在同一区域、同 lane_id=2 附近生成
初始车距较近
keep_lane 下 adv 速度较低
earlier proximity reward 下 agents_rewards 全 0
```

下一步需要在继续训练前优化 spawn 策略：

1. 不再简单取连续 spawn points。
2. 选择和 ego 同路线、纵向间距可控的 adv spawn points。
3. 支持配置 adv 相对 ego 的初始距离，例如：

```text
adv_offsets: [20, 40, 60]
```

4. 如果地图 spawn points 不满足间距，使用 ego waypoint 沿 route 前后生成 transform。

### 12.19 相对路线 spawn 策略

为解决真实 CARLA 中连续 spawn points 导致 ego 和 adv CAV 初始位置混乱、距离不可控的问题，已新增 `relative_route` spawn 策略。

配置位置：

```text
carla_evolution/configs/carla_0915.yaml
```

新增配置：

```yaml
scenario:
  spawn_strategy: relative_route
  spawn_start_index: 0
  spawn_z_offset: 0.5
  adv_offsets: [20.0, 40.0, 60.0]
  hdv_offsets: [80.0, 100.0, 120.0]
```

实现逻辑：

1. 先选择 ego spawn point。
2. 获取 ego 所在 CARLA waypoint。
3. 沿 ego waypoint 前方按 `adv_offsets` 生成 adv CAV transforms。
4. 保持旧训练动作顺序不变：

```text
[adv_cav_0, adv_cav_1, ..., ego, hdv_0, ...]
```

5. 如果 `spawn_strategy` 不是 `relative_route`，仍可使用旧的 sequential spawn points 方式。

真实 CARLA 下一次多车 smoke test 需要检查：

```text
adv CAV 是否位于 ego 前方约 20/40/60m
reset obs shape 是否仍为 (num_cav, 25)
risk_fields / risk_deltas 是否有数值
agents_rewards 是否不再因距离门控长期全 0
```

### 12.20 relative-route 多车真实测试结果

真实 CARLA 中 `relative_route` spawn 已验证生效：

```text
ego:   x ~= 225.25
adv_0: x ~= 245.45  # ego 前方约 20m
adv_1: x ~= 263.83  # ego 前方约 39m
adv_2: x ~= 281.44  # ego 前方约 56m
```

同时验证：

```text
risk_fields 有数值
risk_deltas 有数值
collision sensor 生效
risk-field crash reward 生效：ego crash 后 adv rewards 约 +100
terminated=True
```

本轮也暴露了控制测试问题：

```text
adv_action=keep_lane 时，keep_lane_throttle=0.45 导致 adv 持续给油。
ego_action=accelerate 后 ego 追上 adv_0 并发生碰撞。
```

修正：

```yaml
control:
  keep_lane_throttle: 0.0
```

含义：

1. smoke test 中 `keep_lane` 表示不主动加速的保持动作。
2. 训练时策略仍可通过 `accelerate` 主动加速。
3. 后续更合理的做法是引入目标速度保持控制器，而不是固定 throttle。

下一轮真实多车 smoke test 建议继续使用：

```bash
PYTHONPATH=模型 python3 模型/carla_evolution/scripts/adv_smoke_test.py   --backend carla   --config 模型/carla_evolution/configs/carla_0915.yaml   --num-cav 3   --adv-action keep_lane   --ego-action accelerate   --steps 100
```

期望：

```text
relative spawn 仍正确
若 ego 追尾 adv_0，collision 和 +100 奖励应继续生效
若没有碰撞，agents_rewards 应由 risk_delta 决定
```

### 12.21 MAPPO 对抗预训练入口

已开始从固定动作 smoke test 进入真实策略训练接入。

新增/更新：

```text
carla_evolution/agents/mappo.py
carla_evolution/training/train.py
```

实现内容：

1. `MAPPOAgent` 复用旧 `MARL1` 中的核心组件：

```text
single_agent/Model_common.py     ActorNetwork / CriticNetwork
single_agent/Memory_common.py    OnPolicyReplayMemory
single_agent/utils_common.py     tensor / one-hot 工具
```

2. 保留旧 MAPPO 的 PPO-clip 更新结构：

```text
actor / critic
target actor / target critic
advantage = discounted_reward - critic_target
ratio clipping
target network soft update
```

3. 去掉旧 `MAPPO.py` 中写死的模型依赖：

```text
PPOmodel2
PPOmodel889
```

4. `training/train.py --mode adv` 现在执行 adv CAV MAPPO 预训练：

```text
adv CAV: MAPPO 网络输出动作
ego: 固定/规则动作，用于对抗预训练环境
HDV: 不生成
reward: 风险场 agents_rewards / regional_rewards / global_R 可配置
```

运行示例：

```bash
PYTHONPATH=模型 python3 模型/carla_evolution/training/train.py   --mode adv   --backend carla   --config 模型/carla_evolution/configs/carla_0915.yaml   --num-cav 3   --episodes 10   --max-steps 200   --ego-action accelerate   --reward-type global_R
```

mock 小训练验证命令：

```bash
PYTHONPATH=模型 python3 模型/carla_evolution/training/train.py   --mode adv   --backend mock   --episodes 4   --max-steps 20   --num-cav 3   --ego-action accelerate   --batch-size 20   --episodes-before-train 1   --no-cuda
```

当前本机 shell 环境缺少 PyTorch，所以上述训练命令在当前解释器下无法执行，会提示需要安装/激活 PyTorch 环境。语法检查已通过。实际训练应在 `cav-carla` 或其他已安装 PyTorch 的环境中运行。

说明：

这一步不是简化版主车/对抗车。对抗车已经由 MAPPO 网络控制；ego 暂时使用规则动作，是为了先完成旧 `run_mappo.py` 对应的对抗预训练阶段。完整 joint training 仍按后续步骤继续接入 EgoPPO。

### 12.22 EgoPPO 接入与 joint 交替训练入口

已继续接入主车策略，不再停留在固定动作结构。

新增/更新：

```text
carla_evolution/agents/ego_ppo.py
carla_evolution/training/train.py
```

实现内容：

1. `EgoPPOAdapter` 复用旧 `MARL1/ego` 模块：

```text
ego/config.py   PPOConfig
ego/ppo.py      EgoPPO
ego/model.py    ActorCriticNetwork
ego/memory.py   RolloutBuffer
```

2. `train.py` 新增：

```text
--mode joint
```

3. joint 训练当前覆盖 ego + adv CAV，不生成训练型 HDV：

```text
adv phase:
  adv CAV: MAPPO exploration_action，训练 MAPPO
  ego: 当前 EgoPPO deterministic action，冻结推理不更新

 ego phase:
  adv CAV: MAPPO deterministic action，冻结推理不更新
  ego: EgoPPO select_action，收集 rollout 并 update

HDV:
  当前不生成
```

4. 日志输出：

```text
joint_train_log.csv
round
phase
episode
steps
episode_reward
mean_agent_reward
route_completion
crash
timeout
actor_loss / critic_loss / policy_loss / value_loss
```

运行示例：

```bash
PYTHONPATH=模型 python3 模型/carla_evolution/training/train.py   --mode joint   --backend carla   --config 模型/carla_evolution/configs/carla_0915.yaml   --num-cav 3   --rounds 3   --adv-episodes 2   --ego-episodes 2   --max-steps 200   --reward-type global_R
```

mock 结构验证示例：

```bash
PYTHONPATH=模型 python3 模型/carla_evolution/training/train.py   --mode joint   --backend mock   --num-cav 3   --rounds 1   --adv-episodes 1   --ego-episodes 1   --max-steps 20   --batch-size 20   --ego-batch-size 8   --ego-n-steps 64   --no-cuda
```

当前限制：

1. 本 shell 环境缺少 PyTorch，因此只能完成语法检查，不能实际执行训练。
2. joint 当前未接 HDV，符合当前阶段“先主车 + 对抗 CAV”的顺序。
3. 尚未实现旧 `run_joint_train.py` 中的 adaptive switching、forgetting evaluation、replay；后续在 joint 基础稳定后继续迁移。
4. ego phase 的 adv 使用 MAPPO deterministic action 作为冻结策略；后续可改为加载指定 checkpoint。

### 12.23 joint 训练 readiness 检查

已检查并修正 joint 训练结构：

1. `EgoPPOAdapter` 已接入旧 `MARL1/ego/EgoPPO`。
2. `MAPPOAgent` 已接入旧 `MARL1` 的 actor/critic/memory/PPO 更新逻辑。
3. `train.py --mode joint` 已实现交替训练。
4. adv phase 已修正为：

```text
adv CAV: MAPPO exploration_action，训练 MAPPO
ego: 当前 EgoPPO deterministic action，冻结推理不更新
```

5. ego phase 为：

```text
adv CAV: MAPPO deterministic action，冻结推理不更新
ego: EgoPPO select_action，收集 rollout 并 update
```

6. 动作顺序保持：

```text
[adv_cav_0, adv_cav_1, ..., ego]
```

7. 语法检查通过：

```bash
PYTHONPATH=模型 python3 -m compileall -q 模型/carla_evolution
```

当前运行前提：

```text
必须在安装 PyTorch 的训练环境中运行。
当前普通 shell 缺少 torch，无法实际执行训练。
```

建议先用 mock backend 做最小 joint 结构验证，再切换真实 CARLA：

```bash
PYTHONPATH=模型 python3 模型/carla_evolution/training/train.py   --mode joint   --backend mock   --num-cav 3   --rounds 1   --adv-episodes 1   --ego-episodes 1   --max-steps 20   --batch-size 20   --ego-batch-size 8   --ego-n-steps 64   --no-cuda
```



### 12.24 近期 CARLA 训练稳定性与行为改进记录

本节记录在真实 CARLA 训练反馈后，已经落地到 `carla_evolution` 代码中的修改。以下内容均以当前实际代码为准，未实现的 HDV 迁移、Adv DPF 扩展观测、Adv reward curriculum 不在本节标记为完成。

#### 12.24.1 Ego warmup 训练阶段

已在 `模型/carla_evolution/training/train.py` 中加入主车预训练阶段：

```text
--ego-warmup
--ego-warmup-min-episodes
--ego-warmup-max-episodes
--ego-warmup-target-crash-rate
--ego-warmup-target-completion
--ego-warmup-window
--warmup-adv-action
```

训练逻辑：

```text
warmup 阶段：
  adv CAV 使用规则动作，不更新 MAPPO
  ego 使用 EgoPPO 采样、收集 rollout、更新

达到目标：
  最近窗口 ego_crash_rate <= 指定阈值
  且 road_completion >= 指定阈值

之后进入 joint/adaptive 对抗训练。
```

推荐初始阈值来自旧 `MARL1/joint_results/joint_train_log.csv` 的起始主车表现：

```text
ego_warmup_target_crash_rate = 0.45
ego_warmup_target_completion = 0.34
ego_warmup_window = 100
```

#### 12.24.2 动作编号与高层动作适配器

已将 `模型/carla_evolution/envs/action_adapter.py` 的动作编号对齐 highway-env 旧语义：

```text
0: LANE_LEFT
1: KEEP_LANE / IDLE
2: LANE_RIGHT
3: ACCELERATE / FASTER
4: DECELERATE / SLOWER
```

同时 `ActionAdapter` 不再只表达单步 `steer/throttle`，而是输出高层意图字段：

```text
action
speed_delta
lane_delta
throttle / brake / steer fallback
```

#### 12.24.3 CARLA backend 高层 target_speed / target_lane 控制

已在 `模型/carla_evolution/envs/scenario_manager.py` 的 CARLA runtime 分支中加入 per-actor 控制状态：

```text
target_speed
target_lane_id
lane_change_intent
lane_change_timer
lane_change_success
lane_change_failed
lane_change_cooldown
previous_action
```

CARLA 执行逻辑从：

```text
离散动作 -> 单步 steer/throttle
```

升级为：

```text
离散动作 -> target speed / target lane -> waypoint lateral control + speed control
```

相关配置已写入 `模型/carla_evolution/configs/carla_0915.yaml`：

```yaml
use_high_level_intent: true
max_lane_change_steps: 25
lane_change_cooldown_steps: 20
failed_lane_change_cooldown: 10
lookahead_distance: 8.0
target_speed_min: 2.0
target_speed_max: 28.0
target_speed_cruise: 18.0
delta_v: 2.0
lateral_kp: 0.12
heading_kp: 0.8
speed_kp: 0.08
brake_kp: 0.12
max_steer: 0.45
```

#### 12.24.4 变道稳定性约束

针对训练后期主车频繁跨车道乱开的问题，已加入两类约束。

控制层约束：

```text
1. 变道成功后进入 cooldown。
2. 变道失败后进入较短 cooldown。
3. cooldown 未结束时拒绝新的 LANE_LEFT / LANE_RIGHT。
4. Ego 只有当前车道前方受阻，或目标车道明显更安全时才允许变道。
```

Ego 变道触发配置：

```yaml
ego_min_lanechange_gap_front: 15.0
ego_min_lanechange_gap_rear: 10.0
ego_lanechange_trigger_gap: 30.0
ego_lanechange_min_gap_gain: 8.0
```

奖励层约束已加入 `模型/carla_evolution/envs/reward.py`：

```text
lane_change_cost = 0.10
oscillation_penalty = 0.35
oscillation_window = 30
```

现在只有“前方确实受阻且变道后前方空间明显更大”的变道才会得到小奖励；无意义变道和短时间反复变道会受到惩罚。

#### 12.24.5 25 维结构化观测与风险摘要

当前仍保持：

```text
state_dim = 25
```

未修改 EgoPPO/MAPPO 网络输入维度，避免必须重写网络结构。

已在 `模型/carla_evolution/envs/observation.py` 中将原“最近多车拼接”观测改为：

```text
自身状态
当前车道前方 gap / dv / TTC
左车道可用性、前后 gap、前后相对速度
右车道可用性、前后 gap、前后相对速度
风险摘要：front_risk / rear_risk / left_risk / right_risk
escape_lane_available
is_surrounded
```

25 维末尾已加入风险摘要字段，目的是让主车能显式判断：

```text
前方是否堵塞
左右车道是否安全
是否存在可逃逸车道
是否处于多车合围风险中
```

#### 12.24.6 主车奖励函数调整

`模型/carla_evolution/envs/reward.py` 中的 `CatSafetyEgoReward` 已保留旧版 `CatSafetyReward` 主体：

```text
R_ego = R_driving + R_speed - R_risk + R_terminal
```

并加入 CARLA 必需的轻量 shaping：

```text
前方近距离车辆惩罚
安全变道小奖励
变道成本
短时间反复变道惩罚
```

当前形式：

```text
R_ego = R_driving
      + R_speed
      - R_distance_risk
      + R_terminal
      + R_lane_shaping
```

其中 `R_lane_shaping` 包含：

```text
front_gap_penalty
safe_lane_bonus
lane_change_cost
oscillation_penalty
```

#### 12.24.7 对抗车风险场奖励现状

对抗车仍使用 `模型/carla_evolution/envs/reward.py` 中的 `RiskFieldReward`，迁移自旧 highway-env 风险场思想：

```text
agents_rewards
regional_rewards
risk_fields
risk_deltas
```

当前已实现：

```text
风险势能计算
风险势能变化 shaping
主车碰撞奖励
对抗车自身碰撞惩罚
距离过远时削弱 shaping
```

尚未实现：

```text
Adv DPF 扩展观测
critical interaction reward
surround contribution reward
ego collision reward curriculum
attack anchor 特征
```

这些内容保留为后续阶段。

#### 12.24.8 规则背景车加入

为了缓解主车和对抗车速度不匹配、前车静止或过慢造成训练不稳定的问题，已加入 `ScenarioManager` 控制的背景车。

训练入口 `模型/carla_evolution/training/train.py` 新增：

```text
--num-background
```

默认：

```text
num_background = 2
```

CARLA 配置：

```yaml
num_background: 2
background_offsets: [45.0, 75.0]
background_lane_offsets: [0, 3]
```

背景车特性：

```text
1. 不参与 MAPPO/EgoPPO 学习。
2. 不占用策略动作输出位置。
3. 由 ScenarioManager 使用默认高层 keep-lane / target-speed 控制。
4. 使用 target_speed_cruise 与其他车辆保持大致一致速度。
```

当前默认动作顺序仍保持：

```text
[adv_cav_0, adv_cav_1, ..., ego]
```

背景车和后续 HDV 不在当前策略动作列表中。

#### 12.24.9 CARLA 长时间训练稳定性修复

针对训练约 200 episode 后出现 CARLA timeout 的问题，已做以下处理：

```yaml
timeout: 60.0
tick_retries: 3
```

`ScenarioManager` 中已将直接 `world.tick()` 替换为 `_tick_runtime()`，并在 cleanup 后补 tick，减少 actor/sensor 销毁积压。

仍需注意：

```text
CARLA Vulkan 显存耗尽仍可能导致 Unreal 崩溃。
长时间训练建议使用低画质或 offscreen。
必要时后续实现分段重连 CARLA server。
```

#### 12.24.10 当前建议训练命令

建议从新的 warmup 流程重新训练，不复用旧 checkpoint：

```bash
cd /home/chenyuanwan/download/co-training/code-migration

PYTHONPATH=模型 /home/chenyuanwan/anaconda3/envs/cav-carla/bin/python \
  模型/carla_evolution/training/train.py \
  --mode joint \
  --backend carla \
  --config 模型/carla_evolution/configs/carla_0915.yaml \
  --ego-warmup \
  --ego-warmup-min-episodes 300 \
  --ego-warmup-max-episodes 3000 \
  --ego-warmup-target-crash-rate 0.45 \
  --ego-warmup-target-completion 0.34 \
  --ego-warmup-window 100 \
  --warmup-adv-action keep_lane \
  --joint-strategy adaptive \
  --rounds 100 \
  --num-cav 3 \
  --num-background 2 \
  --max-steps 200 \
  --reward-type global_R \
  --eval-interval 10 \
  --save-interval 50
```

如果背景车导致场景过密，可临时改为：

```bash
--num-background 1
```

#### 12.24.11 已验证内容

已完成以下静态和 mock 验证：

```text
python3 -m py_compile
mock backend warmup + joint 小流程
```

真实 CARLA 中仍需重点观察：

```text
1. Ego 是否仍频繁左右横跳。
2. Ego 是否只在前方受阻/目标车道明显更安全时变道。
3. 背景车是否按巡航速度前进。
4. Ego 与 Adv 速度是否大致匹配。
5. road_completion 是否随 warmup 提升。
```

#### 12.24.12 CARLA 服务端周期性重启已移除

2026-06-16 结论：训练脚本托管并周期性重启 CARLA 服务端的方案已从 `training/train.py` 中删除。

删除原因：

```text
1. CARLA/UE4 在显存压力下可能触发 C++ terminate 或核心转储，Python 主进程难以可靠恢复。
2. 重启过程中旧 UE4 子进程、端口释放、Town 加载和同步 tick 都可能引入新的不稳定点。
3. 当前更稳妥的方案是降低显存占用，而不是在训练脚本中频繁重启服务端。
```

当前保留的稳定性措施：

```text
1. carla_0915.yaml 中启用 no_rendering_mode: true。
2. 训练脚本仅连接外部已经启动的 CARLA 服务端。
3. 保留 --carla-rpc-timeout，用于提高普通 CARLA RPC 调用超时时间。
4. 如需重启 CARLA，应在训练外部手动或用外部 supervisor 分段管理，不再由训练脚本内部管理。
```

当前建议流程：

先单独启动 CARLA：

```bash
/home/chenyuanwan/download/CARLA_0.9.15/CarlaUE4.sh -quality-level=Low -RenderOffScreen -nosound
```

然后启动训练：

```bash
cd /home/chenyuanwan/download/co-training/code-migration

PYTHONPATH=模型 /home/chenyuanwan/anaconda3/envs/cav-carla/bin/python \
  模型/carla_evolution/training/train.py \
  --mode joint \
  --backend carla \
  --config 模型/carla_evolution/configs/carla_0915.yaml \
  --joint-strategy adaptive \
  --rounds 100 \
  --num-cav 3 \
  --num-background 2 \
  --max-steps 600 \
  --reward-type global_R \
  --eval-interval 10 \
  --save-interval 10 \
  --carla-rpc-timeout 180
```

如果仍然出现 Vulkan 显存错误，优先降低 CARLA 侧显存占用：保持 `no_rendering_mode: true`，使用 `-RenderOffScreen -nosound`，必要时尝试 `-opengl` 或缩小并行/后台车辆规模。

#### 12.24.13 主车停车问题前三项修复

依据 `carla_evolution/md/carla_stop_issue_root_analysis.md` 的建议，已完成前三项低风险/中风险修复，目标是缓解主车避让后停在路中间、对抗车远距离学习信号断裂、以及 CARLA 弯道/匝道下前后车判断不准的问题。

已实现内容：

```text
1. Ego 低速惩罚
   文件: carla_evolution/envs/reward.py
   逻辑: ego.speed < 2.0 m/s 且 episode 未终止时，加入平滑 stop penalty。
   当前默认: stop_speed_threshold = 2.0, stop_penalty = 0.3

2. target_speed 自动恢复
   文件: carla_evolution/envs/scenario_manager.py
   配置: carla_evolution/configs/carla_0915.yaml
   逻辑: 当动作没有显式 accelerate/decelerate 时，如果 target_speed 低于 cruise_speed，按比例恢复。
   当前默认: target_speed_recovery = 0.12

3. 前后车 gap 改为前向投影
   文件:
     carla_evolution/envs/vehicle_state.py
     carla_evolution/envs/observation.py
     carla_evolution/envs/reward.py
     carla_evolution/envs/scenario_manager.py
   逻辑: 用车辆 heading/actor yaw 的 forward vector 计算 longitudinal gap，替代直接 other.x - vehicle.x。
   影响范围: observation lane context、ego front_gap reward、RoadModel surrounding vehicles、CARLA lane-change safety。

4. Adv 风险场距离门控改为连续衰减
   文件: carla_evolution/envs/reward.py
   旧逻辑: distance > 40m 时 shaped_delta = 0。
   新逻辑: shaped_delta = delta * max(0.1, exp(-distance / 40.0))。
```

预期影响：

```text
1. Ego 避让后长期停车的局部最优会被削弱。
2. DECELERATE 把 target_speed 压低后，KEEP_LANE/变道动作期间 target_speed 会逐步回到巡航速度。
3. 弯道或非 x 轴道路上，前车/后车判断更接近车辆实际行驶方向。
4. Adv 距离 ego 较远时仍保留较弱风险场学习信号，不再完全 0 梯度。
```

需要真实 CARLA 重点观察：

```text
1. Ego 避让后是否能恢复到 10-18 m/s，而不是停在路中间。
2. Ego 是否因为低速惩罚变得过于激进，导致碰撞率上升。
3. 变道安全判断改为前向投影后，是否减少弯道/匝道误判。
4. Adv 远距离阶段是否更愿意接近 ego，且不出现无意义自撞。
```

已验证：

```text
python -m py_compile vehicle_state.py observation.py reward.py scenario_manager.py
mock backend fixed joint 1 round / 2 adv / 1 background / 5 steps
```

#### 12.24.14 主车停车诊断 info 字段

已加入第四步诊断字段，不改变训练行为，用于判断主车避让后停在路中间时到底是策略动作、低层目标速度、变道失败/cooldown，还是 reward 停滞惩罚导致。

新增环境 `info` 字段：

```text
ego_target_speed
ego_target_lane_id
ego_lane_change_intent
ego_lane_change_timer
ego_lane_change_success
ego_lane_change_failed
ego_lane_change_cooldown
ego_last_action
ego_stop_penalty_value
```

已有并继续保留的相关 reward/metrics 字段：

```text
ego_front_gap
ego_front_gap_penalty
ego_speed_ratio
ego_risk_value
ego_min_adv_distance
ego_reward
```

写入训练 CSV 的 episode 末尾诊断字段：

```text
ego_target_speed
ego_front_gap
ego_last_action
ego_lane_change_failed
ego_lane_change_success
ego_lane_change_cooldown
ego_stop_penalty_value
```

涉及文件：

```text
carla_evolution/envs/scenario_manager.py  -> control_debug_info()
carla_evolution/envs/env.py               -> 合并 control_info 到 step info
carla_evolution/training/train.py         -> episode_metrics / joint_train_log.csv 字段
```

使用方式：

```text
1. 如果 ego_last_action 长期为 4 且 ego_target_speed 很低，说明策略倾向持续减速。
2. 如果 ego_last_action 不是减速但 ego_target_speed 长期低，说明 target_speed 恢复仍不够快。
3. 如果 ego_lane_change_failed=True 或 cooldown 长期较高，说明变道安全约束/cooldown 可能卡住主车。
4. 如果 ego_front_gap 很小且 target_speed 正常，说明低层跟车/避障仍需要更细控制。
5. 如果 ego_stop_penalty_value 长期较大，说明主车确实处在低速停滞状态。
```

已验证：

```text
python -m py_compile env.py scenario_manager.py train.py
mock backend fixed joint 1 round / 2 adv / 1 background / 5 steps
joint_train_log.csv 已写出新增诊断字段
```

#### 12.24.15 MAPPO 对抗奖励 DPF-style shaping 更新

依据 `carla_evolution/md/carla_stop_issue_reward_update.md` 第 106 行之后的建议，并结合当前 MAPPO 与 ego 速度不匹配、合围困难的问题，已在 `RiskFieldReward` 中加入保守版 DPF-style interaction shaping。

旧 adv reward：

```text
adv_reward = 10 * risk_delta * distance_weight + crash_reward
```

当前 adv reward：

```text
adv_reward =
  10 * risk_delta * distance_weight
  + crash_reward
  + 1.0 * speed_match_reward
  + 1.0 * ego_behavior_change
  + 0.5 * interaction_variance
```

新增项：

```text
1. speed_match_reward
   exp(-abs(adv.speed - ego.speed) / 5.0)
   作用: 让 MAPPO 先学会跟上 ego，降低速度不匹配导致的合围失败。

2. ego_behavior_change
   由 ego 速度变化、减速度、变道触发构成。
   作用: 鼓励 adv 触发 ego 的动态避让行为，而不是只形成静态 blocking。

3. interaction_variance
   根据 adv 相对 ego 的交互模式计算 unique modes / num_adv。
   模式包括: front_same_lane, rear_same_lane, front_adjacent, rear_adjacent, side_by_side。
   作用: 鼓励多辆 adv 形成不同交互位置，避免全部收敛到单一逼停模式。
```

保守权重选择：

```text
speed_match_weight = 1.0
behavior_change_weight = 1.0
interaction_variance_weight = 0.5
```

没有直接采用文档中的 `+5 * interaction_variance` 和 `+3 * ego_behavior_change`，原因是当前 risk_delta 项经过限幅和距离衰减，直接使用大权重可能压过风险场主目标，导致 adv 为了制造变化而乱跑。当前先用较小权重作为辅助 shaping，后续根据真实 CARLA 训练结果再调整。

新增日志字段：

```text
adv_ego_behavior_change
adv_ego_speed_change
adv_ego_deceleration
adv_ego_lane_change
adv_interaction_variance
adv_mean_speed_match_reward
```

需要真实 CARLA 重点观察：

```text
1. adv_mean_speed_match_reward 是否逐步升高，说明 adv 与 ego 速度更匹配。
2. adv_interaction_variance 是否不长期接近 0，说明多 adv 没有完全挤在同一种交互模式。
3. adv_ego_behavior_change 是否有非零波动，说明 adv 能触发 ego 减速/变道/速度变化。
4. ego crash rate 是否回升，但 adv self-crash 不应明显升高。
5. 如果 adv 变得过于扰动或自撞增加，优先降低 behavior_change_weight 或 interaction_variance_weight。
```

已验证：

```text
python -m py_compile reward.py train.py
mock backend fixed joint 1 round / 3 adv / 1 background / 5 steps
joint_train_log.csv 已写出新增 MAPPO shaping 指标
```

#### 12.24.16 Ego 训练阶段冻结 MAPPO 改为随机动作采样

问题：主车训练阶段中，MAPPO 参数冻结且动作使用 `argmax`，在固定递增 seed 的场景下，对抗车行为高度重复，导致 ego 容易过拟合同一类对手行为。

修改：

```text
文件: carla_evolution/training/train.py
位置: run_ego_episode()
旧逻辑: adv_actions = mappo.action(state, n_agents)              # deterministic argmax
新逻辑: adv_actions = mappo.exploration_action(state, n_agents)  # stochastic sampling
```

保持不变：

```text
1. MAPPO 在 ego 训练阶段仍然冻结，不更新参数。
2. ego warmup 中 fixed_adv_action 仍然使用规则动作，不受影响。
3. MAPPO 自己训练阶段仍然使用 exploration_action。
```

预期影响：

```text
1. Ego 训练阶段面对的 adv 行为更丰富。
2. 减少 ego 对固定 MAPPO argmax 轨迹的过拟合。
3. 主车碰撞率/道路完成率短期可能波动变大，但泛化压力更合理。
```

已验证：

```text
python -m py_compile train.py
mock backend fixed joint 1 round / 3 adv / 1 background / 5 steps
```

#### 12.24.17 MAPPO/Ego 交替训练场景随机化

问题：之前每个 episode 的 seed 使用 `args.seed + totals["global"]`，是简单递增序列；同时 relative_route 初始化中的车辆 offsets 固定，导致多次交替训练中 ego/MAPPO 经常面对高度相似的初始场景。

已实现两层随机化：

```text
1. 训练侧 episode seed 随机化
   文件: carla_evolution/training/train.py
   新增函数: episode_seed(args, episode_index)
   默认不再使用简单递增 seed，而是基于 base seed 生成可复现的随机 episode seed。

2. CARLA relative_route 初始相对位置扰动
   文件: carla_evolution/envs/scenario_manager.py
   配置: carla_evolution/configs/carla_0915.yaml
   对 adv / ego / hdv / background 的纵向 offset 加均匀扰动。
```

新增配置：

```yaml
randomize_scenarios: true
relative_offset_jitter: 5.0
randomize_spawn_start_index: false
```

新增训练参数：

```text
--no-scenario-randomization   关闭场景随机化，恢复 args.seed + episode_index 的简单递增 seed
--scenario-seed-span          随机 episode seed 的范围，默认 1000000
```

设计取舍：

```text
1. 默认只扰动车辆相对纵向 offset，不随机起始 spawn point。
2. randomize_spawn_start_index 默认 false，避免 Town04 中路线突然变化过大。
3. 使用 base seed 生成随机 episode seed，因此同一命令仍可复现实验，只是不再是简单递增场景。
```

预期影响：

```text
1. Ego 训练阶段不再长期面对同一组 adv 初始相对位置。
2. MAPPO 训练阶段能看到更多合围初始距离，有利于提升泛化。
3. 训练曲线短期波动可能增大，但不容易过拟合固定场景。
```

已验证：

```text
python -m py_compile train.py scenario_manager.py
mock backend fixed joint 1 round / 3 adv / 1 background / 5 steps
episode_seed 输出为可复现随机序列，而非简单递增
```

#### 12.24.18 Ego 追尾抑制与停车奖励条件化

问题：CARLA 真实动力学下，ego 在同车道前车 gap 很小时仍可能保持较高 target_speed，导致反复追尾；同时旧版 ego crash penalty 只有 -5，碰撞代价偏轻。之前加入的 stop penalty 只要低速就罚，在前方堵塞或近车时会错误惩罚合理低速。

已实现内容：

```text
1. 修复 ScenarioManager._jitter_offsets 缺失
   文件: carla_evolution/envs/scenario_manager.py
   原因: relative_route 随机化调用 _jitter_offsets，但类中未定义该方法。
   逻辑: 对 adv / hdv / background offsets 进行 [-jitter, +jitter] 均匀扰动；jitter <= 0 时保持原 offset。

2. Ego 低层前车安全速度限制
   文件: carla_evolution/envs/scenario_manager.py
   配置: carla_evolution/configs/carla_0915.yaml
   逻辑: CARLA high-level intent 转 VehicleControl 前，检查 ego 当前车道最近前车 gap。
        当前车道前方 gap 小于 ego_front_safety_distance 时，限制 target_speed。
        gap 进入 light/emergency 区间时，覆盖为轻刹/紧急刹车，直接降低追尾概率。

3. Mock backend 同步加入近前车刹车保护
   文件: carla_evolution/envs/scenario_manager.py
   逻辑: mock _apply_control 中，ego 前方同车道 gap 过小时压制 throttle 并提高 brake。
   目的: smoke test 和无 CARLA 环境下也能覆盖近前车保护路径。

4. 提高 ego crash penalty
   文件: carla_evolution/envs/reward.py
   旧默认: crash_vehicle_penalty = 5.0
   新默认: crash_vehicle_penalty = 20.0
   影响: ego 碰撞终止奖励从 -5 提高到 -20，更贴近 CARLA 中碰撞后果。

5. stop_penalty 改为条件式
   文件: carla_evolution/envs/reward.py
   旧逻辑: ego.speed < stop_speed_threshold 时直接惩罚。
   新逻辑: 先检查同车道前方 gap。
        若 front_gap <= blocked_front_distance，认为前方堵塞/近车，允许短暂低速，不加 stop penalty。
        若前方无车或 gap 较大，ego 仍低速，才按 stop_penalty 强罚。
```

新增/调整配置：

```yaml
ego_front_safety_distance: 22.0
ego_front_time_headway: 1.5
ego_front_light_brake_gap: 12.0
ego_front_light_brake: 0.20
ego_front_emergency_gap: 8.0
ego_front_emergency_brake: 0.45
```

新增调试信息：

```text
ego_front_gap_control
ego_front_speed_control
ego_safety_speed_limit
ego_safety_brake
ego_stop_front_gap
ego_stop_front_blocked
ego_stop_penalty_value
```

预期影响：

```text
1. Ego 在同车道近前车场景下会先降低 target_speed，必要时轻刹，减少反复追尾。
2. 碰撞终止奖励更重，训练中 ego 不应再把追尾当作低成本行为。
3. 前方堵塞时低速不再被 stop penalty 错罚；前方空旷还低速才会被强罚。
4. 场景随机化不再因 _jitter_offsets 缺失而在 reset 阶段 AttributeError。
```

已验证：

```text
python3 -m compileall carla_evolution/envs/scenario_manager.py carla_evolution/envs/reward.py
PYTHONPATH=模型 python3 模型/carla_evolution/scripts/control_smoke_test.py --backend mock --action accelerate --steps 5
单元式检查: _jitter_offsets 可调用；堵塞低速不罚；空旷低速会罚；ego crash reward = -20.0
```

### 12.24.19 奖励函数更新：自适应风险阈值与 MAPPO 多样性约束

根据 `carla_evolution/md/carla_reward_speed_safety_adaptive_diversity.md` 的建议，当前奖励函数已从固定风险惩罚进一步调整为速度-安全自适应约束形式：

1. Ego 奖励中的风险项改为约束违反惩罚：
   - 仍保留原来的距离风险 `ego_risk_raw`。
   - 新增自适应风险阈值 `ego_adaptive_tau`。
   - 只有当 `ego_risk_raw > ego_adaptive_tau` 时才产生 `ego_risk_violation`。
   - 实际惩罚为 `ego_risk_constraint_penalty = risk_violation_weight * ego_risk_violation`。

2. 自适应阈值 `tau` 由三部分动态调节：
   - `ego_interaction_entropy`：MAPPO 与主车交互模式越丰富，允许的探索边界越宽。
   - `ego_interaction_coverage`：adv 在主车前/后/侧方覆盖越充分，阈值适当提高。
   - `ego_performance_easing`：主车表现较差时提高阈值，避免早期训练被固定风险惩罚压死。

3. MAPPO 奖励新增交互熵多样性项：
   - 原有风险场势能变化、碰撞奖励、速度匹配、ego 行为扰动项保留。
   - `adv_interaction_variance` 继续表示交互模式覆盖度。
   - 新增 `adv_interaction_entropy` 和 `adv_diversity_reward`，减少 MAPPO 长期坍缩到单一阻挡/追尾模式。

4. 训练日志新增诊断字段：
   - Ego: `ego_risk_raw`, `ego_adaptive_tau`, `ego_risk_violation`, `ego_risk_constraint_penalty`, `ego_interaction_entropy`, `ego_interaction_coverage`, `ego_performance_easing`。
   - MAPPO: `adv_interaction_coverage`, `adv_interaction_entropy`, `adv_diversity_reward`。

该修改不会改变观测维度和网络结构，只改变奖励计算与日志输出。后续训练时重点观察：

- 主车是否不再通过长期低速规避风险惩罚；
- `ego_risk_violation` 是否只在真正近距离高风险交互时出现；
- `adv_interaction_entropy` 是否逐步上升，避免 MAPPO 只学会单一模式；
- MAPPO 是否能在速度匹配基础上形成更多前/后/侧方组合交互。

### 12.24.20 MAPPO 训练算法补全

当前 CARLA MAPPO 已从旧版的简化 PPO 更新流程升级为完整 rollout 驱动的 PPO/GAE 流程。旧版 `MARL1/MAPPO.py` 和迁移初版都使用 target actor 近似旧策略，并以折扣回报减 value 作为 advantage；该结构保留了 PPO clip 形式，但不具备标准 PPO rollout 所需的完整数据。

本次改动包括：

1. 共享离散 Actor 保持不变：
   - 每辆 adv 仍使用相同的 25 维局部观测和五动作离散策略。
   - Actor 网络层级保持 `25 -> 128 -> 128 -> 5`，便于继续使用现有策略结构。

2. Critic 改为 centralized critic：
   - 每个 agent 的 Critic 输入由 `local_state + mean(all_adv_states) + max(all_adv_states)` 组成。
   - 输入维度固定为 `3 * state_dim`，因此可支持不同数量的 adv。
   - 该结构先解决 Critic 看不到整体协作状态的问题，后续可直接将 pooled context 替换为 GNN embedding。

3. rollout 完整保存：
   - `state`、离散 `action`、`reward`；
   - 采样时的 `old_log_prob` 和 `old_value`；
   - `done`、`next_state`；
   - agent `active_mask`。

4. 使用 GAE：
   - 默认 `gamma=0.98`；
   - 默认 `gae_lambda=0.95`；
   - timeout 截断时允许从 next value bootstrap，真正终止时 value 归零。

5. 使用标准 PPO mini-batch 多轮更新：
   - 默认每批 rollout 执行 10 个 PPO epoch；
   - 使用采样时保存的 old log-prob 计算 probability ratio；
   - Actor 使用 clipped surrogate objective；
   - Critic 使用 clipped value loss；
   - advantage 默认归一化；
   - entropy coefficient 正式进入 Actor loss。

6. adv 碰撞后的样本屏蔽：
   - adv 自身碰撞的当步仍保留碰撞惩罚并写入训练样本；
   - 碰撞后的后续步骤通过 `active_mask` 排除，避免同一失效车辆连续重复贡献 `-40` 奖励并污染 GAE。

7. 新增命令行参数：

```text
--gae-lambda 0.95
--mappo-ppo-epochs 10
--value-coef 0.5
--entropy-reg 0.01
```

8. 新增训练日志：

```text
entropy
clip_fraction
value_error
ppo_epochs
```

9. checkpoint 变化：
   - 新 checkpoint 同时保存模型参数、优化器参数和维度信息。
   - Actor 层结构未变，可以单独迁移旧 Actor 权重。
   - 旧 Critic 使用局部 `state + action` 输入，与新 centralized critic 不兼容，必须重新训练 Critic。

验证状态：

- `py_compile` 已通过；
- mock 后端 5 episode 短训练已实际执行 GAE 和 PPO 更新；
- checkpoint 保存、加载和确定性推理已通过；
- CSV 已输出 entropy、clip fraction 和 value error。

### 12.24.21 非 Ego 碰撞退出、Adv 团队安全奖励与低层安全约束

为解决“非主车碰撞后阻塞道路并使 Ego 停车”和“对抗车内部频繁碰撞”问题，已完成以下修改。本次没有提高 adv 自碰撞终端惩罚，仍保持 `-40`，避免 MAPPO 通过停车或远离 Ego 规避大额惩罚。

#### 1. 非 Ego 碰撞车辆生命周期

- episode 终止条件仍只使用 Ego 碰撞或路线成功，adv/background/HDV 自身碰撞不会结束场景。
- 碰撞传感器在碰撞当步记录 `newly_collided_adv_indices`，对应 adv 当步保留 `-40` 惩罚。
- 碰撞后的后续 step 不再重复计算 `-40`，MAPPO rollout 的 `active_mask` 同时屏蔽其后续样本。
- 碰撞车辆立即从以下有效集合排除：
  - Ego 和 adv 的前车安全控制候选；
  - 25 维观测中的邻车；
  - MAPPO 风险场、交互熵和覆盖度；
  - Ego 风险、自适应 tau 和前车间距奖励；
  - 平均速度等有效车辆统计。
- 为保持 MAPPO agent 数量、动作顺序和张量维度固定，CARLA 中不直接删除 adv 槽位：
  - 默认等待 5 个后续 simulation step；
  - 将碰撞 actor 移到 `z=-50m`；
  - 关闭 actor physics；
  - 对应 `VehicleState` 保留为 `active=False` 的零观测占位。
- mock 后端实现了相同的失效和延迟退出语义。

配置：

```yaml
non_ego_collision_retire_delay_steps: 5
retired_actor_z: -50.0
```

#### 2. Adv-Adv 连续安全奖励

在原有风险场、速度匹配、行为扰动和多样性奖励基础上加入三类平滑惩罚：

1. 团队近距离惩罚：
   - 10m 内开始生效；
   - 最大 `-1.0/step`。
2. 团队低 TTC 惩罚：
   - 同车道且 TTC 小于 3s 时生效；
   - 最大 `-2.0/step`。
3. 重复占位惩罚：
   - 两辆 adv 处于相同 Ego 交互模式且相距小于 15m；
   - 最大约 `-0.5/step`。

这些奖励在真正碰撞前提供连续梯度，目标是让 adv 调整速度、轨迹和角色位置，而不是依赖更大的碰撞惩罚。

新增 info/CSV 字段：

```text
adv_mean_close_distance_penalty
adv_mean_ttc_penalty
adv_mean_duplicate_mode_penalty
adv_min_team_distance
adv_min_team_ttc
active_adv_count
newly_collided_non_ego_count
retired_non_ego_count
```

#### 3. Adv 低层安全控制

- adv 保留 MAPPO 输出的离散攻击动作和换道意图。
- 仅在同车道前车距离过近时施加轻量 target-speed cap 和制动。
- 默认参数比 Ego 更激进，不会把 adv 变成普通保守跟车车辆：

```yaml
adv_front_safety_distance: 14.0
adv_front_time_headway: 0.8
adv_front_light_brake_gap: 8.0
adv_front_light_brake: 0.12
adv_front_emergency_gap: 5.0
adv_front_emergency_brake: 0.30
```

- adv 换道安全间距从前/后 `8m/6m` 提高到 `12m/9m`：

```yaml
adv_min_lanechange_gap_front: 12.0
adv_min_lanechange_gap_rear: 9.0
```

新增低层控制诊断：

```text
adv_front_safety_active_count
adv_front_safety_mean_brake
adv_front_control_min_gap
adv_front_control_mean_speed_limit
```

#### 4. 当前验证

- Python 语法编译通过。
- 定向 mock 碰撞测试通过：
  - 两辆 adv 碰撞时 episode 不终止；
  - 当步奖励分别为 `-40`；
  - 下一观测立即清零；
  - 后续奖励为 0；
  - 配置延迟后车辆变为 `active=False`。
- 定向安全奖励测试通过：在 6m 团队距离和 0.6s TTC 下产生连续距离、TTC 和重复占位惩罚。
- mock MAPPO 训练已完成实际 GAE/PPO 更新。
- 真实 CARLA 中 actor 移出道路和传感器回调时序仍需通过短程 smoke test 验证。

### 12.24.22 修正：碰撞车辆保留为物理事故障碍

根据训练目标调整 12.24.21 中的非 Ego 碰撞退出方式：碰撞车辆不再延迟移到地下，也不关闭物理模拟。实际状态拆分为：

```text
active / policy_active = False
physical_active = True
crashed = True
```

当前行为：

1. MAPPO 策略层：
   - adv 碰撞当步保留一次 `-40`；
   - 后续不再接收策略控制；
   - 对应 rollout 通过 `agents_dones` 和 `active_mask` 排除；
   - 自身 25 维 MAPPO 观测保持全零；
   - 不参与风险场、速度匹配、交互熵、覆盖度、合围和 adv-adv 安全奖励。

2. CARLA 物理层：
   - actor 保留在原道路位置；
   - 继续保持 physics 和 collision；
   - 每个 step 施加制动，使其逐渐成为静态或低速事故障碍物；
   - 不再执行移动到 `z=-50m` 或 destroy。

3. Ego 与其它车辆：
   - Ego 的车道结构观测仍能看到事故车辆；
   - Ego 前车安全限速、紧急制动和换道安全检查仍将其视为障碍物；
   - Ego 前车间距奖励与条件式停车逻辑仍包含事故车辆；
   - 其它 active adv 的前车安全控制也会规避事故车辆。

4. 配置清理：
   - 删除 `non_ego_collision_retire_delay_steps`；
   - 删除 `retired_actor_z`。

5. 新增日志：

```text
policy_inactive_non_ego_count
physical_accident_vehicle_count
```

定向 mock 验证结果：

- 两辆 adv 碰撞当步奖励均为 `-40`；
- episode 未终止；
- 两车状态为 `active=False, physical_active=True, crashed=True`；
- 继续运行 6 step 后位置仍保留；
- Ego 位于事故车后方约 5m 时，25 维观测正确报告前方障碍并触发制动；
- 两个失效 adv 的自身观测保持全零。

该修正使事故车辆成为 Ego 需要学习规避的二次风险，同时避免碰撞 adv 继续污染 MAPPO 策略训练。

### 12.24.23 急弯停车修复：路线弧长坐标与道路相对观测

针对主车在没有邻车时进入大曲率弯道后减速或停车的问题，已移除奖励和观测对 CARLA 世界直角坐标的依赖。原实现使用世界 `x` 作为前进距离，并将 Ego 初始航向延伸成直线路线；车辆正常沿弯道行驶时，世界 `x` 可能增长变慢、保持不变或下降，导致前进奖励和道路完成率错误。

#### 1. Waypoint 路线与弧长坐标

- reset 时从 Ego 当前 CARLA waypoint 构造道路中心线路线。
- waypoint 默认每 2m 采样一次。
- 路线分叉时选择与当前道路切线转角最小的后继。
- 为覆盖初始位于 Ego 后方的 adv，路线向后额外采样 100m。
- Ego 初始位置定义为 `route_s=0`：
  - 后方车辆 `route_s < 0`；
  - 前方车辆 `route_s > 0`。
- 每辆 actor 每步投影到最近路线段，得到：

```text
route_s         沿道路中心线的累计弧长
lateral_offset  相对路线中心线的有符号横向偏移
heading_error   车辆航向与道路切线的夹角
```

配置：

```yaml
route_sample_distance: 2.0
route_backtrack_distance: 100.0
```

#### 2. 道路完成率与 Ego 前进奖励

道路完成率改为：

```text
route_completion = route_s / route_length
```

Ego 前进奖励改为：

```text
forward_progress = route_s(t) - route_s(t-1)
```

不再使用世界 `x(t) - x(t-1)`。因此车辆沿急弯正常前进时仍持续获得正奖励。

风险场中的车辆平面坐标也改为：

```text
(route_s, lateral_offset)
```

避免同一条曲线道路上的纵向和横向关系被世界 x/y 扭曲。

#### 3. 25 维观测语义更新

维度仍为 25，网络输入大小不变，但前四维语义调整为：

```text
0: route_s / route_length
1: lateral_offset / lane_width
2: speed / 35
3: heading_error / pi
```

旧语义为世界 x、世界 y、速度、绝对 heading。其余相邻车道、TTC、风险和逃逸空间字段保持原结构。

#### 4. 曲线道路同车道识别

`VehicleState` 新增：

```text
road_id
section_id
raw_lane_id
```

CARLA 前车安全控制、换道间距检查、25 维邻车筛选和 Ego 前车奖励不再只比较 `lane_id`，而是同时比较道路和 section，避免急弯附近其它道路中相同 lane_id 的车辆被误判为本车道前车。

CARLA actor 的前后距离使用两车 `route_s` 差，不再使用当前车辆朝向的直线投影。

#### 5. 新增日志

```text
ego_route_s
ego_lateral_offset
ego_heading_error
ego_road_id
ego_section_id
ego_raw_lane_id
```

真实 CARLA smoke test 时应重点检查：

- 急弯中 `ego_route_s` 是否持续增加；
- `route_completion` 是否持续增加；
- 正常跟踪车道时 `abs(ego_heading_error)` 是否保持较小；
- 无前车时是否仍出现安全制动；
- road/section 切换时是否产生异常进度跳变。

#### 6. 验证与 checkpoint 注意事项

- 合成 90 度曲线测试中，世界 x 下降时 `route_s` 仍从 10m、30m、50m、70m 单调增加。
- 路线上的横向误差和 heading error 接近 0。
- Ego 沿曲线前进 3m 时，奖励正确报告 `ego_forward_progress=3.0`。
- 不同 road_id 但相同 lane_id 的车辆不再进入本车道前车观测。
- mock joint 交替训练和 Python 语法检查通过。

虽然观测仍为 25 维，但前四维语义发生变化。旧 EgoPPO 和 MAPPO Actor checkpoint 不应直接继续训练，建议从头训练；否则旧网络会把世界坐标学习到的权重错误应用到路线相对坐标。

## 2026-06-19：ADV 碰撞恢复与独立碰撞指标

本次修改不增加角色分工，不改变 25 维观测长度，也不修改 MAPPO Actor/Critic 网络结构。现有 checkpoint 的参数形状保持兼容。

### 1. 非 Ego 碰撞车辆不再永久退出

- ADV/背景车碰撞不终止 episode。
- `agents_dones` 不再因 ADV 的瞬时碰撞置为 `True`；只有车辆被真正移除、场景成功或 episode 结束时才终止对应 transition。
- 碰撞车辆继续保留完整 25 维观测，不再返回全零观测。
- 碰撞车辆继续作为物理障碍参与 Ego 前车检测、相邻车道观测、风险奖励和 ADV 安全距离计算。
- CARLA backend 使用短暂恢复状态机：先制动，再根据与碰撞对象的前后关系执行前进或倒车分离，并向可用相邻车道施加轻量转向。满足冷却时间和物理间距后恢复 MAPPO 控制。
- mock backend 使用确定性的相邻车道分离逻辑，供无 CARLA 服务端测试。

配置参数：

```yaml
collision_contact_clear_steps: 8
collision_recovery_brake_steps: 3
collision_recovery_throttle: 0.35
collision_recovery_reverse_throttle: 0.25
collision_recovery_steer: 0.18
collision_recovery_clearance: 6.0
```

### 2. ADV 碰撞奖励

在原有风险场、速度匹配、行为变化、多样性、距离、TTC 和重复占位奖励基础上增加：

```text
首次进入碰撞：-40
持续接触每步：-2
成功分离恢复：+5
```

恢复奖励仅用于鼓励摆脱无效接触，不抵消首次碰撞惩罚。角色分工尚未加入，现有行为多样性奖励保持不变。

### 3. 日志口径修正

旧 `agents_crash_rate` 由 `agents_dones` 推导，场景成功也会把所有 ADV 记为 done，因此不能代表 ADV 碰撞率。现改为“episode 内是否发生过 ADV 相关碰撞”，并新增：

```text
adv_adv_collision_rate
adv_ego_collision_rate
adv_other_collision_rate
adv_recovery_rate
adv_collision_event_count
adv_adv_collision_event_count
adv_ego_collision_event_count
adv_other_collision_event_count
adv_recovery_count
recovering_adv_count
```

上述类型碰撞率同时写入窗口日志和整轮日志。`active_adv_count` 表示当前可由策略正常控制的 ADV 数量，`recovering_adv_count` 表示正在执行低层分离控制的 ADV 数量。

### 4. 验证结果

- Python 语法检查通过。
- `test_adv_collision_recovery` 通过：碰撞时观测非零、ADV `done=False`、首次惩罚约 -40、分离后恢复 active 并获得小额奖励。
- 1 个 ADV episode + 1 个 Ego episode、每 episode 5 步的 mock 联合训练通过。
- 新的窗口和整轮 CSV 字段均可由 `csv.DictReader` 正常解析。

注意：修改代码不会影响已经启动并驻留内存的训练进程。应用该逻辑时需要停止旧训练，在保存 checkpoint 后启动新进程。由于本次修改没有改变观测维度和网络结构，可以复用当前这轮采用路线相对 25 维观测训练得到的 MAPPO/Ego checkpoint；Highway checkpoint 或更改观测语义之前的旧 checkpoint 仍不应直接复用。新旧日志不应追加到同一个 CSV 文件。

## 2026-06-19：第40个 phase 后的 Highway 等价 50 轮续训

现有 CARLA `adaptive` 模式把一次单方训练记为一个 round。完成40个 phase 后，Ego 与 ADV 实际各训练20次。为使两者最终都达到 Highway 口径的70轮，新增 `staged-pairs` 模式：

- 续训50轮；
- 每轮固定执行 `ADV -> Ego`；
- ADV 和 Ego 各增加50次训练；
- 加上已有20次，最终两者各70次；
- 不加入 ADV 角色分工。

阶段和 episode 区间：

| 续训轮次 | 数量 | ADV episode | Ego episode |
|---|---:|---:|---:|
| 初始阶段 | 10轮 | 150-300 | 300-500 |
| 中间阶段 | 15轮 | 100-200 | 500-800 |
| 最后阶段 | 25轮 | 80-150 | 800-1000 |

新增参数：

```text
--joint-strategy staged-pairs
--rounds 50
--round-offset 20
--stage-cycles 10,15,25
--resume-adv-model-dir
--resume-adv-step
--resume-ego-checkpoint
--initial-adv-episodes
--initial-ego-episodes
```

`round-offset=20` 使用 Highway 等价轮次编号，因此新日志从第21轮写到第70轮。checkpoint 加载时恢复 MAPPO Actor、集中式 Critic和优化器，以及 EgoPPO 网络、优化器、step和episode计数。新训练写入独立结果目录，不与修改前的 CSV 混写。

如果 `--resume-adv-step` 省略，程序自动选择 MAPPO 目录中 Actor/Critic 共同存在的最大 step。如果 `--resume-ego-checkpoint` 指向目录，程序自动选择最大的 `checkpoint-N.pt`。未显式提供初始 episode 计数时，MAPPO 使用所选 step，Ego 使用 checkpoint 内保存的 `n_episodes`。

## 2026-06-22：CARLA 静态障碍物识别与风险车道避让

针对第54轮后 Ego 明显偏向 `LANE_RIGHT`、进入最右侧右转道后撞到路灯的问题，本次修改不采用硬编码“向左变道”或固定目标车道，而是在 CARLA 低层控制和奖励中加入静态障碍物风险。

### 1. 静态障碍物缓存

`ScenarioManager` 在设置 route reference 后，从 CARLA environment objects 中读取配置标签：

```yaml
static_obstacle_labels: [Poles, TrafficLight, TrafficSigns]
```

对每个障碍物投影到路线坐标和 driving lane，缓存：

```text
route_s
lateral
road_id
section_id
raw_lane_id
label
x/y
```

默认只在 ego 当前 lane 的前方 lookahead 范围内计算风险，避免把其它道路、其它 section 或相邻车道的物体误判为本车道障碍。

### 2. 当前车道与候选车道风险评分

新增 `VehicleState` 字段：

```text
obstacle_distance
obstacle_lateral
obstacle_risk
```

低层控制每步计算 ego 当前车道前方静态障碍风险。风险随距离增大而降低：

```text
risk = (1 - distance / lookahead)^2
```

当动作请求变道时，同时对目标相邻车道计算 `target_lane_obstacle_risk`。若目标车道风险明显高于当前车道，或目标风险超过 veto 阈值，则拒绝该次变道并记录：

```text
ego_target_lane_obstacle_risk
ego_lane_obstacle_veto
```

该逻辑只比较风险，不指定左侧或右侧。因此如果障碍物在左侧，左侧候选车道会被判为高风险；如果右侧路灯/右转道更危险，右侧候选车道会被 veto。

### 3. 障碍物安全限速

当 ego 当前车道前方存在静态障碍风险时，低层控制会限制 `target_speed`，接近紧急距离时施加轻刹或更强制动：

```yaml
ego_obstacle_lookahead_distance: 80.0
ego_obstacle_safety_risk_threshold: 0.25
ego_obstacle_emergency_distance: 12.0
ego_obstacle_light_brake: 0.18
ego_obstacle_emergency_brake: 0.55
```

这用于处理“已经处在风险车道上但尚未安全变道”的阶段，防止车辆继续高速驶向路灯或其它静态物体。

### 4. 奖励 shaping

Ego 奖励新增静态障碍风险惩罚：

```yaml
ego_obstacle_penalty_weight: 2.0
```

如果前一车道存在较高障碍风险，变道后 `obstacle_risk` 明显下降，则给小额 escape bonus：

```yaml
ego_obstacle_escape_bonus: 0.5
ego_obstacle_escape_threshold: 0.35
ego_obstacle_escape_margin: 0.15
```

这鼓励车辆学习“离开风险车道”，但不会奖励固定方向的变道。

### 5. 终止奖励顺序修正

Ego terminal reward 现在先判断：

```text
out_of_road
crash_vehicle
success
```

避免接近终点时发生路边/物体碰撞却因 route completion 达标被误判为 success。

### 6. 新增日志字段

训练 CSV 新增：

```text
ego_obstacle_distance
ego_obstacle_lateral
ego_obstacle_risk
ego_obstacle_penalty
ego_obstacle_escape_bonus
ego_obstacle_distance_control
ego_obstacle_risk_control
ego_target_lane_obstacle_risk
ego_lane_obstacle_veto
```

后续评估时应重点观察：`LANE_RIGHT` 触发时目标车道风险是否升高、`ego_lane_obstacle_veto` 是否生效、已经进入右转道后 `ego_obstacle_risk` 是否提前上升，以及车辆是否能在风险下降的车道恢复速度。

### 7. 验证结果

- Python 语法检查通过。
- mock backend `control_smoke_test.py --action accelerate --steps 5` 通过。
- 该修改不改变 25 维观测长度，但改变低层控制和奖励信号；建议停止旧训练进程后用新结果目录重新训练或至少不要把新旧 CSV 追加到同一个文件。

## 2026-06-22：从第52轮 checkpoint 续训到第70轮

目标：避开第54轮后 Ego 表现急转直下的 checkpoint，从第52轮恢复，继续训练到 Highway 等价第70轮。当前第52轮到第70轮还差18轮，18轮全部使用最后阶段 episode 参数：

```text
ADV: 80-150 episodes / round
Ego: 800-1000 episodes / round
```

为支持“只跑最后阶段”，`--stage-cycles` 现在允许 0。使用：

```text
--rounds 18
--round-offset 52
--stage-cycles 0,0,18
```

这样 local round 1-18 全部进入 final stage，不会混入 initial/middle episode 档。

第52轮对应 checkpoint：

```text
MAPPO actor/critic: 模型/carla_evolution/results/joint_Jun_19_16_48_04/models/adv @ step 10320
EgoPPO checkpoint: 模型/carla_evolution/results/joint_Jun_19_16_48_04/models/ego/checkpoint-37900.pt
initial ADV episodes: 10324
initial Ego episodes: 37900
```

注意：第52轮 round log 中 ADV 总 episode 为 10324，但磁盘上 actor/critic 共同存在的最新整十 step 是 10320，因此恢复模型参数应使用 `--resume-adv-step 10320`，同时用 `--initial-adv-episodes 10324` 保持 episode 计数与第52轮日志一致。

建议新开结果目录运行，不追加到旧 CSV：

```bash
PYTHONPATH=模型 /home/chenyuanwan/anaconda3/envs/cav-carla/bin/python 模型/carla_evolution/training/train.py \
  --mode joint \
  --backend carla \
  --config 模型/carla_evolution/configs/carla_0915.yaml \
  --joint-strategy staged-pairs \
  --rounds 18 \
  --round-offset 52 \
  --stage-cycles 0,0,18 \
  --num-cav 3 \
  --num-background 2 \
  --max-steps 700 \
  --reward-type agents_rewards \
  --eval-interval 10 \
  --save-interval 10 \
  --carla-rpc-timeout 180 \
  --resume-adv-model-dir 模型/carla_evolution/results/joint_Jun_19_16_48_04/models/adv \
  --resume-adv-step 10320 \
  --resume-ego-checkpoint 模型/carla_evolution/results/joint_Jun_19_16_48_04/models/ego/checkpoint-37900.pt \
  --initial-adv-episodes 10324 \
  --initial-ego-episodes 37900
```

该命令沿用上一轮长训练的环境与关键参数：`cav-carla` 解释器、3 个 CAV、2 个 background、`max_steps=700`、`agents_rewards`、每 10 轮 eval/save、`carla-rpc-timeout=180`。当前 shell 的 `/home/chenyuanwan/anaconda3/bin/python3` 缺少 PyTorch，只适合做语法检查，不适合直接启动训练。

## 2026-06-22：障碍物危险距离收紧与 lane center 横向过滤

最新第53轮日志中，`ego_front_gap` 约 60-100m，但 `ego_obstacle_distance` 约 35-40m、`ego_obstacle_risk` 约 0.25-0.32，导致 `ego_target_speed` 被压到 2.0。这说明 80m lookahead 与 4.5m 横向容忍过于保守，容易把路边杆/路灯投影成当前车道风险。

本次调整：

```yaml
ego_obstacle_lookahead_distance: 20.0
ego_obstacle_lateral_tolerance: 1.5
```

同时，静态障碍物风险判断不再使用“相对整条 route 的横向距离”作为危险过滤，而是使用障碍物相对其投影 driving waypoint 的 lane center 横向距离：

```text
lane_lateral = dot(obstacle_location - waypoint_center, lane_left_normal)
```

含义：

- route 横向距离：障碍物相对 ego 全局参考路线中心线的左右偏移，适合描述路线偏移，但不严格等同于车道内障碍。
- lane center 横向距离：障碍物相对 CARLA 投影车道中心线的左右偏移，更适合判断该物体是否真正占据当前车道。

风险仍要求 `road_id / section_id / raw_lane_id` 与 ego 当前车道一致；在此基础上，只有 `abs(lane_lateral) <= 1.5m` 且距离在 20m 内的静态物体才会触发 obstacle risk。这样可以保留对车道内障碍物的规避，同时降低路边灯杆导致主车停死的概率。

验证：

- `python3 -m compileall 模型/carla_evolution/envs/scenario_manager.py` 通过。
- 使用 `carla_0915.yaml` 的 mock env reset/step 通过。

## 2026-06-22：障碍后重新起步与成功避障小奖励

新增一个小的恢复奖励，用于鼓励 Ego 在被静态障碍迫使低速后，完成避障并恢复速度。该奖励不奖励停车本身，只奖励“低速等待/避让之后风险解除并重新起步”。

配置：

```yaml
ego_obstacle_recovery_bonus: 0.3
ego_obstacle_recovery_speed: 6.0
ego_obstacle_recovery_low_speed: 2.5
ego_obstacle_recovery_risk_threshold: 0.25
ego_obstacle_recovery_clear_risk: 0.10
ego_obstacle_recovery_cooldown: 50
```

触发条件：

1. 之前曾出现 `obstacle_risk >= 0.25` 且 `ego.speed <= 2.5m/s`，说明低速与静态障碍风险相关；
2. 当前 `obstacle_risk <= 0.10`，说明风险已解除或已避开；
3. 当前 `ego.speed >= 6.0m/s`，说明已经重新起步；
4. 未碰撞、未出路；
5. cooldown 满足后才可再次触发，避免重复刷奖励。

新增日志字段：

```text
ego_obstacle_recovery_bonus
ego_obstacle_recovered
ego_obstacle_low_speed_pending
```

验证：

- `python3 -m compileall 模型/carla_evolution/envs/reward.py 模型/carla_evolution/training/train.py` 通过。
- 单元式检查：障碍风险低速阶段只置 pending，不给奖励；风险清除且速度恢复到 7m/s 时给一次 0.3；下一步不重复给。

## 2026-06-24：静态障碍避障覆盖策略动作

最新日志显示，`ego_obstacle_escape_available=True` 但 `ego_obstacle_avoidance_active=False` 的情况较多，尤其右侧有 `Poles` 风险时仍没有主动换道。原因之一是旧逻辑只在策略当前 `lane_delta == 0` 时才允许自动避障覆盖；如果策略输出 accelerate/decelerate 或错误方向换道，低层不会主动切换到安全逃逸车道。

本次修改：

1. 当当前车道静态障碍风险高、障碍物在避障距离内、且存在安全相邻逃逸车道时，自动避障可以覆盖策略动作：
   - 覆盖 accelerate/decelerate；
   - 覆盖错误方向换道；
   - 仍然要求目标车道车辆间距安全、目标静态障碍风险更低。
2. 如果没有正在进行的换道，且 cooldown 为 0，则直接触发避障换道。
3. 如果处于 cooldown，但障碍距离已经进入 `ego_obstacle_emergency_distance`，允许清除 cooldown 后触发安全避障换道。
4. 如果正在换道中，不强行打断，避免频繁横摆。

新增日志字段：

```text
ego_obstacle_avoidance_blocked_reason
```

取值用于判断为什么 `escape_available=True` 但没有触发避障：

```text
lane_change_in_progress
cooldown
policy
```

对最新日志的解释：

- 最右侧路灯/杆子风险主要表现为 `ego_obstacle_label=Poles`，常见于 `ego_raw_lane_id=-2`，但旧逻辑下自动避障触发率过低。
- 最左侧带黄线车道的无障碍低速样本，多数 `ego_obstacle_risk=0`，但 `ego_last_action=DECELERATE` 或刚完成换道后 target speed 仍低，属于策略/历史 target speed 问题，不是静态障碍触发。

验证：

- `python3 -m compileall 模型/carla_evolution/envs/scenario_manager.py 模型/carla_evolution/training/train.py` 通过。
- 单元式检查：无 cooldown 时允许覆盖；远距离 cooldown 不覆盖；进入 emergency distance 的 cooldown 允许覆盖；正在换道中不覆盖。

## 2026-06-25：清空道路低速卡死恢复

最新日志中，Ego 在最左侧或边缘车道出现 `target_speed=2.0` 且长时间不动的情况。部分样本没有前车、没有静态障碍风险，通常是历史 `DECELERATE`、换道成功后的 cooldown 或之前的安全限速把 target speed 压低后没有及时恢复。

本次新增低层保护：当路径清空时，把历史低速 target speed 抬回最低通行速度，而不是继续慢慢恢复。

配置：

```yaml
ego_clear_path_recovery_speed: 8.0
ego_clear_path_recovery_trigger_speed: 4.0
ego_clear_path_front_gap: 25.0
ego_clear_path_obstacle_risk: 0.10
ego_clear_path_recovery_during_lane_change: false
```

触发条件：

1. 仅对 Ego 生效；
2. 当前无安全制动；
3. `target_speed <= 4.0`；
4. 当前静态障碍风险 `<= 0.10`；
5. 前方无近车，或前车距离 `>= 25m`；
6. 当前不在换道中。

满足条件时：

```text
target_speed = max(target_speed, 8.0)
```

新增日志字段：

```text
ego_clear_path_recovery_active
```

验证：

- `python3 -m compileall 模型/carla_evolution/envs/scenario_manager.py 模型/carla_evolution/training/train.py` 通过。
- 单元式检查：清空道路会从 2m/s 抬到 8m/s；有近前车、有障碍风险、有安全刹车、正在换道时都不会抬速。

## 2026-06-22：提高 target speed 自动恢复速度

前一版 `target_speed_recovery=0.04` 从低速恢复到巡航速度较慢。例如 target speed 已被压到 2m/s 时，每步只恢复当前差值的 4%，在安全风险解除后仍可能长时间低速。

本次调整：

```yaml
target_speed_recovery: 0.12
```

控制逻辑仍保持平滑恢复：

```text
target_speed += recovery * (cruise_speed - target_speed)
```

因此不会一步跳回巡航速度，但恢复速度约为原来的 3 倍。该恢复只在当前动作没有显式 accelerate/decelerate 时生效；若前车安全限速或静态障碍物限速仍有效，低层安全限制仍会覆盖恢复后的 target speed。

验证：

- `python3 -m compileall 模型/carla_evolution/envs/scenario_manager.py` 通过。
- `carla_0915.yaml` 加载后 `target_speed_recovery=0.12`。

## 2026-06-22：静态障碍物避障优先换道，不再优先停车

目标调整：Ego 不应通过停车作为默认避障方式。只有当前车道前方确有静态障碍，且没有安全相邻车道可换，或已经接近极限距离时，才允许低层控制强制停车/强刹。若左右任一相邻车道车辆间距安全且静态障碍风险更低，则优先触发避障换道。

新增控制参数：

```yaml
ego_obstacle_critical_distance: 3.0
ego_obstacle_min_pass_speed: 6.0
ego_obstacle_avoidance_distance: 18.0
ego_obstacle_escape_risk_margin: 0.15
```

新逻辑：

1. 当前车道 `obstacle_risk` 超过阈值且障碍物在避障距离内时，低层控制扫描左右相邻 driving lane。
2. 候选车道必须同时满足：
   - 换道前后车安全间距满足 `_lane_change_safe`；
   - 目标车道静态障碍风险明显低于当前车道；
   - 目标车道未被 `ego_lane_obstacle_veto` 判定为更危险。
3. 若存在候选车道，自动选择风险最低的一侧发起避障换道；不是固定向左或向右。
4. 有安全逃逸车道且距离大于 `ego_obstacle_critical_distance` 时，不再把 `target_speed` 压到 `target_speed_min=2.0`，而是至少保留 `ego_obstacle_min_pass_speed=6.0`。
5. 只有没有安全逃逸车道，或距离小于等于 `ego_obstacle_critical_distance` 时，才允许强制降到 2.0 并施加 emergency brake。

同时修复一个导致突然停车的低层限速 bug：

```text
safety_speed_limit 初始值为 -1.0 时，旧逻辑 min(-1.0, safe_speed) 会把本次障碍物安全限速错误压回 2.0。
```

现在若不存在已有安全限速，会直接使用本次计算出的 `safe_speed`；只有已经存在更严格的前车/障碍安全限速时，才取二者较小值。

新增日志字段：

```text
ego_obstacle_label
ego_obstacle_lane_lateral
ego_obstacle_escape_available
ego_obstacle_avoidance_active
```

这些字段用于判断：停车时是否真的没有可换车道、触发风险的静态障碍物是什么、该障碍物是否真正靠近 lane center。

验证：

- `python3 -m compileall 模型/carla_evolution/envs/scenario_manager.py 模型/carla_evolution/training/train.py` 通过。
- 单元式检查：障碍物距离 8m 且有 escape lane 时，`target_speed=6.0`、无强刹；没有 escape lane 时，`target_speed=2.0`、`safety_brake=0.55`。

## 2026-06-22：静态障碍物避障速度分级，减少不必要减速

进一步调整避障速度策略：如果相邻车道足够安全，可以不减速或少减速；只有距离非常近时才刹停。无安全逃逸车道时，也采用连续降速而不是一进入 emergency distance 就急刹。

配置更新：

```yaml
ego_obstacle_emergency_distance: 8.0
ego_obstacle_critical_distance: 3.0
ego_obstacle_min_pass_speed: 8.0
ego_obstacle_light_brake: 0.08
ego_obstacle_emergency_brake: 0.35
```

新行为：

- 有安全逃逸车道且障碍距离大于 8m：不施加障碍物限速，保持原 target speed，由换道控制完成避障。
- 距离进入 8m 内但仍大于 3m：按距离连续降低 target speed，保留至少 8m/s 级别的通行速度，并只施加轻刹。
- 距离小于等于 3m：才把 target speed 降到 `target_speed_min` 并施加 emergency brake。
- 没有安全逃逸车道时，从 8m 外到 3m 内也按连续速度曲线缓慢降速，不再直接刹停。

单元式检查结果：

```text
distance=12m, escape=True  -> target_speed=18.0, brake=0.0
distance=8m,  escape=True  -> target_speed=10.5, brake=0.04
distance=5m,  escape=True  -> target_speed=9.0,  brake=0.04
distance=2.5m,escape=True  -> target_speed=2.0,  brake=0.35
distance=12m, escape=False -> target_speed=12.5, brake=0.0
distance=8m,  escape=False -> target_speed=10.5, brake=0.04
distance=5m,  escape=False -> target_speed=9.0,  brake=0.04
distance=2.5m,escape=False -> target_speed=2.0,  brake=0.35
```

验证：

- `python3 -m compileall 模型/carla_evolution/envs/scenario_manager.py` 通过。
- 使用 `carla_0915.yaml` 的 mock env reset/step 通过。
