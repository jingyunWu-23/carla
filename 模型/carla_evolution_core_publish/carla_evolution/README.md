# CARLA Ego-Adversary Co-Training

This repository contains the core implementation for CARLA-based ego-vehicle and adversarial-vehicle co-training. The method combines MAPPO adversarial CAV policies, EgoPPO ego policies, scenario-level evolution, and conservative-behavior fine-tuning for the ego vehicle.

The public repository is intended to include source code and runnable configuration only. Generated results, logs, trained checkpoints, experiment notes, videos, and ad hoc test scripts should be kept out of version control.

## Environment

The codebase targets:

- Python 3.8
- CARLA 0.9.15
- PyTorch 1.12.1 with CUDA 11.3
- Gymnasium, Stable-Baselines3, NumPy, pandas, PyYAML, matplotlib

Create and activate the Python environment:

```bash
conda create -n cav-carla python=3.8 -y
conda activate cav-carla
```

Install dependencies:

```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

If you use a different CUDA version or CPU-only PyTorch, install the matching PyTorch package first, then install the rest of `requirements.txt`.

## CARLA Setup

Install CARLA 0.9.15 separately and start a CARLA server before running training:

```bash
./CarlaUE4.sh -carla-rpc-port=2000 -RenderOffScreen
```

The default CARLA config is:

```text
configs/carla_0915.yaml
```

It connects to:

```text
host: 127.0.0.1
port: 2000
town: Town04
fixed_delta_seconds: 0.05
synchronous_mode: true
```

When running package modules directly, expose the parent directory of this repository:

```bash
export PYTHONPATH="$(pwd)/..:${PYTHONPATH}"
```

This code reuses a small subset of the original `MARL1` algorithm modules. Keep
`MARL1` as a sibling directory of `carla_evolution`:

```text
workspace/
  carla_evolution/
  MARL1/
```

Only the lightweight source dependencies from `MARL1` are required. Trained
models, zip files, logs, caches, and old experiment outputs under `MARL1` are
not required.

## Main Entry Points

### Enhanced POET Ego Fine-Tuning

This is the main entry for the proposed conservative-scenario evolution and ego fine-tuning workflow:

```bash
python scripts/finetune_ego_enhanced_poet.py \
  --config configs/carla_0915.yaml \
  --frozen-ego-set-size 20 \
  --population-size 10 \
  --generations 15 \
  --no-improvement-patience 0
```

Useful scale parameters:

```text
--frozen-ego-set-size 20|30|50
--population-size 10|30|50
```

The script sets approximately 10 training episodes per evolved environment per generation:

```text
population-size=10 -> 100 episodes/generation
population-size=30 -> 300 episodes/generation
population-size=50 -> 500 episodes/generation
```

### Base Ego-Adversary Co-Training

The lower-level joint training entry is:

```bash
python -m carla_evolution.training.train \
  --mode joint \
  --backend carla \
  --config carla_evolution/configs/carla_0915.yaml
```

Use `--mode adv` for adversarial MAPPO-only training and `--mode ego` for EgoPPO-only training.

## Core Implementation

- `agents/mappo.py`: MAPPO policy used by adversarial CAV agents.
- `agents/ego_ppo.py`: EgoPPO wrapper and training adapter for the ego vehicle.
- `../MARL1/single_agent/Model_common.py`: actor network reused by MAPPO.
- `../MARL1/ego/`: original EgoPPO modules reused by `agents/ego_ppo.py`.
- `envs/env.py`: Gym-style CARLA environment wrapper.
- `envs/scenario_manager.py`: CARLA actor spawning, route scenario construction, control, safety logic, and metrics state.
- `envs/observation.py`: 25-dimensional low-dimensional observation builder shared by ego and adversarial vehicles.
- `envs/reward.py`: ego reward, adversarial risk-field reward, and unified reward manager.
- `training/train.py`: base adversarial, ego, and joint training loops.
- `scripts/finetune_ego_enhanced_poet.py`: Enhanced POET scenario evolution and ego fine-tuning workflow.

## Observation and Reward Summary

Both ego and adversarial CAV policies use the same 25-dimensional observation layout. The observation includes route progress, lateral offset, speed, heading error, lane availability, front/rear gaps, relative speeds, TTC, lane risks, escape-lane availability, and surrounded status.

The ego reward is computed by `CatSafetyEgoReward` in `envs/reward.py`:

```text
reward = driving + speed - risk + lane_bonus + obstacle_bonus - stop_penalty - obstacle_penalty
```

The adversarial reward is computed by `RiskFieldReward` in `envs/reward.py`:

```text
reward =
  10.0 * shaped_delta
  + crash_reward
  + speed_match_weight * speed_match
  + interaction_bonus
  + behavior_bonus
  + diversity_reward
  - close_penalty
  - ttc_penalty
  - duplicate_penalty
```

## Outputs

Training writes generated artifacts under `results/`, including logs, model checkpoints, summaries, and evaluation outputs. These files can be large and are not part of the source release.

Keep the following out of version control:

```text
results/
models/
*.pt
*.pth
*.zip
*.mp4
*.log
__pycache__/
.venv/
.idea/
md/
tests/
../MARL1/*.zip
../MARL1/PPOmodel/
../MARL1/joint_results/
../MARL1/__pycache__/
../MARL1/**/__pycache__/
```
