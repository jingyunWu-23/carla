# Co-evolution Migration

This package is the migration target for the existing
highway-env CAV-HDV co-evolution project.

The current `MARL1/` and `highway_env/` directories remain the baseline. New
code should be added here so the simulator layer can be replaced without
rewriting the training algorithms first.

## First Milestone

The first CARLA milestone is interface stability, not training performance:

- `reset()` can spawn the configured scenario.
- `step()` can run repeated low-dimensional control steps.
- discrete actions are converted into CARLA vehicle controls.
- collision and route-progress metrics are reported through `info`.
- the environment can be adapted to legacy `done`-style training loops.


## HDV Policy Note

Do not reuse the old `PPOmodel889` as a trainable checkpoint in this environment.
Use it only as a highway-env baseline reference. The preferred route is to train
a neutral HDV PPO policy in the new environment first, then fine-tune an
aggressive or human-like HDV policy from that neutral checkpoint.
