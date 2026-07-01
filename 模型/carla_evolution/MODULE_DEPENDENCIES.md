# Module Dependencies

This file records the current dependency contract between the old highway-env
training code and the new migration package.

## Existing Training Flow

```text
run_joint_train.py
  -> build_mappo(config, env, test_seeds)
     -> MAPPO(env=env, state_dim=env.n_s, action_dim=env.n_a)
  -> build_ego_config(config, env)
     -> PPOConfig(state_dim=env.n_s, action_dim=env.n_a)
  -> JointTrainer(env, mappo, ego_config, natural_vehicle_model, ego_reward_fn)
     -> EgoEnvWrapper(env, mappo, reward_fn, natural_vehicle_model)
     -> EgoPPO(ego_config)
```

## Environment Contract Required By MAPPO

`MAPPO.py` currently requires:

```text
env.n_s
env.n_a
env.controlled_vehicles
env.reset() -> env_state, action_mask, obs2, obs3
env.step(action_tuple) -> next_state, global_reward, done, info, obs2, obs3
info["regional_rewards"]
info["average_speed"]
info["crashed"]
info["vehicle_speed"]
info["vehicle_position"]
render(mode="rgb_array")
close()
```

`controlled_vehicles` means adversarial CAVs controlled by MAPPO.

## Environment Contract Required By EgoEnvWrapper

`ego/ego_env_wrapper.py` additionally requires:

```text
env.ego_vehicle
env.hdv_vehicles
env.obs_hdv_list
env.observation_type.observe()
vehicle.position
vehicle.speed
vehicle.crashed
vehicle.on_road
vehicle.lane
vehicle.lane_index
```

The full action order is:

```text
[adv_cav_0, adv_cav_1, ..., ego, hdv_0, hdv_1, ...]
```

## Environment Contract Required By HDV Wrappers

`hdv/hdv_env_wrapper.py` and `hdv/coevolution_hdv_env_wrapper.py` require:

```text
env.hdv_vehicles
env.road.surrounding_vehicles(vehicle)
vehicle.lane_distance_to(front_vehicle)
obs3 or obs_hdv_list for HDV observations
```

## Migration Boundary

The new package keeps this contract at the environment boundary. Algorithm code
should only need small adapter-level changes while the simulator implementation
is replaced underneath.

The internal state object is `VehicleState`. Real simulator actors should be
converted into `VehicleState` before observations, rewards, and metrics are
computed. This keeps EgoPPO, MAPPO, and HDV PPO on the same interface.
