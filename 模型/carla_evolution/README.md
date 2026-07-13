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
