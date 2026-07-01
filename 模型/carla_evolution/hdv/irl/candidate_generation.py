from __future__ import annotations

import itertools
from typing import Iterable, List, Optional, Sequence

import numpy as np

from carla_evolution.envs.factory import make_env
from carla_evolution.hdv.irl.features import CarlaHDVFeatureTracker, aggregate_step_features


ACTION_NAMES = {
    0: "LANE_LEFT",
    1: "KEEP_LANE",
    2: "LANE_RIGHT",
    3: "ACCELERATE",
    4: "DECELERATE",
}


def build_action_sequences(depth: int, max_candidates: int, seed: int = 0) -> List[tuple]:
    actions = sorted(ACTION_NAMES)
    depth = max(int(depth), 1)
    sequences = list(itertools.product(actions, repeat=depth))
    if len(sequences) <= max_candidates:
        return sequences

    archetypes = [(action,) * depth for action in actions]
    remaining = [sequence for sequence in sequences if sequence not in archetypes]
    rng = np.random.RandomState(seed)
    keep = max(0, int(max_candidates) - len(archetypes))
    sampled = []
    if keep > 0 and remaining:
        indices = rng.choice(len(remaining), size=min(keep, len(remaining)), replace=False)
        sampled = [remaining[int(index)] for index in indices]
    return archetypes + sampled


def build_candidate_scenes(
    config_path: Optional[str],
    episodes: int = 30,
    horizon: int = 8,
    seeds: Optional[Sequence[int]] = None,
    backend: Optional[str] = None,
    num_cav: int = 3,
    num_hdv: int = 3,
    num_background: int = 0,
    target_hdv_index: int = 0,
    candidate_depth: int = 2,
    max_candidates: int = 25,
    ego_action: int = 1,
    adv_action: int = 1,
    other_hdv_action: int = 1,
) -> List[List[np.ndarray]]:
    seeds = list(seeds or range(int(episodes)))
    if not seeds:
        seeds = [0]

    scenes = []
    action_sequences = build_action_sequences(candidate_depth, max_candidates)
    for episode in range(int(episodes)):
        seed = int(seeds[episode % len(seeds)])
        scene = []
        for action_sequence in action_sequences:
            env = make_env(
                config={"backend": backend} if backend else None,
                config_path=config_path,
                legacy=False,
            )
            try:
                env.reset(
                    seed=seed,
                    options={
                        "num_cav": int(num_cav),
                        "num_hdv": int(num_hdv),
                        "num_background": int(num_background),
                    },
                )
                scene.append(
                    rollout_hdv_candidate(
                        env,
                        hdv_actions=action_sequence,
                        horizon=horizon,
                        target_hdv_index=target_hdv_index,
                        ego_action=ego_action,
                        adv_action=adv_action,
                        other_hdv_action=other_hdv_action,
                    )
                )
            finally:
                env.close()
        scenes.append(scene)
    return scenes


def rollout_hdv_candidate(
    env,
    hdv_actions: Iterable[int],
    horizon: int,
    target_hdv_index: int = 0,
    ego_action: int = 1,
    adv_action: int = 1,
    other_hdv_action: int = 1,
) -> np.ndarray:
    hdv_actions = tuple(hdv_actions) or (1,)
    segment = max(1, int(np.ceil(float(horizon) / len(hdv_actions))))
    tracker = CarlaHDVFeatureTracker()
    step_features = []

    for step in range(int(horizon)):
        scenario = env.scenario_manager.state
        hdvs = list(getattr(scenario, "hdv_vehicles", []))
        if not hdvs:
            break
        target_index = int(np.clip(target_hdv_index, 0, len(hdvs) - 1))
        target_hdv = hdvs[target_index]
        step_features.append(tracker.observe(scenario, target_hdv))

        action_index = min(step // segment, len(hdv_actions) - 1)
        target_action = int(hdv_actions[action_index])
        actions = _full_action_vector(
            scenario,
            target_hdv_index=target_index,
            target_action=target_action,
            ego_action=ego_action,
            adv_action=adv_action,
            other_hdv_action=other_hdv_action,
        )
        _, _, terminated, truncated, _ = env.step(actions)
        if terminated or truncated:
            break
    return aggregate_step_features(step_features)


def _full_action_vector(
    scenario,
    target_hdv_index: int,
    target_action: int,
    ego_action: int,
    adv_action: int,
    other_hdv_action: int,
) -> List[int]:
    adv_count = len(getattr(scenario, "adv_cav_vehicles", []))
    hdv_count = len(getattr(scenario, "hdv_vehicles", []))
    background_count = len(getattr(scenario, "background_vehicles", []))
    actions = [int(adv_action)] * adv_count
    actions.append(int(ego_action))
    for idx in range(hdv_count):
        actions.append(int(target_action if idx == target_hdv_index else other_hdv_action))
    actions.extend([1] * background_count)
    return actions
