"""Single-ego control smoke test for the environment."""

import argparse

from carla_evolution.envs.action_adapter import DiscreteDrivingAction
from carla_evolution.envs.factory import make_env


ACTION_NAMES = {action.name.lower(): int(action.value) for action in DiscreteDrivingAction}
ACTION_NAMES.update({str(int(action.value)): int(action.value) for action in DiscreteDrivingAction})


def parse_args():
    parser = argparse.ArgumentParser(description="Single-ego control smoke test")
    parser.add_argument("--backend", choices=["mock", "carla"], default="mock")
    parser.add_argument("--config", default=None)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--action", choices=sorted(ACTION_NAMES), default="accelerate")
    parser.add_argument("--spawn-start-index", type=int, default=None)
    parser.add_argument("--route-length", type=float, default=None)
    return parser.parse_args()


def build_config(args):
    config = {
        "backend": args.backend,
        "max_episode_steps": args.steps,
        "num_cav": 0,
        "num_hdv": 0,
        "num_background": 0,
    }
    if args.spawn_start_index is not None:
        config["spawn_start_index"] = args.spawn_start_index
    if args.route_length is not None:
        config["route_length"] = args.route_length
    return config


def main():
    args = parse_args()
    action = ACTION_NAMES[args.action]
    env = make_env(build_config(args), config_path=args.config)
    obs, reset_info = env.reset(seed=669)
    print("reset", obs.shape, reset_info)
    print_vehicle("initial", env)

    reward = 0.0
    terminated = False
    truncated = False
    info = {}
    for step in range(args.steps):
        obs, reward, terminated, truncated, info = env.step([action])
        if step in {0, args.steps // 2, args.steps - 1} or terminated or truncated:
            print_vehicle(f"step_{step + 1}", env)
        if terminated or truncated:
            break

    print("result", {
        "action": args.action,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "route_completion": info.get("route_completion"),
        "average_speed": info.get("average_speed"),
        "crash": info.get("crash"),
        "timeout": info.get("timeout"),
    })
    env.close()


def print_vehicle(label, env):
    ego = env.ego_vehicle
    if ego is None:
        print(label, {"ego": None})
        return
    print(label, {
        "x": float(ego.x),
        "y": float(ego.y),
        "speed": float(ego.speed),
        "lane_id": int(ego.lane_id),
        "crashed": bool(ego.crashed),
        "on_road": bool(ego.on_road),
    })


if __name__ == "__main__":
    main()
