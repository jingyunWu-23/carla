"""Ego + adversarial CAV control smoke test."""

import argparse

from carla_evolution.envs.action_adapter import DiscreteDrivingAction
from carla_evolution.envs.factory import make_env


ACTION_NAMES = {action.name.lower(): int(action.value) for action in DiscreteDrivingAction}
ACTION_NAMES.update({str(int(action.value)): int(action.value) for action in DiscreteDrivingAction})


def parse_args():
    parser = argparse.ArgumentParser(description="Ego + adversarial CAV smoke test")
    parser.add_argument("--backend", choices=["mock", "carla"], default="mock")
    parser.add_argument("--config", default=None)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--num-cav", type=int, default=3)
    parser.add_argument("--ego-action", choices=sorted(ACTION_NAMES), default="accelerate")
    parser.add_argument("--adv-action", choices=sorted(ACTION_NAMES), default="keep_lane")
    parser.add_argument("--spawn-start-index", type=int, default=None)
    parser.add_argument("--route-length", type=float, default=None)
    return parser.parse_args()


def build_config(args):
    config = {
        "backend": args.backend,
        "max_episode_steps": args.steps,
        "num_cav": args.num_cav,
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
    adv_action = ACTION_NAMES[args.adv_action]
    ego_action = ACTION_NAMES[args.ego_action]
    env = make_env(build_config(args), config_path=args.config)
    obs, reset_info = env.reset(seed=669)
    print("reset", obs.shape, reset_info)
    print_snapshot("initial", env)

    reward = 0.0
    terminated = False
    truncated = False
    info = {}
    for step in range(args.steps):
        actions = [adv_action] * len(env.controlled_vehicles) + [ego_action]
        obs, reward, terminated, truncated, info = env.step(actions)
        if step in {0, args.steps // 2, args.steps - 1} or terminated or truncated:
            print_snapshot(f"step_{step + 1}", env)
        if terminated or truncated:
            break

    print("result", {
        "num_cav": len(env.controlled_vehicles),
        "ego_action": args.ego_action,
        "adv_action": args.adv_action,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "route_completion": info.get("route_completion"),
        "average_speed": info.get("average_speed"),
        "crash": info.get("crash"),
        "timeout": info.get("timeout"),
        "agents_rewards": info.get("agents_rewards"),
        "regional_rewards": info.get("regional_rewards"),
        "risk_fields": info.get("risk_fields"),
        "risk_deltas": info.get("risk_deltas"),
        "agents_dones": info.get("agents_dones"),
    })
    env.close()


def print_snapshot(label, env):
    data = {
        "ego": vehicle_summary(env.ego_vehicle),
        "advs": [vehicle_summary(vehicle) for vehicle in env.controlled_vehicles],
    }
    print(label, data)


def vehicle_summary(vehicle):
    if vehicle is None:
        return None
    return {
        "x": float(vehicle.x),
        "y": float(vehicle.y),
        "speed": float(vehicle.speed),
        "lane_id": int(vehicle.lane_id),
        "crashed": bool(vehicle.crashed),
        "on_road": bool(vehicle.on_road),
    }


if __name__ == "__main__":
    main()
