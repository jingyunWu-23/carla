"""Smoke test for the co-evolution environment interface."""

import argparse

from carla_evolution.envs.factory import make_env


def parse_args():
    parser = argparse.ArgumentParser(description="Environment smoke test")
    parser.add_argument("--backend", choices=["mock", "carla"], default="mock")
    parser.add_argument("--config", default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--num-cav", type=int, default=2)
    parser.add_argument("--num-hdv", type=int, default=2)
    parser.add_argument("--legacy", action="store_true", default=False)
    return parser.parse_args()


def run_gymnasium_style(args):
    config = {
        "backend": args.backend,
        "max_episode_steps": args.steps,
        "num_cav": args.num_cav,
        "num_hdv": args.num_hdv,
    }
    env = make_env(config, config_path=args.config)
    obs, info = env.reset(seed=669)
    print("reset", obs.shape, info)
    for _ in range(args.steps):
        action = [0] * (len(env.controlled_vehicles) + 1 + len(env.hdv_vehicles))
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    print("step", obs.shape, reward, terminated, truncated, info)
    env.close()


def run_legacy_style(args):
    config = {
        "backend": args.backend,
        "max_episode_steps": args.steps,
        "num_cav": args.num_cav,
        "num_hdv": args.num_hdv,
    }
    env = make_env(config, config_path=args.config, legacy=True)
    obs, action_mask, obs2, obs3 = env.reset(testing_seeds=669)
    print("legacy_reset", obs.shape, action_mask, obs2.shape, obs3.shape)
    action = tuple([0] * (len(env.controlled_vehicles) + 1 + len(env.hdv_vehicles)))
    obs, reward, done, info, obs2, obs3 = env.step(action)
    print("legacy_step", obs.shape, reward, done, obs2.shape, obs3.shape, sorted(info.keys())[:8])
    env.close()


def main():
    args = parse_args()
    if args.legacy:
        run_legacy_style(args)
    else:
        run_gymnasium_style(args)
        if args.backend == "mock":
            args.legacy = True
            run_legacy_style(args)


if __name__ == "__main__":
    main()
