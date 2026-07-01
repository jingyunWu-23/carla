import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from carla_evolution.training.train import (
    latest_ego_checkpoint,
    latest_mappo_step,
    parse_stage_cycles,
    staged_pair_eps,
)


class StagedPairScheduleTest(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(stage_cycles="10,15,25", rounds=50)

    def test_stage_boundaries(self):
        self.assertEqual(parse_stage_cycles(self.args.stage_cycles), (10, 15, 25))
        self.assertEqual(staged_pair_eps(self.args, 1), (150, 300, 300, 500))
        self.assertEqual(staged_pair_eps(self.args, 10), (150, 300, 300, 500))
        self.assertEqual(staged_pair_eps(self.args, 11), (100, 200, 500, 800))
        self.assertEqual(staged_pair_eps(self.args, 25), (100, 200, 500, 800))
        self.assertEqual(staged_pair_eps(self.args, 26), (80, 150, 800, 1000))
        self.assertEqual(staged_pair_eps(self.args, 50), (80, 150, 800, 1000))

    def test_round_count_must_match_stage_sum(self):
        self.args.rounds = 49
        with self.assertRaises(ValueError):
            staged_pair_eps(self.args, 1)

    def test_latest_checkpoint_resolution(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "actor_100.pt",
                "critic_100.pt",
                "actor_120.pt",
                "critic_110.pt",
                "checkpoint-300.pt",
                "checkpoint-500.pt",
                "ego_round_20.pt",
            ):
                (root / name).touch()
            self.assertEqual(latest_mappo_step(directory), 100)
            self.assertEqual(
                latest_ego_checkpoint(directory),
                str(root / "checkpoint-500.pt"),
            )


if __name__ == "__main__":
    unittest.main()
