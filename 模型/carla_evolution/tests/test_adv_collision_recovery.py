import unittest

import numpy as np

from carla_evolution.envs.metrics import EpisodeMetrics
from carla_evolution.envs.observation import LowDimObservationBuilder
from carla_evolution.envs.reward import RiskFieldReward
from carla_evolution.envs.scenario_manager import ScenarioManager


class AdvCollisionRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.manager = ScenarioManager({
            "backend": "mock",
            "num_cav": 2,
            "num_hdv": 0,
            "route_length": 500.0,
            "policy_frequency": 5.0,
        })
        self.state = self.manager.reset(seed=1, options={"num_cav": 2, "num_hdv": 0})
        adv0, adv1 = self.state.adv_cav_vehicles
        adv0.set_lane(1)
        adv1.set_lane(1)
        adv0.position[0] = adv0.route_s = 20.0
        adv1.position[0] = adv1.route_s = 21.0
        self.state.ego_vehicle.set_lane(0)
        self.state.ego_vehicle.position[0] = self.state.ego_vehicle.route_s = 0.0

    def test_collision_keeps_observation_and_recovers(self):
        self.manager._detect_collisions()
        self.manager._update_metrics()

        observations = LowDimObservationBuilder().build_multi_agent_observation(self.state)
        info = EpisodeMetrics().update(self.state, 0.0)
        rewarder = RiskFieldReward()
        entry_rewards, _, _, _ = rewarder.compute_agent_rewards(self.state)

        self.assertEqual(self.state.newly_collided_adv_indices, (0, 1))
        self.assertEqual(self.state.colliding_adv_indices, (0, 1))
        self.assertEqual(self.state.adv_adv_collision_event_count, 1)
        self.assertTrue(np.all(np.linalg.norm(observations, axis=1) > 0.0))
        self.assertEqual(info["agents_dones"], (False, False))
        self.assertEqual(info["recovering_adv_count"], 2)
        self.assertLess(max(entry_rewards), -30.0)

        self.state.newly_collided_adv_indices = ()
        contact_rewards, _, _, _ = rewarder.compute_agent_rewards(self.state)
        self.assertLess(max(contact_rewards), 5.0)

        self.manager._step_mock([
            {"throttle": 0.0, "brake": 0.0, "lane_change": 0},
            {"throttle": 0.0, "brake": 0.0, "lane_change": 0},
            {"throttle": 0.0, "brake": 0.0, "lane_change": 0},
        ])
        recovery_rewards, _, _, _ = rewarder.compute_agent_rewards(self.state)
        info = EpisodeMetrics().update(self.state, 0.0)

        self.assertEqual(self.state.recovered_adv_indices, (0, 1))
        self.assertEqual(self.state.colliding_adv_indices, ())
        self.assertTrue(all(not adv.crashed and adv.active for adv in self.state.adv_cav_vehicles))
        self.assertEqual(self.state.adv_recovery_count, 2)
        self.assertEqual(info["active_adv_count"], 2)
        self.assertEqual(info["agents_dones"], (False, False))
        self.assertGreater(min(recovery_rewards), 0.0)


if __name__ == "__main__":
    unittest.main()
