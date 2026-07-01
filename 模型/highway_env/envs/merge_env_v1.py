"""
This environment is built on HighwayEnv with one main road and one merging lane.
Dong Chen: chendon9@msu.edu
Date: 01/05/2021
"""
import copy
import random
import this
import math
import numpy as np
from gym.envs.registration import register
from typing import Tuple

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.road.objects import Obstacle
from highway_env.vehicle.kinematics import Vehicle


class MergeEnv(AbstractEnv):
    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """
    n_a = 5
    n_s = 25
    ad=0
    bd=0
    cd=0

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"},
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": True},
            "controlled_vehicles": 1,
            "screen_width": 1200,
            "screen_height": 600,
            "centering_position": [0.3, 0.5],
            "scaling": 3,
            "simulation_frequency": 15,  # [Hz]
            "duration": 20,  # time step
            "policy_frequency": 5,  # [Hz]
            "reward_speed_range": [10, 30],
            "COLLISION_REWARD": 200,  # default=200
            "HIGH_SPEED_REWARD": 1,  # default=0.5
            "HEADWAY_COST": 4,  # default=1
            "HEADWAY_TIME": 1.2,  # default=1.2[s]
            "MERGING_LANE_COST": 4,  # default=4
            "traffic_density": 1,  # easy or hard modes
            "scenario": "merge",  # "straight" or "merge"
        })
        return config

    def _reward(self, action: int) -> float:
        # Cooperative multi-agent reward
        return sum(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles) \
               / len(self.controlled_vehicles)

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        other_advs = [v for v in self.controlled_vehicles if v != vehicle]
        ego_vehicle = getattr(self, "ego_vehicle", None) or self.road.vehicles[min(len(self.road.vehicles) - 1, 3)]
        hdv_vehicles = getattr(self, "hdv_vehicles", None) or []
        hdv_vehicle = hdv_vehicles[0] if hdv_vehicles else self.road.vehicles[min(len(self.road.vehicles) - 1, 4)]

        p1_vehicle = other_advs[0] if len(other_advs) > 0 else ego_vehicle
        p2_vehicle = other_advs[1] if len(other_advs) > 1 else hdv_vehicle

        P = self.weixian(vehicle.position, ego_vehicle.position,
                         p1_vehicle.position, p2_vehicle.position, hdv_vehicle.position,
                         vehicle.speed, p1_vehicle.speed, p2_vehicle.speed,
                         hdv_vehicle.speed, ego_vehicle.speed)
        self.zuobiao.append(copy.copy(P))
        n_initial = len(self.controlled_vehicles)
        R = self.zuobiao[-(n_initial + 1)] - self.zuobiao[-1]

        # Crash reward logic: +100 if test vehicle crashes, -40 if adversarial vehicle crashes, 0 otherwise
        if ego_vehicle.crashed:
            rrr = 100
        elif vehicle.crashed:
            rrr = -40
        else:
            rrr = 0

        # Clamp and scale the potential field change
        if R < -1:
            R = -1
        if R > 1:
            R = 1
        if R < 0:
            R = 0.5 * R

        # Distance-based reward scaling
        # If distance > 40, reward is 0; if distance < 20, no scaling; otherwise scale by 0.5
        distance = abs(vehicle.position[0] - ego_vehicle.position[0])
        if distance > 40:
            R = 0
        elif distance < 20:
            R = R
        else:
            R = 0.5 * R

        # Combine potential field reward and crash reward
        reward = 10 * R + rrr
        return reward

    def _regional_reward(self):
        for vehicle in self.controlled_vehicles:
            neighbor_vehicle = []

            # vehicle is on the main road
            if vehicle.lane_index == ("a", "b", 0) or vehicle.lane_index == ("b", "c", 0) or vehicle.lane_index == (
                    "c", "d", 0):
                v_fl, v_rl = self.road.surrounding_vehicles(vehicle)
                if len(self.road.network.side_lanes(vehicle.lane_index)) != 0:
                    v_fr, v_rr = self.road.surrounding_vehicles(vehicle,
                                                                self.road.network.side_lanes(
                                                                    vehicle.lane_index)[0])
                # assume we can observe the ramp on this road
                elif vehicle.lane_index == ("a", "b", 0) and vehicle.position[0] > self.ends[0]:
                    v_fr, v_rr = self.road.surrounding_vehicles(vehicle, ("k", "b", 0))
                else:
                    v_fr, v_rr = None, None
            else:
                # vehicle is on the ramp
                v_fr, v_rr = self.road.surrounding_vehicles(vehicle)
                if len(self.road.network.side_lanes(vehicle.lane_index)) != 0:
                    v_fl, v_rl = self.road.surrounding_vehicles(vehicle,
                                                                self.road.network.side_lanes(
                                                                    vehicle.lane_index)[0])
                # assume we can observe the straight road on the ramp
                elif vehicle.lane_index == ("k", "b", 0):
                    v_fl, v_rl = self.road.surrounding_vehicles(vehicle, ("a", "b", 0))
                else:
                    v_fl, v_rl = None, None
            for v in [v_fl, v_fr, vehicle, v_rl, v_rr]:
                if type(v) is MDPVehicle and v is not None:
                    neighbor_vehicle.append(v)
            regional_reward = sum(v.local_reward for v in neighbor_vehicle)
            vehicle.regional_reward = regional_reward / sum(1 for _ in filter(None.__ne__, neighbor_vehicle))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        agent_info = []
        obs, reward, done, info,obs2,obs3 = super().step(action)

        info["agents_dones"] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        for v in self.controlled_vehicles:
            agent_info.append([v.position[0], v.position[1], v.speed])
        info["agents_info"] = agent_info

        # for vehicle in self.controlled_vehicles:
        #     vehicle.local_reward = self._agent_reward(action, vehicle)
        # local reward
        info["agents_rewards"] = tuple(vehicle.local_reward for vehicle in self.controlled_vehicles)
        # regional reward
        self._regional_reward()
        info["regional_rewards"] = tuple(vehicle.regional_reward for vehicle in self.controlled_vehicles)

        obs = np.asarray(obs).reshape((len(obs), -1))
        return obs, reward, done, info,obs2,obs3

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        #return any(vehicle.crashed for vehicle in self.controlled_vehicles) \
        #return self.road.vehicles[3].crashed\
        return any(vehicle.crashed for vehicle in self.road.vehicles) \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"]*0.4         #20*5

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return any(vehicle.crashed for vehicle in self.controlled_vehicles) \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"]*0.4

    def weixian(self, p0, pg, p1, p2, pz, v0, v1, v2, vz, vg):
        # p0: Current adversarial vehicle position
        # p1-p2: Other adversarial vehicles positions
        # pg: Test vehicle position (target)
        # pz: Natural vehicle position
        M = 1560  # Vehicle mass (kg)
        v0 = v0  # Controlled vehicle speed (m/s)
        v1 = v1  # Obstacle vehicle 1 speed (m/s)
        v2 = v2  # Obstacle vehicle 2 speed (m/s)
        vg = vg  # Test vehicle speed (m/s)
        vz = vz  # Natural vehicle speed (m/s)
        Ll = 5  # Vehicle length (m)
        Lw = 2  # Vehicle width (m)
        Lr = 12  # Road width (m)
        Fm = 3000  # Max braking force per wheel (N)
        Pm = 0.5  # Road center hazard potential threshold
        Pt = 0.01  # Hazard potential switching threshold

        # Extract coordinates (Y: longitudinal, X: lateral)
        Y, X = p0[0], p0[1]  # Current vehicle position (long, lat)
        Y0, X0 = p1[0], p1[1]  # Obstacle 1 position
        Y1, X1 = p2[0], p2[1]  # Obstacle 2 position
        Yg, Xg = pg[0], pg[1]  # Test vehicle position
        Yz, Xz = pz[0], pz[1]  # Natural vehicle position
        Ds = 150  # Safety distance threshold (m)

        # Calculate braking distance based on relative speed
        if v1 >= v0:
            Db = 5  # Minimum braking distance when obstacle is faster
        else:
            Db = M * (v0 ** 2 - v1 ** 2) / (8 * Fm) + Ll / 2  # Braking distance formula

        Dt = Db + 10  # Danger threshold distance (m)
        c2 = -np.log(Pt) / (Db ** 2)  # Exponential decay coefficient

        # Calculate hazard potential from obstacle 1
        if abs(Y - Y0) <= Db:
            c1 = -4 * (np.log(Pt) + c2 * (Y - Y0) ** 2) / (
                    (Lw * (np.sin((Y - Y0 - Db) / Db * np.pi - np.pi / 2) + 2)) ** 2 * 0.4
            )
        else:
            c1 = 0

        # Repeat for obstacle 2
        if v2 >= v0:
            Db1 = 5
        else:
            Db1 = M * (v0 ** 2 - v2 ** 2) / (8 * Fm) + Ll / 2
        Dt1 = Db1 + 10
        c21 = -np.log(Pt) / (Db1 ** 2)
        if abs(Y - Y1) <= Db1:
            c11 = -4 * (np.log(Pt) + c21 * (Y - Y1) ** 2) / (
                    (Lw * (np.sin((Y - Y1 - Db1) / Db1 * np.pi - np.pi / 2) + 2)) ** 2 * 0.4
            )
        else:
            c11 = 0

        # Repeat for natural vehicle
        if vz >= v0:
            Dbz = 5
        else:
            Dbz = M * (v0 ** 2 - vz ** 2) / (8 * Fm) + Ll / 2
        Dtz = Dbz + 10
        cz2 = -np.log(Pt) / (Dbz ** 2)
        if abs(Y - Yz) <= Dbz:
            cz1 = -4 * (np.log(Pt) + cz2 * (Y - Yz) ** 2) / (
                    (Lw * (np.sin((Y - Yz - Dbz) / Dbz * np.pi - np.pi / 2) + 2)) ** 2 * 0.4
            )
        else:
            cz1 = 0

        # Road center potential components
        Ax = np.exp(-(6 - abs(X - 6)) ** 2)  # Center bias potential
        Az = np.exp(-(X - 2) ** 2)  # Left lane potential
        Az1 = np.exp(-(X - 6) ** 2)  # Right lane potential

        # Distance-based potential modifiers
        Ay = 0 if abs(Y - Y0) <= Db else (
            1 if abs(Y - Y0) >= Dt else (abs(Y - Y0) - Db) / (Dt - Db)
        )
        Ay1 = 0 if abs(Y - Y1) <= Db1 else (
            1 if abs(Y - Y1) >= Dt1 else (abs(Y - Y1) - Db1) / (Dt1 - Db1)
        )
        Ayz = 0 if abs(Y - Yz) <= Dbz else (
            1 if abs(Y - Yz) >= Dtz else (abs(Y - Yz) - Dbz) / (Dtz - Dbz)
        )
        Pr = Az * Ay * Ay1 * Ayz + Az1 * Ay * Ay1 * Ayz  # Combined road potential

        # Predictive test vehicle position (0.2s ahead)
        ygg = Yg + vg * 0.2
        a = math.sqrt((ygg - Y) ** 2 + (Xg - X) ** 2)  # Euclidean distance to future position

        # Attractive potential field (towards test vehicle)
        if a < 20:
            Ps = a ** 2 / 150  # Quadratic potential within close range
        else:
            Ps = (abs(Yg - Y)) ** 2 / 150  # Linear potential at distance
        if a < 6:  # Clamp minimum potential when very close
            Ps = 6 ** 2 / 150

        # Repulsive potential fields from obstacles
        Po = abs(np.exp(-c1 * (X - X0) ** 2 - c2 * (Y - Y0) ** 2) - Pt) / (1 - Pt)
        P1 = abs(np.exp(-c11 * (X - X1) ** 2 - c21 * (Y - Y1) ** 2) - Pt) / (1 - Pt)
        Pz = abs(np.exp(-cz1 * (X - Xz) ** 2 - cz2 * (Y - Yz) ** 2) - Pt) / (1 - Pt)

        # Total hazard potential (3D virtual field)
        P = P1 + Po + Ps + 0.1 * Pr + Pz  # Combined potential with road bias
        return P

    def _reset(self, num_CAV=0, num_HDV=0) -> None:
        self._make_road()

        cfg_num_cav = self.config.get("num_CAV", 0)
        cfg_num_hdv = self.config.get("num_HDV", 0)
        if num_CAV == 0 and cfg_num_cav:
            num_CAV = int(cfg_num_cav)
        if num_HDV == 0 and cfg_num_hdv:
            num_HDV = int(cfg_num_hdv)

        if self.config["traffic_density"] == 1:
            if num_CAV == 0:
                num_CAV = 4 if self.config["scenario"] == "merge" else 3
            if num_HDV == 0:
                num_HDV = 1

        elif self.config["traffic_density"] == 2:
            if num_CAV == 0:
                num_CAV = np.random.choice(np.arange(2, 5), 1)[0]
            else:
                num_CAV = num_CAV
            if num_HDV == 0:
                num_HDV = np.random.choice(np.arange(2, 5), 1)[0]

        elif self.config["traffic_density"] == 3:
            if num_CAV == 0:
                num_CAV = np.random.choice(np.arange(4, 7), 1)[0]
            else:
                num_CAV = num_CAV
            if num_HDV == 0:
                num_HDV = np.random.choice(np.arange(3, 6), 1)[0]
        self._make_vehicles(num_CAV, num_HDV)
        self.action_is_safe = True
        self.T = int(self.config["duration"] * self.config["policy_frequency"])

        n_agents = len(self.controlled_vehicles)
        self.zuobiao = []
        for i in range(n_agents):
            other_advs = [v for j, v in enumerate(self.controlled_vehicles) if j != i]
            ego_vehicle = getattr(self, "ego_vehicle", None)
            if ego_vehicle is None:
                ego_vehicle = self.road.vehicles[min(len(self.road.vehicles) - 1, max(n_agents, 0))]
            hdv_vehicles = getattr(self, "hdv_vehicles", [])
            hdv_vehicle = hdv_vehicles[0] if hdv_vehicles else self.road.vehicles[-1]
            p1_vehicle = other_advs[0] if len(other_advs) > 0 else ego_vehicle
            p2_vehicle = other_advs[1] if len(other_advs) > 1 else hdv_vehicle

            P = self.weixian(self.controlled_vehicles[i].position, ego_vehicle.position,
                             p1_vehicle.position, p2_vehicle.position, hdv_vehicle.position,
                             self.controlled_vehicles[i].speed, p1_vehicle.speed, p2_vehicle.speed,
                             hdv_vehicle.speed, ego_vehicle.speed)
            self.zuobiao.append(copy.copy(P))

    def _make_road(self, ) -> None:

        net = RoadNetwork()

        # Highway lanes
        ends = [200, 80, 80, 150]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH, 8]
        line_type = [[c, s], [n, s], [n, c]]
        line_type_merge = [[c, s], [n, s], [n, c]]
        for i in range(3):
            net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]))

        if self.config["scenario"] == "merge":
            ramp_y = y[2] + StraightLane.DEFAULT_WIDTH
            net.add_lane("j", "k", StraightLane([ends[0], ramp_y], [ends[0] + ends[1], ramp_y], line_types=[c, s]))
            net.add_lane("k", "b", SineLane([ends[0] + ends[1], ramp_y], [sum(ends[:2]), y[2]],
                                            amplitude=ramp_y - y[2], pulsation=np.pi / (2 * ends[1]), phase=np.pi / 2,
                                            line_types=[c, c]))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

        self.road = road

    def _make_vehicles(self, num_CAV=3, num_HDV=2) -> None:
        """
        Populate the road using a dynamic CAV/HDV composition.

        num_CAV controls adversarial CAVs handled by MAPPO. The ego vehicle is
        always present and is not counted in num_CAV. num_HDV controls natural
        background vehicles sharing the same HDV policy.
        """
        road = self.road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        self.controlled_vehicles = []
        self.non_vehicle = None
        self.ziran_vehicle = []
        self.ego_vehicle = None
        self.hdv_vehicles = []

        num_CAV = max(int(num_CAV), 1)
        num_HDV = max(int(num_HDV), 1)
        main_laterals = [0, 4, 8]
        cav_longitudes = [0, 20, 40, 60, 80, 100]
        hdv_longitudes = [30, 50, 70, 90, 110, 130]
        initial_speed = list(np.random.rand(num_CAV + num_HDV + 1) * 2 + 20)

        ramp_cav_count = 1 if self.config["scenario"] == "merge" and num_CAV > 1 else 0
        main_cav_count = num_CAV - ramp_cav_count

        for i in range(main_cav_count):
            adv_vehicle = self.action_type.vehicle_class(
                road,
                road.network.get_lane(("a", "b", 0)).position(
                    cav_longitudes[i % len(cav_longitudes)],
                    main_laterals[i % len(main_laterals)],
                ),
                speed=initial_speed.pop(0),
            )
            adv_vehicle.role = "adv_cav"
            self.controlled_vehicles.append(adv_vehicle)
            road.vehicles.append(adv_vehicle)

        if ramp_cav_count:
            ramp_adv = self.action_type.vehicle_class(
                road,
                road.network.get_lane(("j", "k", 0)).position(30, 0),
                speed=initial_speed.pop(0),
            )
            ramp_adv.role = "adv_cav"
            self.controlled_vehicles.append(ramp_adv)
            road.vehicles.append(ramp_adv)

        ego_vehicle = other_vehicles_type(
            road,
            road.network.get_lane(("a", "b", 0)).position(
                10,
                main_laterals[(main_cav_count + 1) % len(main_laterals)],
            ),
            speed=20,
        )
        ego_vehicle.role = "ego"
        road.vehicles.append(ego_vehicle)
        self.non_vehicle = ego_vehicle
        self.ego_vehicle = ego_vehicle

        for i in range(num_HDV):
            hdv_vehicle = other_vehicles_type(
                road,
                road.network.get_lane(("a", "b", 0)).position(
                    hdv_longitudes[i % len(hdv_longitudes)],
                    main_laterals[(i + main_cav_count + 2) % len(main_laterals)],
                ),
                speed=initial_speed.pop(0),
            )
            hdv_vehicle.role = "hdv"
            hdv_vehicle.color = (1, 1, 1)
            road.vehicles.append(hdv_vehicle)
            self.ziran_vehicle.append(hdv_vehicle)
            self.hdv_vehicles.append(hdv_vehicle)
    def terminate(self):
        return

    def init_test_seeds(self, test_seeds):
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds


class MergeEnvMARL(MergeEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                    "lateral": True,
                    "longitudinal": True
                }},
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics"
                }},
            "controlled_vehicles": 3
        })
        return config


register(
    id='merge-v1',
    entry_point='highway_env.envs.merge_env_v1:MergeEnv',
)

register(
    id='merge-multi-agent-v0',
    entry_point='highway_env.envs.merge_env_v1:MergeEnvMARL',
)
