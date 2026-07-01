import itertools
from typing import List, Tuple, TYPE_CHECKING
import os
import numpy as np
import pygame

from highway_env.types import Vector
from highway_env.vehicle.dynamics import BicycleVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle, LinearVehicle

if TYPE_CHECKING:
    from highway_env.road.graphics import WorldSurface


class VehicleGraphics(object):
    RED = (255, 100, 100)
    GREEN = (50, 200, 0)
    BLUE = (100, 200, 255)
    YELLOW = (200, 200, 0)
    BLACK = (60, 60, 60)
    PURPLE = (200, 0, 150)
    DEFAULT_COLOR = YELLOW
    EGO_COLOR = BLUE

    @classmethod
    def display(cls, vehicle: Vehicle, surface: "WorldSurface", transparent: bool = False, offscreen: bool = False,
                label: bool = True) -> None:
        """
        Display a vehicle on a pygame surface.

        The vehicle is represented as a colored rotated rectangle.

        :param vehicle: the vehicle to be drawn
        :param surface: the surface to draw the vehicle on
        :param transparent: whether the vehicle should be drawn slightly transparent
        :param offscreen: whether the rendering should be done offscreen or not
        :param label: whether a text label should be rendered
        """
        if not surface.is_visible(vehicle.position):
            return

        v = vehicle
        tire_length, tire_width = 1, 0.3

        # Vehicle rectangle
        length = v.LENGTH + 2 * tire_length
        vehicle_surface = pygame.Surface((surface.pix(length), surface.pix(length)), flags=pygame.SRCALPHA)  # per-pixel alpha
        rect = (surface.pix(tire_length), surface.pix(length / 2 - v.WIDTH / 2), surface.pix(v.LENGTH), surface.pix(v.WIDTH))
        pygame.draw.rect(vehicle_surface, cls.get_color(v, transparent), rect, 0)
        pygame.draw.rect(vehicle_surface, cls.BLACK, rect, 1)

        # Tires
        if type(vehicle) in [Vehicle, BicycleVehicle]:
            tire_positions = [[surface.pix(tire_length), surface.pix(length / 2 - v.WIDTH / 2)],
                              [surface.pix(tire_length), surface.pix(length / 2 + v.WIDTH / 2)],
                              [surface.pix(length - tire_length), surface.pix(length / 2 - v.WIDTH / 2)],
                              [surface.pix(length - tire_length), surface.pix(length / 2 + v.WIDTH / 2)]]
            tire_angles = [0, 0, v.action["steering"], v.action["steering"]]
            for tire_position, tire_angle in zip(tire_positions, tire_angles):
                tire_surface = pygame.Surface((surface.pix(tire_length), surface.pix(tire_length)), pygame.SRCALPHA)
                rect = (0, surface.pix(tire_length/2-tire_width/2), surface.pix(tire_length), surface.pix(tire_width))
                pygame.draw.rect(tire_surface, cls.BLACK, rect, 0)
                cls.blit_rotate(vehicle_surface, tire_surface, tire_position, np.rad2deg(-tire_angle))

        # Centered rotation
        h = v.heading if abs(v.heading) > 2 * np.pi / 180 else 0
        position = [*surface.pos2pix(v.position[0], v.position[1])]
        if not offscreen:
            # convert_alpha throws errors in offscreen mode
            # see https://stackoverflow.com/a/19057853
            vehicle_surface = pygame.Surface.convert_alpha(vehicle_surface)
        cls.blit_rotate(surface, vehicle_surface, position, np.rad2deg(-h))

        # Label
        if label:
            font = pygame.font.Font(None, 20)
            # text = "#{}".format(id(v) % 1000)
            text = "#{}".format(v.id)
            text = font.render(text, 1, (10, 10, 10), (255, 255, 255))
            surface.blit(text, position)

    @staticmethod
    def blit_rotate(surf: pygame.SurfaceType, image: pygame.SurfaceType, pos: Vector, angle: float,
                    origin_pos: Vector = None, show_rect: bool = False) -> None:
        """Many thanks to https://stackoverflow.com/a/54714144."""
        # calculate the axis aligned bounding box of the rotated image
        w, h = image.get_size()
        box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
        box_rotate = [p.rotate(angle) for p in box]
        min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
        max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])

        # calculate the translation of the pivot
        if origin_pos is None:
            origin_pos = w / 2, h / 2
        pivot = pygame.math.Vector2(origin_pos[0], -origin_pos[1])
        pivot_rotate = pivot.rotate(angle)
        pivot_move = pivot_rotate - pivot

        # calculate the upper left origin of the rotated image
        origin = (pos[0] - origin_pos[0] + min_box[0] - pivot_move[0], pos[1] - origin_pos[1] - max_box[1] + pivot_move[1])
        # get a rotated image
        rotated_image = pygame.transform.rotate(image, angle)
        # rotate and blit the image
        surf.blit(rotated_image, origin)
        # draw rectangle around the image
        if show_rect:
            pygame.draw.rect(surf, (255, 0, 0), (*origin, *rotated_image.get_size()), 2)

    @classmethod
    def display_trajectory(cls, states: List[Vehicle], surface: "WorldSurface", offscreen: bool = False) -> None:
        """
        Display the whole trajectory of a vehicle on a pygame surface.

        :param states: the list of vehicle states within the trajectory to be displayed
        :param surface: the surface to draw the vehicle future states on
        :param offscreen: whether the rendering should be done offscreen or not
        """
        for vehicle in states:
            cls.display(vehicle, surface, transparent=True, offscreen=offscreen)

    @classmethod
    def display_history(cls, vehicle: Vehicle, surface: "WorldSurface", frequency: float = 3, duration: float = 2,
                        simulation: int = 15, offscreen: bool = False) -> None:
        """
        Display the whole trajectory of a vehicle on a pygame surface.

        :param vehicle: the vehicle states within the trajectory to be displayed
        :param surface: the surface to draw the vehicle future states on
        :param frequency: frequency of displayed positions in history
        :param duration: length of displayed history
        :param simulation: simulation frequency
        :param offscreen: whether the rendering should be done offscreen or not
        """
        for v in itertools.islice(vehicle.history,
                                  None,
                                  int(simulation * duration),
                                  int(simulation / frequency)):
            cls.display(v, surface, transparent=True, offscreen=offscreen)

    @classmethod
    def get_color(cls, vehicle: Vehicle, transparent: bool = False) -> Tuple[int]:
        color = cls.DEFAULT_COLOR
        if getattr(vehicle, "color", None):
            color = vehicle.color
        elif vehicle.crashed:
            color = cls.RED
        elif isinstance(vehicle, LinearVehicle):
            color = cls.YELLOW
        elif isinstance(vehicle, IDMVehicle):
            color = cls.GREEN
        elif isinstance(vehicle, MDPVehicle):
            color = cls.EGO_COLOR
        if transparent:
            color = (color[0], color[1], color[2], 30)
        return color

class WorldSurface(pygame.Surface):
    BLACK = (0, 0, 0)
    GREY = (100, 100, 100)
    WHITE = (255, 255, 255)

    def __init__(self, size: Tuple[int, int] = None, flags: int = 0, offscreen: bool = False,
                 position: Vector = None, scaling: float = None):
        super().__init__(size or (600, 600), flags)
        self.scaling = scaling if scaling is not None else 10
        self.position = position if position is not None else np.array([300, 300])
        self.offscreen = offscreen

    def pix(self, length: float) -> int:
        return int(self.scaling * length)

    def pos2pix(self, x: float, y: float) -> Tuple[int, int]:
        return self.position + (np.array([x, y]) * self.scaling).astype(int)

    def is_visible(self, position: Vector, margin: int = 50) -> bool:
        pixel_pos = self.pos2pix(*position)
        return (margin < pixel_pos[0] < self.get_width() - margin and
                margin < pixel_pos[1] < self.get_height() - margin)


class EnvViewer(object):

    def __init__(self, env, offscreen: bool = False) -> None:
        self.env = env
        self.offscreen = offscreen
        screen_width = env.config.get("screen_width", 1200)
        screen_height = env.config.get("screen_height", 600)
        centering_position = env.config.get("centering_position", [0.3, 0.5])
        scaling = env.config.get("scaling", 5.5)
        pygame.init()
        if offscreen:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        self.screen_size = (screen_width, screen_height)
        self.surface = WorldSurface(
            size=self.screen_size,
            offscreen=offscreen,
            position=np.array(centering_position) * np.array(self.screen_size),
            scaling=scaling
        )
        if not offscreen:
            pygame.display.set_caption("Highway-env")
            pygame.display.set_mode(self.screen_size)

    def display(self) -> None:
        self.surface.fill(WorldSurface.GREY)
        if hasattr(self.env, 'road') and self.env.road is not None:
            self.draw_road()
        for vehicle in getattr(self.env, 'road', type('', (), {'vehicles': []})).vehicles:
            VehicleGraphics.display(vehicle, self.surface, offscreen=self.offscreen)
        if not self.offscreen:
            pygame.display.flip()

    def draw_road(self) -> None:
        road = self.env.road
        if road is None:
            return
        for _, from_lanes in road.network.graph.items():
            for _, lanes in from_lanes.items():
                for lane in lanes:
                    self._draw_lane(lane)

    def _draw_lane(self, lane) -> None:
        start = lane.start
        end = lane.end
        width = lane.width
        start_pix = self.surface.pos2pix(start[0], start[1])
        end_pix = self.surface.pos2pix(end[0], end[1])
        direction = end_pix[0] - start_pix[0], end_pix[1] - start_pix[1]
        lateral_dir = (-direction[1], direction[0])
        norm_lat = np.linalg.norm(lateral_dir)
        if norm_lat > 0:
            lateral_dir = (lateral_dir[0] / norm_lat, lateral_dir[1] / norm_lat)
        half_w = self.surface.pix(width / 2)
        corners = [
            (start_pix[0] + lateral_dir[0] * half_w, start_pix[1] + lateral_dir[1] * half_w),
            (start_pix[0] - lateral_dir[0] * half_w, start_pix[1] - lateral_dir[1] * half_w),
            (end_pix[0] - lateral_dir[0] * half_w, end_pix[1] - lateral_dir[1] * half_w),
            (end_pix[0] + lateral_dir[0] * half_w, end_pix[1] + lateral_dir[1] * half_w),
        ]
        pygame.draw.polygon(self.surface, WorldSurface.WHITE, corners)
        pygame.draw.polygon(self.surface, WorldSurface.BLACK, corners, 2)

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()

    def get_image(self) -> np.ndarray:
        data = pygame.surfarray.array3d(self.surface).swapaxes(0, 1)
        return data

    def close(self) -> None:
        pygame.quit()