import math
from functools import cached_property

from neat import BasePlayer
from car_drive_app.car.base_car import BaseCar
from car_drive_app.cartesians import Vector
from car_drive_app.track.base_track import BaseTrack
from car_drive_app.car import Acceleration


class Player(BaseCar, BasePlayer):

    def __init__(self, *player_args: dict) -> None:
        super().__init__()
        self.vision: list[float]

    def look_in_direction(self, direction: Vector, origin: Vector, track: BaseTrack) -> float:
        """Return the normalised distance to the edge of the Track from the point origin on the Car.
        
        Finds the distance using binary search.
        Normalisation is 3*self.LENGTH (and if no track is found in 3*self.LENGTH/4 steps returns 1).
        """

        # First check max distance
        if track.check_in_bounds([origin + 3 * self.LENGTH * direction]):
            return 1
        
        # Otherwise we know there is a first point thats not in bounds
        l = 0  # noqa: E741
        r = (3 * self.LENGTH - 1) // 4  # step size of 4
        step = 4 * direction

        while l < r:
            m = math.floor((l + r ) / 2)
            if track.check_in_bounds([origin + m * step]):
                l = m + 1  # noqa: E741
            else:
                r = m

        return l * 4 / (3 * self.LENGTH)
    
    @property
    def lidar_rays(self) -> list[Vector]:
        """Return the list of Vectors pointing in all the directions we are searching in.
        
        Returns the directions in the [f, fl, fr, fll, frr, l, r, lb, rb, b]"""

        return [
            Vector.unit_from_angle(self.angle),
            Vector.unit_from_angle(self.angle - math.pi/6),
            Vector.unit_from_angle(self.angle + math.pi/6),
            Vector.unit_from_angle(self.angle - math.pi/3),
            Vector.unit_from_angle(self.angle + math.pi/3),
            Vector.unit_from_angle(self.angle - math.pi/2),
            Vector.unit_from_angle(self.angle + math.pi/2),
            Vector.unit_from_angle(self.angle - 3 * math.pi/4),
            Vector.unit_from_angle(self.angle + 3 * math.pi/4),
            Vector.unit_from_angle(self.angle + math.pi),
        ]
            
    @cached_property
    def MAX_SPEED(self) -> float:
        return (2 * self.POWER * -1 * Acceleration.FORWARD.value)/(self.fl_wheel.RADIUS*self.DRAG_COEFFICIENT)
    
    @property
    def drift_angle(self) -> float:
        """Return the angle between the Car's velocity and its orientation."""
        return self.angle - self.velocity.angle if self.velocity.angle else 0

    def look(self, track: BaseTrack) -> None:
        """Set the Car's vision.
        
        Can see the distance to edge of the Track in front, -+ 30,60,90,135 degrees, and behind (max
        distance is 3*self.LENGTH) as well as own speed, difference in angle between self.velocity
        and self.angle, and the front wheels' turn angle.
        """

        # Get positions on the Car to look from
        half_length = self.direction * self.LENGTH / 2
        half_width = Vector.unit_from_angle(self.angle + math.pi/2) * self.WIDTH / 2
        front = self.position + half_length
        front_left = front - half_width
        front_right = front + half_width
        back = self.position - half_length
        back_left = back - half_width
        back_right = back + half_width

        # Get the directions to look in
        lidar_rays = self.lidar_rays

        # Look by using lidar from different points on the Car
        self.vision = [
            self.look_in_direction(lidar_rays[0], front, track),
            self.look_in_direction(lidar_rays[1], front_left, track),
            self.look_in_direction(lidar_rays[2], front_right, track),
            self.look_in_direction(lidar_rays[3], front_left, track),
            self.look_in_direction(lidar_rays[4], front_right, track),
            self.look_in_direction(lidar_rays[5], front_left, track),
            self.look_in_direction(lidar_rays[6], front_right, track),
            self.look_in_direction(lidar_rays[7], back_left, track),
            self.look_in_direction(lidar_rays[8], back_right, track),
            self.look_in_direction(lidar_rays[9], back, track)
        ]

        # Include some of the Car's attributes
        self.vision.append(self.velocity.magnitude / self.MAX_SPEED)
        self.vision.append(self.drift_angle / math.pi)
        self.vision.append(4 * self.fl_wheel.turn_angle / math.pi)   # Max turn angle is pi/4

    def think(self) -> tuple[float, Acceleration]:
        """Feed the input into the Genome and return the output as a valid move."""

        choices = self.genome.propagate(self.vision)
        acceleration_choice = max(enumerate(choices[0:3]), key=lambda choice: choice[1])[0]

        match acceleration_choice:
            case 0:
                acceleration = Acceleration.NONE
            case 1:
                acceleration = Acceleration.FORWARD
            case 2:
                acceleration = Acceleration.REVERSE

        turn_angle = (choices[3] - 0.5) * math.pi / 2

        return turn_angle, acceleration