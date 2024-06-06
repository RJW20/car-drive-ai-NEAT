import math
from functools import cached_property

from neat import BasePlayer
from car_drive_app.car.base_car import BaseCar
from car_drive_app.cartesians import Vector
from car_drive_app.track.base_track import BaseTrack
from car_drive_app.car import Turn, Acceleration


class Player(BaseCar, BasePlayer):

    def __init__(self, *player_args: dict) -> None:
        super().__init__()
        self.vision: list[float]

    def look_in_direction(self, direction: Vector, origin: Vector, track: BaseTrack) -> float:
        """Return the normalised distance to the edge of the Track from the point origin on the Car.
        
        Normalisation is 3*self.LENGTH (and if no track is found in 3*self.LENGTH steps reutrns 1)
        """

        for i in range(3 * self.LENGTH, -1, -1):
            if not track.check_in_bounds([origin + i * direction]):
                return i / 3 * self.LENGTH
            
    @cached_property
    def MAX_SPEED(self) -> float:
        return (2 * self.POWER * -1 * Acceleration.FORWARD.value)/(self.fl_wheel.RADIUS*self.DRAG_COEFFICIENT)
    
    @property
    def drift_angle(self) -> float:
        """Return the angle between the Car's velocity and its orientation."""
        return self.angle - self.velocity.angle

    def look(self, track: BaseTrack) -> None:
        """Set the Car's vision.
        
        Can see the distance to edge of the Track in front, -+ 30,60,90,135 degrees, and behind (max
        distance is 3*self.LENGTH) as well as own speed, difference in angle between self.velocity
        and self.angle, and the front wheels' turn angle.
        """

        # Get positions on the Car to look from
        half_length = Vector.unit_from_angle(self.angle) * self.LENGTH / 2
        half_width = Vector.unit_from_angle(self.angle + math.pi/2) * self.WIDTH / 2
        front = self.position + half_length
        front_left = front - half_width
        front_right = front + half_width
        back = self.position - half_length
        back_left = back - half_width
        back_right = back + half_width

        # Look by using sonar from different points on the Car
        self.vision = [
            self.look_in_direction(Vector.unit_from_angle(self.angle), front, track),
            self.look_in_direction(Vector.unit_from_angle(self.angle - math.pi/6), front_left, track),
            self.look_in_direction(Vector.unit_from_angle(self.angle + math.pi/6), front_right, track),
            self.look_in_direction(Vector.unit_from_angle(self.angle - math.pi/3), front_left, track),
            self.look_in_direction(Vector.unit_from_angle(self.angle + math.pi/3), front_right, track),
            self.look_in_direction(Vector.unit_from_angle(self.angle - math.pi/2), front_left, track),
            self.look_in_direction(Vector.unit_from_angle(self.angle + math.pi/2), front_right, track),
            self.look_in_direction(Vector.unit_from_angle(self.angle - 3 * math.pi/4), back_left, track),
            self.look_in_direction(Vector.unit_from_angle(self.angle + 3 * math.pi/4), back_right, track),
            self.look_in_direction(Vector.unit_from_angle(self.angle + math.pi), back, track)
        ]

        # Include some of the Car's attributes
        self.vision.append(self.velocity.magnitude / self.MAX_SPEED)
        self.vision.append(self.drift_angle / math.pi)
        self.vision.append(2 * self.fl_wheel.turn_angle / math.pi)   # Max turn angle < 90 degrees

    def think(self) -> tuple[Turn, Acceleration]:
        """Feed the input into the Genome and return the output as a valid move."""

        choices = self.genome.propagate(self.vision)
        choice = max(enumerate(choices), key = lambda choice: choice[1])[0]

        match(choice // 3):
            case 0:
                turn = Turn.STRAIGHT
            case 1:
                turn = Turn.LEFT
            case 2:
                turn = Turn.RIGHT

        match(choice % 3):
            case 0:
                acceration = Acceleration.FORWARD
            case 1:
                acceration = Acceleration.REVERSE
            case 2:
                acceration = Acceleration.NONE

        return turn, acceration