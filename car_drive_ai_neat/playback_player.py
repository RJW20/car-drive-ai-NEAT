from car_drive_ai_neat.player import Player
from car_drive_app.cartesians import Vector


class PlaybackPlayer(Player):
    """Car with Genome and drawable properties/methods."""

    @property
    def wheel_rects(self) -> list[tuple[Vector, Vector]]:
        """Return a list of the Wheel centers and directions."""

        result = []

        for wheel in self.front_wheels:
            w_center = self.position + self.relative_to_world(wheel.offset)
            w_direction = Vector.unit_from_angle(self.angle + wheel.turn_angle)
            result.append((w_center, w_direction))

        for wheel in self.back_wheels:
            w_center = self.position + self.relative_to_world(wheel.offset)
            w_direction = Vector.unit_from_angle(self.angle)
            result.append((w_center, w_direction))

        return result