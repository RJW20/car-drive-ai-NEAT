import sys
import math

import pygame

from car_drive_app.game import Game
from car_drive_app.track.track import Track
from car_drive_app.cartesians import Vector
from car_drive_ai_neat.playback_player import PlaybackPlayer


class Vision(Game):
    """Runs the main game whilst also drawing the Car's vision."""

    def __init__(self, track_save_name: str) -> None:

        # Load the Track
        self.track = Track.load(track_save_name)

        # Start the Car
        self.car = PlaybackPlayer()
        self.track.place_car_at_start(self.car)
        
        # Pygame set up
        self.dimensions = self.track.dimensions
        pygame.init()
        self.screen = pygame.display.set_mode((self.dimensions.x, self.dimensions.y))
        pygame.display.set_caption("Car Drive")
        self.clock = pygame.time.Clock()
        try:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
        except pygame.error:
            raise Exception('No Controller found')
        
    def update_screen(self) -> None:
        """Draw the current frame to the screen."""

        # Wipe the last frame
        self.screen.fill((37,255,0))

        # Draw the Track
        self.track.draw(self.screen)

        # Draw the Car
        for point in self.car.outline:
            pygame.draw.circle(self.screen, 'black', (point.x, point.y), 1)

        # Draw the Car's Wheels
        for w_cen, w_dir in self.car.wheel_rects:
            start_pos = w_cen + w_dir * 10
            end_pos = w_cen - w_dir * 10
            pygame.draw.line(self.screen, 'black', (start_pos.x, start_pos.y), (end_pos.x, end_pos.y), width=10)

        # Draw the Car's vision
        half_length = self.car.direction * self.car.LENGTH / 2
        half_width = Vector.unit_from_angle(self.car.angle + math.pi/2) * self.car.WIDTH / 2
        front = self.car.position + half_length
        front_left = front - half_width
        front_right = front + half_width
        back = self.car.position - half_length
        back_left = back - half_width
        back_right = back + half_width
        lidar_rays = self.car.lidar_rays

        starts = [front, front_left, front_right, front_left, front_right, front_left, front_right, back_left, back_right, back]
        for i, start in enumerate(starts):
            max_sight = start + self.car.LENGTH * 3 * lidar_rays[i]
            lidar = start + self.car.LENGTH * 3 * self.car.vision[i] * lidar_rays[i]
            pygame.draw.line(self.screen, 'red', (start.x, start.y), (max_sight.x, max_sight.y), width=3)
            pygame.draw.line(self.screen, 'blue', (start.x, start.y), (lidar.x, lidar.y), width=3)

        # Show the changes
        pygame.display.flip()

    def run(self) -> None:
        """Run the main game loop."""

        while True:
            self.advance(*self.check_move())
            self.car.look(self.track)
            self.update_screen()

            self.clock.tick(60)


if __name__ == '__main__':
    track_save_name = sys.argv[1]
    game = Vision(track_save_name)
    game.run()