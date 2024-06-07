import pygame

from car_drive_app.track.track import Track

from neat import PlaybackPlayers
from neat.settings import settings_handler

from car_drive_ai_neat.playback_player import PlaybackPlayer
from car_drive_ai_neat.settings import settings, simulation_settings


class Playback:
    """Controller of all objects that are present in the Playback.
    
    Switch between generations with the left and right arrow keys.
    Switch between Species with the up and down arrow keys.
    """

    def __init__(
        self,
        playback_folder: str,
        playback_player: type,
        player_args: dict,
        track_save_name: str,
    ) -> None:

        # Load the Track
        self.track = Track.load(track_save_name)

        # Prepare the Cars
        self.players = PlaybackPlayers(playback_folder, playback_player, player_args)
        self.track_positions: list[int]
        self.dead_car_indices: set[int]
        self.new_episode()

        # Pygame set up
        self.dimensions = self.track.dimensions
        pygame.init()
        self.screen = pygame.display.set_mode((self.dimensions.x, self.dimensions.y))
        pygame.display.set_caption("AI Car Drive")
        self.clock = pygame.time.Clock()
        self.font_height = int(0.028 * self.dimensions.y)
        self.stats_font = pygame.font.Font(pygame.font.get_default_font(), self.font_height)

    def new_episode(self) -> None:
        """Place all Cars at the start of the Track and set up tracking of their gate indices
        and if they have died."""

        for car in self.players:
            self.track.place_car_at_start(car)
        self.track_positions = [0 for car in self.players]
        self.dead_car_indices = set()

    def check_event(self) -> None:
        """Check for new user inputs."""

        for event in pygame.event.get():

            if event.type == pygame.QUIT: 
                pygame.quit()
                exit()

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_RIGHT:
                    self.players.generation += 1
                    self.new_episode()
                elif event.key == pygame.K_LEFT:
                    self.players.generation -= 1
                    self.new_episode()

                elif event.key == pygame.K_UP:
                    self.players.species_no += 1
                    self.new_episode()
                elif event.key == pygame.K_DOWN:
                    self.players.species_no -= 1
                    self.new_episode()

                elif event.key == pygame.K_SPACE:
                    self.players.per_species = not self.players.per_species
                    self.new_episode()

    def advance(self) -> None:
        """Advance all Cars to the next frame."""

        for i, (car, track_position) in enumerate(zip(self.players, self.track_positions)):
            if i in self.dead_car_indices:
                continue
            car.look(self.track)
            car.move(*car.think())
            self.track.current_gate_index = track_position
            self.track.update_gate(car)
            if not self.track.check_in_bounds(car.outline):
                self.dead_car_indices.add(i)
            else:
                self.track_positions[i] = self.track.current_gate_index

        if len(self.dead_car_indices) == len(self.track_positions):
            self.new_episode()

    def update_screen(self) -> None:
        """Draw the current frame to the screen."""

        # Wipe the last frame
        self.screen.fill((37,255,0))

        # Draw the Track
        self.track.draw(self.screen)

        for i, car in enumerate(self.players):

            if i in self.dead_car_indices:
                continue

            # Draw the Car
            for point in car.outline:
                pygame.draw.circle(self.screen, 'black', (point.x, point.y), 1)

            # Draw the Car's Wheels
            for w_cen, w_dir in car.wheel_rects:
                start_pos = w_cen + w_dir * 10
                end_pos = w_cen - w_dir * 10
                pygame.draw.line(self.screen, 'black', (start_pos.x, start_pos.y), (end_pos.x, end_pos.y), width=10)

    # Show the gen
        gen = self.stats_font.render(f'Gen: {self.players.generation}', True, 'white')
        gen_rect = gen.get_rect(topleft=(self.font_height, 0.5 * self.font_height))
        self.screen.blit(gen, gen_rect)

        # Show the species_no
        species_no = self.stats_font.render(f'Species: {self.players.species_no + 1}', True, 'white')
        species_no_rect = gen.get_rect(topleft=(self.font_height, 2 * self.font_height))
        self.screen.blit(species_no, species_no_rect)

        # Show the changes
        pygame.display.flip()

    def run(self) -> None:
        """Run the main playback loop."""

        while True:
            self.check_event()
            self.advance()
            self.update_screen()

            self.clock.tick(60)


def playback() -> None:

    handled_settings = settings_handler(settings, silent=True)

    playback_folder = handled_settings['playback_settings']['save_folder']
    player_args = handled_settings['player_args']
    track_save_name = simulation_settings['track_name']
    pb = Playback(playback_folder, PlaybackPlayer, player_args, track_save_name)
    pb.run()