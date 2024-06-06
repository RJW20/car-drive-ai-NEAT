import neat

from car_drive_ai_neat.player import Player
from car_drive_ai_neat.settings import settings
from car_drive_ai_neat.simulator import simulate


def main() -> None:

    neat.run(
        PlayerClass=Player,
        simulate=simulate,
        settings=settings,
    )


if __name__ == '__main__':
    main()