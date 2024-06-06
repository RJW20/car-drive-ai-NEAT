import neat

from .player import Player
from .settings import settings
from .simulator import simulate


def main() -> None:

    neat.run(
        PlayerClass=Player,
        simulate=simulate,
        settings=settings,
    )


if __name__ == '__main__':
    main()