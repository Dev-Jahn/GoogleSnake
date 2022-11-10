import random
from typing import Literal, get_args

from .utils import SnakeState


class GoogleSnakeConfig:
    """
    Class for Game Configurations
    """
    _REWARDS = Literal['basic', 'time_constrained', 'time_constrained_and_food']

    def __init__(self, width=15, height=10, multi_channel=False, direction_channel=False, reward_mode: _REWARDS = 'basic', reward_scale=1.0,
                 n_foods=1, wall=False, portal=False, cheese=False, loop=False, reverse=False, moving=False,
                 yinyang=False, key=False, box=False, poison=False, transparent=False, flag=False, slough=False,
                 peaceful=False, mixed=False, seed=42, *args, **kwargs):

        self.seed = seed
        # width of the game map
        self.width = width
        # height of the game map
        self.height = height

        # whether to use multichannel one-hot or single channel image for observation
        self.multi_channel = multi_channel
        # every direction has its own channel
        self.direction_channel = direction_channel

        # start position and direction of the snake
        self.start_pos = (self.height // 2, self.width // 4)
        self.start_dir = SnakeState.SNAKE_R
        # Count of foods generated at once (V)
        self.n_foods = n_foods if not mixed else random.choice([1, 3, 5])
        # If True, the obstacles are generated every odd number of foods eaten (V)
        self.wall = wall if not mixed else random.choice([True, False])
        # If True, total number of foods doubled and foods work as portals pairwise
        self.portal = portal if not mixed else random.choice([True, False])
        # If True, snake body becomes penetrable alternately
        self.cheese = cheese if not mixed else random.choice([True, False])
        # If True, map boarders are circularly connected
        self.loop = loop if not mixed else random.choice([True, False])
        # If True, head and tail are flipped when snake eats food
        self.reverse = reverse if not mixed else random.choice([True, False])
        # If True, foods move linearly in a random direction
        self.moving = moving if not mixed else random.choice([True, False])
        # If True, generate point symmetric snake body
        self.yinyang = yinyang if not mixed else random.choice([True, False])
        # If True, food is locked and can be eaten only after taking a key
        self.key = key if not mixed else random.choice([True, False])
        # If True, goal is changed to move box to the specific point
        self.box = box if not mixed else random.choice([True, False])
        # If True, generate poison foods with normal foods, which paralyze snake for a while
        self.poison = poison if not mixed else random.choice([True, False])
        # If True, show next apple position and body at that point become penetrable for a while
        self.transparent = transparent if not mixed else random.choice([True, False])
        # If True, eating foods generates a flag obstacle. Passing by it removes it
        self.flag = flag if not mixed else random.choice([True, False])
        # If True, eating foods generates a slough obstacle. Last for a while and will be destroyed by itself
        self.slough = slough if not mixed else random.choice([True, False])
        # If True, do not die with any collision. Reaching certain length will win the game
        self.peaceful = peaceful if not mixed else random.choice([True, False])

        self.reward_mode = reward_mode
        options = get_args(self._REWARDS)
        assert reward_mode in options, f"'{reward_mode}' is not in {options}"
        if reward_mode == 'basic':
            self.DEATH = -10 * reward_scale
            self.FOOD = 100 * reward_scale
            self.IDLE = 0 * reward_scale
            self.DIST = 1 * reward_scale
        elif reward_mode == 'time_constrained':
            self.DEATH = -10 * reward_scale
            self.FOOD = 100 * reward_scale
            self.IDLE = -1 * reward_scale
            self.DIST = 2 * reward_scale
        elif reward_mode == 'time_constrained_and_food':
            self.DEATH = 0
            self.FOOD = (self.height + self.width) * 2
            self.IDLE = -1
            self.DIST = 0


    @property
    def grid_shape(self):
        return self.height, self.width


class GUIConfig:
    # General
    CAPTION = "Google Snake"

    # PATHS
    PATH_ICON = 'snake_logo.png'
    PATH_APPLE = 'apple.png'

    # Pixel sizes of elements
    TILE_W = 50
    TILE_H = 50
    SNAK_W = 30
    SNAK_H = 30
    CONN_W = TILE_W - SNAK_W
    CONN_H = TILE_H - SNAK_H
    LSPACE_W = 5
    LSPACE_H = 5

    # Colors
    LIGHT_BLUE = (71, 117, 235)
    BLACK = (0, 0, 0)
    GREEN1 = (170, 215, 81)
    GREEN2 = (162, 209, 73)


LIGHT_BLUE = (71, 117, 235)


class TUIConfig:
    # General
    CAPTION = "Google Snake"

    # Pixel sizes of elements
    TILE_W = 20
    TILE_H = 20

    # Colors
    BLUE = (0, 0, 255)
    LIGHT_BLUE = (71, 117, 235)
    BLACK = (0, 0, 0)
    GREEN1 = (170, 215, 81)
    GREEN2 = (162, 209, 73)
    RED = (255, 0, 0)

    CMAP = {
        SnakeState.EMPTY: GREEN1,
        **{enum: LIGHT_BLUE for enum in SnakeState if enum.name.startswith('SNAKE')},
        SnakeState.FOOD: RED,
        SnakeState.OBSTACLE: BLACK
    }
