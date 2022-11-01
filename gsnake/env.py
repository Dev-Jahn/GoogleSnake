from typing import Optional

import numpy as np
from gym import Env, spaces

from .ui import SnakeGUI, SnakeTUI
from .utils import SnakeState, SnakeGrid, SnakeAction, SnakeNode
from .configs import GoogleSnakeConfig


class GoogleSnakeEnv(Env):
    def __init__(self, config: GoogleSnakeConfig, seed=None, ui=None):
        self.config = config
        self.seed = seed
        self.action_space = spaces.Discrete(n=3)
        self.observation_space = spaces.Box(
            low=1, high=len(SnakeState)+1,
            shape=self.config.grid_shape, dtype=np.uint8
        )
        self.state = SnakeGrid(self.config, seed=seed)
        self.food_taken = 0

        assert ui in (None, 'gui', 'tui')
        if ui == 'gui':
            self.ui = SnakeGUI(width=self.config.width, height=self.config.height)
        elif ui == 'tui':
            self.ui = SnakeTUI()
        else:
            self.ui = None

        self.reset(seed=seed)

    def step(self, action: SnakeAction) -> tuple:
        """
        Take a step in the environment.
        :param action: Integer in range [0, 2] representing the direction of the snake
        :return: Tuple of (observation, reward, done, info)
        """
        reward = self.config.IDLE
        assert self.action_space.contains(action)
        # Get the new head position
        new_head_pos = SnakeAction.absolute_position(self.state.head.direction, action)(*self.state.head.pos)
        # If the snake is dead, terminate the episode with a negative reward
        if self.state.is_dead(*new_head_pos):
            self.state.reset()
            return self.state.grid, self.config.DEATH, True, {}

        # Update the internal state grid
        new_head = SnakeNode(*new_head_pos, direction=SnakeAction.absolute_direction(self.state.head.direction, action))
        # Check if the snake eats food
        eat_food = self.state.grid[new_head_pos] == SnakeState.FOOD.value
        self.state.move(new_head, eat_food=eat_food)
        # Generate new food
        if eat_food:
            self.food_taken += 1
            reward += self.config.FOOD
            self.state.generate_food()
            # If wall option is enabled, generate an obstacle every odd number of foods eaten
            if self.config.wall and self.food_taken % 2 == 1:
                self.state.generate_obstacles()
            # If portal option is enabled, save a portal marker in advance
            if self.config.portal:
                raise NotImplementedError
                # if self.food_taken % 2 == 1:
                #     self.state.generate_portal()
                # else:
                #     self.state.move(self.state.portalmarker, portal=True)
        return self.state.grid, reward, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.state.reset()
        self.food_taken = 0
        return self.state.grid

    def render(self, mode='human'):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with
        """
        if self.ui is not None:
            if mode == 'human':
                self.ui.render(self.state)
            elif mode == 'rgb_array':
                raise NotImplementedError
            elif mode == 'ansi':
                raise NotImplementedError

    def close(self):
        if self.ui is not None:
            self.ui.close()
