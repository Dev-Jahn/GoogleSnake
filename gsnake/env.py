from typing import Optional

import numpy as np
from gym import Env, spaces

from .ui import SnakeGUI, SnakeTUI
from .utils import SnakeState, SnakeGrid, SnakeAction, SnakeNode, SnakeObservation
from .configs import GoogleSnakeConfig


class GoogleSnakeEnv(Env):
    def __init__(self, config: GoogleSnakeConfig = GoogleSnakeConfig(), seed=None, ui=None):
        self.config = config
        self.seed = seed
        self.action_space = spaces.Discrete(n=3)
        self.observation_space = SnakeObservation(self.config.grid_shape, multi_channel=self.config.multi_channel,
                                                  direction_channel=self.config.direction_channel)
        self.state = SnakeGrid(self.config, seed=seed)
        self.food_taken = 0

        assert ui in (None, 'gui', 'tui')
        if ui == 'gui':
            self.ui = SnakeGUI(width=self.config.width, height=self.config.height)
        elif ui == 'tui':
            self.ui = SnakeTUI(width=self.config.width, height=self.config.height)
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
        new_head_pos, new_head_direction = self.state.next_head_position(action)
        if self.config.loop:
            new_head_pos = (new_head_pos[0] % self.config.grid_shape[0], new_head_pos[1] % self.config.grid_shape[1])
        # If the snake is dead, terminate the episode with a negative reward
        if self.state.is_dead(*new_head_pos):
            self.state.reset()
            return self.observation_space.convert(self.state), self.config.DEATH, True, {}

        # Update the internal state grid
        new_head = SnakeNode(*new_head_pos, direction=new_head_direction)

        # Check if the snake eats food
        eat_food = self.state.grid[new_head_pos] == SnakeState.FOOD.value
        dist = self.state.closest_food_dist()
        eat_anti_food = self.state.grid[new_head_pos] == SnakeState.ANTI_FOOD.value

        self.state.remove_apple_node(new_head_pos, eat_food=eat_food, eat_anti_food=eat_anti_food)
        self.state.move(new_head, eat_food=eat_food, eat_anti_food=eat_anti_food)

        # Generate new food
        if eat_food or eat_anti_food:
            if not (eat_anti_food and self.config.poison):
                self.food_taken += 1
                reward += self.config.FOOD
                if self.config.reward_mode == 'time_constrained_and_food':
                    reward += (self.food_taken - 1) * 2
            self.state.generate_food()
            # If wall option is enabled, generate an obstacle every odd number of foods eaten
            if self.config.wall and self.food_taken % 2 == 1:
                self.state.generate_obstacles()
            # head and tail are flipped when snake eats food
            if self.config.reverse:
                self.state.reverse_snake()
        # Distance based reward
        elif self.state.closest_food_dist() - dist < 0:
            reward += self.config.DIST
        elif self.state.closest_food_dist() - dist > 0:
            reward -= self.config.DIST

        if self.config.moving:
            self.state.move_apple()

        return self.observation_space.convert(self.state), reward, False, {'food_taken': self.food_taken}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.state.reset()
        self.food_taken = 0
        return self.observation_space.convert(self.state)

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
                return self.ui.render(self.state)
            elif mode == 'rgb_array':
                raise NotImplementedError
            elif mode == 'ansi':
                raise NotImplementedError

    def close(self):
        if self.ui is not None:
            self.ui.close()
