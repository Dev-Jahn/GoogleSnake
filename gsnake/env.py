from typing import Optional

import numpy as np
from gym import Env, spaces
import pygame

from .utils import SnakeState, SnakeGrid, SnakeAction, SnakeNode
from .configs import GUIConfig, GoogleSnakeConfig


class GoogleSnakeEnv(Env):
    def __init__(self, config: GoogleSnakeConfig, seed=None, ui=None):
        self.config = config
        self.seed = seed
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=SnakeState.EMPTY.value, high=len(SnakeState),
            shape=self.config.grid_shape, dtype=np.uint8
        )
        self.state = SnakeGrid(self.config, seed=seed)
        self.food_taken = 0

        assert ui in (None, 'gui', 'tui')
        if ui == 'gui':
            self.ui = SnakeGUI(self.config.grid_shape)
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
            return self.state.grid, self.config.DEATH, True, None

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
        return self.state.grid, reward, False, None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.food_taken = 0

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
        if mode == 'human':
            self.ui.render(self.state)
        elif mode == 'rgb_array':
            raise NotImplementedError
        elif mode == 'ansi':
            raise NotImplementedError

    def close(self):
        if self.ui is not None:
            self.ui.close()


class SnakeGUI:
    def __init__(self, grid_shape):
        self.grid_shape = grid_shape
        pygame.init()
        self.screen = pygame.display.set_mode(
            np.multiply(grid_shape, (GUIConfig.TILE_W, GUIConfig.TILE_H)))
        self.screen.fill((0, 0, 0))
        pygame.display.set_caption("Google Snake")
        self.image_cache = {
            SnakeState.FOOD: pygame.image.load(GUIConfig.PATH_APPLE),
        }

    def render(self, state):
        self.draw_screen()
        self.draw_foods(state.get_foods())
        self.draw_snake(state.head)

    def draw_foods(self, coords):
        # draw on screen
        for x, y in coords:
            self.screen.blit(self.image_cache[SnakeState.FOOD], (x * GUIConfig.TILE_W, y * GUIConfig.TILE_H))

    def draw_snake(self, head: SnakeNode):
        cursor = head
        while cursor is not None:
            xtile = cursor.row * GUIConfig.TILE_W
            ytile = cursor.col * GUIConfig.TILE_H
            if cursor.direction == SnakeState.SNAKE_U:
                pygame.draw.rect(self.screen, GUIConfig.BLUE,
                                 (xtile + GUIConfig.CONN_W // 2, ytile - GUIConfig.CONN_H // 2,
                                  GUIConfig.SNAK_W, GUIConfig.SNAK_H + GUIConfig.CONN_H))
            if cursor.direction == SnakeState.SNAKE_R:
                pygame.draw.rect(self.screen, GUIConfig.BLUE,
                                 (xtile + GUIConfig.CONN_W // 2, ytile + GUIConfig.CONN_H // 2,
                                  GUIConfig.SNAK_W + GUIConfig.CONN_W, GUIConfig.SNAK_H))
            if cursor.direction == SnakeState.SNAKE_D:
                pygame.draw.rect(self.screen, GUIConfig.BLUE,
                                 (xtile + GUIConfig.CONN_W // 2, ytile + GUIConfig.CONN_H // 2,
                                  GUIConfig.SNAK_W, GUIConfig.SNAK_H + GUIConfig.CONN_H))
            if cursor.direction == SnakeState.SNAKE_L:
                pygame.draw.rect(self.screen, GUIConfig.BLUE,
                                 (xtile - GUIConfig.CONN_W // 2, ytile + GUIConfig.CONN_H // 2,
                                  GUIConfig.SNAK_W + GUIConfig.CONN_W, GUIConfig.SNAK_H))
            # If head
            if cursor.prev_node is None:
                pygame.draw.rect(self.screen, GUIConfig.BLUE,
                                 (xtile + GUIConfig.CONN_W // 2, ytile + GUIConfig.CONN_H // 2,
                                  GUIConfig.SNAK_W, GUIConfig.SNAK_H))
                # draw eyes
                eh, ew = 9, 6
                if cursor.direction in [SnakeState.SNAKE_R, SnakeState.SNAKE_L]:
                    pygame.draw.rect(self.screen, GUIConfig.BLACK,
                                     (xtile + GUIConfig.TILE_W // 2 + eh * (cursor.direction // 2 - 2),
                                      ytile + GUIConfig.TILE_H // 2 - (ew * 3 // 2), eh, ew))
                    pygame.draw.rect(self.screen, GUIConfig.BLACK,
                                     (xtile + GUIConfig.TILE_W // 2 + eh * (cursor.direction // 2 - 2),
                                      ytile + GUIConfig.TILE_H // 2 + (ew // 2), eh, ew))
                else:
                    pygame.draw.rect(self.screen, GUIConfig.BLACK,
                                     (xtile + GUIConfig.TILE_W // 2 - (ew * 3 // 2),
                                      ytile + GUIConfig.TILE_H // 2 - eh * (cursor.direction // 2),
                                      ew, eh))
                    pygame.draw.rect(self.screen, GUIConfig.BLACK,
                                     (xtile + GUIConfig.TILE_W // 2 + (ew // 2),
                                      ytile + GUIConfig.TILE_H // 2 - eh * (cursor.direction // 2),
                                      ew, eh))
            cursor = cursor.next_node

    def draw_screen(self):
        self.screen.fill(GUIConfig.GREEN1)
        for col in range(self.grid_shape[0]):
            for row in range(col % 2, self.grid_shape[1], 2):
                pygame.draw.rect(self.screen, GUIConfig.GREEN2,
                                 (col * GUIConfig.TILE_W, row * GUIConfig.TILE_H,
                                  GUIConfig.TILE_W, GUIConfig.TILE_H))

    def close(self):
        pygame.display.quit()
        pygame.quit()


class SnakeTUI:
    def __init__(self):
        raise NotImplementedError
