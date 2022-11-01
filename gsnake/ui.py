import numpy as np
import pygame
from pygame import Rect

from gsnake.configs import GUIConfig
from gsnake.utils import SnakeState, SnakeNode


class SnakeGUI:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        pygame.init()
        self.screen = pygame.display.set_mode(
            np.multiply((self.width, self.height), (GUIConfig.TILE_W, GUIConfig.TILE_H)))
        self.screen.fill((0, 0, 0))
        pygame.display.set_caption("Google Snake")
        self.image_cache = {
            SnakeState.FOOD: pygame.transform.scale(pygame.image.load(GUIConfig.PATH_APPLE),
                                                    (GUIConfig.TILE_W, GUIConfig.TILE_H)),
        }

    def render(self, state):
        self.draw_screen()
        self.draw_foods(state.get_foods())
        self.draw_snake(state.head)
        pygame.display.update()

    def draw_foods(self, coords):
        # draw on screen
        for row, col in coords:
            self.screen.blit(self.image_cache[SnakeState.FOOD], (col * GUIConfig.TILE_W, row * GUIConfig.TILE_H))

    def draw_snake(self, head: SnakeNode):
        cursor = head
        while cursor is not None:
            xtile = cursor.col * GUIConfig.TILE_W
            ytile = cursor.row * GUIConfig.TILE_H

            # If head
            if cursor.prev_node is None:
                self._draw_head(cursor)
            # elif cursor.next_node is None:
            #     self._draw_tail(cursor)
            else:
                self._draw_body(cursor)

            cursor = cursor.next_node

    def _draw_head(self, node):
        xtile = node.col * GUIConfig.TILE_W
        ytile = node.row * GUIConfig.TILE_H

        left = xtile + GUIConfig.CONN_W // 2 * (node.direction != SnakeState.SNAKE_R)
        top = ytile + GUIConfig.CONN_H // 2 * (node.direction != SnakeState.SNAKE_D)
        width = GUIConfig.SNAK_W + GUIConfig.CONN_W // 2 * (node.direction in [SnakeState.SNAKE_L, SnakeState.SNAKE_R])
        height = GUIConfig.SNAK_H + GUIConfig.CONN_H // 2 * (node.direction in [SnakeState.SNAKE_U, SnakeState.SNAKE_D])

        pygame.draw.rect(surface=self.screen,
                         color=GUIConfig.BLUE,
                         rect=Rect(left, top, width, height))
        # draw eyes
        eh, ew = 9, 6
        if node.direction in [SnakeState.SNAKE_R, SnakeState.SNAKE_L]:
            pygame.draw.rect(surface=self.screen,
                             color=GUIConfig.BLACK,
                             rect=Rect(xtile + GUIConfig.TILE_W // 2 + eh * (node.direction // 2 - 2),
                                       ytile + GUIConfig.TILE_H // 2 - (ew * 3 // 2),
                                       eh,
                                       ew))
            pygame.draw.rect(surface=self.screen,
                             color=GUIConfig.BLACK,
                             rect=Rect(xtile + GUIConfig.TILE_W // 2 + eh * (node.direction // 2 - 2),
                                       ytile + GUIConfig.TILE_H // 2 + (ew // 2),
                                       eh,
                                       ew))
        else:
            pygame.draw.rect(surface=self.screen,
                             color=GUIConfig.BLACK,
                             rect=Rect(xtile + GUIConfig.TILE_W // 2 - (ew * 3 // 2),
                                       ytile + GUIConfig.TILE_H // 2 - eh * (node.direction // 2),
                                       ew,
                                       eh))
            pygame.draw.rect(surface=self.screen,
                             color=GUIConfig.BLACK,
                             rect=Rect(xtile + GUIConfig.TILE_W // 2 + (ew // 2),
                                       ytile + GUIConfig.TILE_H // 2 - eh * (node.direction // 2),
                                       ew,
                                       eh))

    def _draw_body(self, node):
        xtile = node.col * GUIConfig.TILE_W
        ytile = node.row * GUIConfig.TILE_H

        # left = xtile if node.direction in [SnakeState.SNAKE_L, SnakeState.SNAKE_R] else xtile + GUIConfig.CONN_W // 2
        # top = ytile if node.direction in [SnakeState.SNAKE_U, SnakeState.SNAKE_D] else ytile + GUIConfig.CONN_H // 2
        # width = GUIConfig.TILE_W if node.direction in [SnakeState.SNAKE_R, SnakeState.SNAKE_L] else GUIConfig.SNAK_W
        # height = GUIConfig.TILE_H if node.direction in [SnakeState.SNAKE_U, SnakeState.SNAKE_D] else GUIConfig.SNAK_H
        #
        # pygame.draw.rect(surface=self.screen,
        #                  color=GUIConfig.BLUE,
        #                  rect=Rect(left, top, width, height))

        # if node.direction in (SnakeState.SNAKE_U, SnakeState.SNAKE_D):
        #     left += GUIConfig.CONN_W // 2
        #     width -= GUIConfig.CONN_W

        if node.direction == SnakeState.SNAKE_U:
            pygame.draw.rect(surface=self.screen,
                             color=GUIConfig.BLUE,
                             rect=Rect(xtile + GUIConfig.CONN_W // 2,
                                       ytile - GUIConfig.CONN_H // 2,
                                       GUIConfig.SNAK_W,
                                       GUIConfig.SNAK_H + GUIConfig.CONN_H))
        if node.direction == SnakeState.SNAKE_R:
            pygame.draw.rect(surface=self.screen,
                             color=GUIConfig.BLUE,
                             rect=Rect(xtile + GUIConfig.CONN_W // 2,
                                       ytile + GUIConfig.CONN_H // 2,
                                       GUIConfig.SNAK_W + GUIConfig.CONN_W,
                                       GUIConfig.SNAK_H))
        if node.direction == SnakeState.SNAKE_D:
            pygame.draw.rect(surface=self.screen,
                             color=GUIConfig.BLUE,
                             rect=Rect(xtile + GUIConfig.CONN_W // 2,
                                       ytile + GUIConfig.CONN_H // 2,
                                       GUIConfig.SNAK_W,
                                       GUIConfig.SNAK_H + GUIConfig.CONN_H))
        if node.direction == SnakeState.SNAKE_L:
            pygame.draw.rect(surface=self.screen,
                             color=GUIConfig.BLUE,
                             rect=Rect(xtile - GUIConfig.CONN_W // 2,
                                       ytile + GUIConfig.CONN_H // 2,
                                       GUIConfig.SNAK_W + GUIConfig.CONN_W,
                                       GUIConfig.SNAK_H))

    def _draw_tail(self, node):
        pass

    def draw_screen(self):
        self.screen.fill(GUIConfig.GREEN1)
        for col in range(self.width):
            for row in range(col % 2, self.height, 2):
                pygame.draw.rect(self.screen, GUIConfig.GREEN2,
                                 (col * GUIConfig.TILE_W, row * GUIConfig.TILE_H,
                                  GUIConfig.TILE_W, GUIConfig.TILE_H))

    def close(self):
        pygame.display.quit()
        pygame.quit()


class SnakeTUI:
    def __init__(self):
        raise NotImplementedError
