import time

from gym.envs.registration import register

from gsnake.env import GoogleSnakeEnv
from gsnake.configs import GoogleSnakeConfig
import pygame


def main():
    register(
        id='GoogleSnake-v1',
        entry_point=GoogleSnakeEnv,
        max_episode_steps=500,
    )

    config = GoogleSnakeConfig(
        # reward_mode='basic',
        multi_channel=True,
        reward_mode='time_constrained',
        reward_scale=1,
        n_foods=3, wall=True, portal=True, cheese=False, loop=True, reverse=False, moving=True,
        yinyang=False, key=False, box=False, poison=True, transparent=False, flag=False, slough=False,
        peaceful=False, mixed=False
    )

    env = GoogleSnakeEnv(config, 42, 'gui')
    env.reset()
    env.render()

    while True:
        pygame.event.clear()
        event = pygame.event.wait()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                env.step(0)
            elif event.key == pygame.K_UP:
                env.step(1)
            elif event.key == pygame.K_RIGHT:
                env.step(2)
            elif event.key == pygame.K_ESCAPE:
                break
            env.render()

    pygame.quit()
    env.close()
    exit(0)


if __name__ == '__main__':
    main()
