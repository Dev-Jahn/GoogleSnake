from gsnake.env import GoogleSnakeEnv
from gsnake.configs import GoogleSnakeConfig
from gsnake.utils import SnakeAction
import pygame


def main():
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
                action = SnakeAction.LEFT
            elif event.key == pygame.K_UP:
                action = SnakeAction.FORWARD
            elif event.key == pygame.K_RIGHT:
                action = SnakeAction.RIGHT
            elif event.key == pygame.K_ESCAPE:
                break
            obs, reward, done, info = env.step(action)
            print('reward: ', reward)
            print(info)
            if done:
                print('Game Over')
                env.reset()
            env.render()

    pygame.quit()
    env.close()
    exit(0)


if __name__ == '__main__':
    main()
