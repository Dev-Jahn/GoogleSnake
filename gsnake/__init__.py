from gym.envs.registration import register

from gsnake.env import GoogleSnakeEnv
from gsnake.configs import GoogleSnakeConfig

register(
    id='GoogleSnake-v1',
    entry_point=GoogleSnakeEnv,
    max_episode_steps=1000,
)

__all__ = ['GoogleSnakeEnv', 'GoogleSnakeConfig']
