from time import sleep

import numpy as np
import gym
from gym.envs.registration import register
from stable_baselines3 import *
from stable_baselines3.common.env_util import make_vec_env

from gsnake.env import GoogleSnakeEnv
from gsnake.configs import GoogleSnakeConfig
import pygame

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
    n_foods=3, wall=True, portal=False, cheese=False, loop=False, reverse=False, moving=False,
    yinyang=False, key=False, box=False, poison=False, transparent=False, flag=False, slough=False,
    peaceful=False, mixed=False
)

env = GoogleSnakeEnv(config, 42, 'gui')
obs = env.reset()
print('env reset')
env.render()

while True:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:  env.step(0)
            if event.key == pygame.K_UP:    env.step(1)
            if event.key == pygame.K_RIGHT: env.step(2)
            if event.key == pygame.K_q:     pygame.quit()
            env.render()

env.close()
