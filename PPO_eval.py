from time import sleep

from gym.envs.registration import register
from stable_baselines3 import *

import pygame

from gsnake.env import GoogleSnakeEnv
from gsnake.configs import GoogleSnakeConfig

register(
    id='GoogleSnake-v1',
    entry_point=GoogleSnakeEnv,
    max_episode_steps=5000,
)

####################################################################
# Human evaluation
####################################################################
model = PPO.load("PPO_2000_max_step_nch_50M/model")
config = GoogleSnakeConfig(
    # reward_mode='basic',
    multi_channel=True,
    reward_mode='time_constrained_and_food',
    direction_channel=False,
    n_foods=3
)
env = GoogleSnakeEnv(config, 42, "gui")
obs = env.reset()
try:
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        sleep(0.1)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:     pygame.quit()
except KeyboardInterrupt:
    print('Terminated')
finally:
    env.close()