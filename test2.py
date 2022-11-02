from time import sleep

import numpy as np
import gym
from gym.envs.registration import register
from stable_baselines3 import *
from stable_baselines3.common.env_util import make_vec_env
import wandb
from wandb.integration.sb3 import WandbCallback

from gsnake.env import GoogleSnakeEnv
from gsnake.configs import GoogleSnakeConfig

register(
    id='GoogleSnake-v1',
    entry_point=GoogleSnakeEnv,
    max_episode_steps=500,
)

model = PPO.load("PPO_MLP_time.pt")
config = GoogleSnakeConfig(
    # reward_mode='basic',
    multi_channel=True,
    reward_mode='time_constrained',
    reward_scale=1,
    n_foods=3
)
env = GoogleSnakeEnv(config, 42, 'gui')
obs = env.reset()
print('env reset')
try:
    while True:
        print('before pred')
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        sleep(0.5)
except KeyboardInterrupt:
    print('Terminated')
finally:
    env.close()
