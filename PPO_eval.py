from time import sleep

from gym.envs.registration import register
from stable_baselines3 import *

from gsnake.env import GoogleSnakeEnv
from gsnake.configs import GoogleSnakeConfig

register(
    id='GoogleSnake-v1',
    entry_point=GoogleSnakeEnv,
    max_episode_steps=1000,
)

####################################################################
# Human evaluation
####################################################################
model = PPO.load("PPO_MLP_time_food_only_nch_obsfix_1M.pt")
config = GoogleSnakeConfig(
    # reward_mode='basic',
    multi_channel=True,
    reward_mode='time_constrained',
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
except KeyboardInterrupt:
    print('Terminated')
finally:
    env.close()