from gym.envs.registration import register
from stable_baselines3 import *
from stable_baselines3.common.env_util import make_vec_env
import wandb
from wandb.integration.sb3 import WandbCallback

from gsnake.env import GoogleSnakeEnv
from gsnake.configs import GoogleSnakeConfig

after_name = 'PPO_2000to5000_max_step_channel_dir_100M'

register(
    id='GoogleSnake-v1',
    entry_point=GoogleSnakeEnv,
    max_episode_steps=5000,
)

config = GoogleSnakeConfig(
    # reward_mode='basic',
    multi_channel=True,
    direction_channel=True,
    reward_mode='time_constrained_and_food',
    reward_scale=1,
    n_foods=3
)

run = wandb.init(
    job_type='train', config=config.__dict__,
    project='RL2',
    tags=[after_name.split('_')[0], 'gsnake'],
    name=after_name,
    sync_tensorboard=True,
    monitor_gym=False
)
env = make_vec_env("GoogleSnake-v1", n_envs=10, env_kwargs={'config':config})
policy_kwargs = {'normalize_images': False}
model = PPO.load("PPO_2000_max_step_channel_dir_50M/model", env = env)
model.learn(total_timesteps=50_000_000, callback=WandbCallback(model_save_path=f'{after_name}', verbose=2))
run.finish()
