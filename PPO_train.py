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
    max_episode_steps=5000,
)

name = '5000_dir_channel_50M'

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
    tags=[name.split('_')[0], 'gsnake'],
    name=name,
    sync_tensorboard=True,
    monitor_gym=False
)
# Parallel environments
env = make_vec_env("GoogleSnake-v1", n_envs=10, env_kwargs={'config':config})
policy_kwargs = {'normalize_images': False}
model = PPO(
    "MultiInputPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=0, tensorboard_log=f'runs/{run.id}')

# for i in tqdm(range(10)):
model.learn(total_timesteps=1_000_000, callback=WandbCallback(model_save_path=f'{name}', verbose=2))
run.finish()
