import os
import gym
import sys
sys.path.append("../")

import torch
import torchopt
from gym.envs.registration import register
from stable_baselines3.common.env_util import make_vec_env
import wandb
from wandb.integration.sb3 import WandbCallback

from algs.bmg.bmg_sb3_v2 import BMG
from algs.policies.extractor import MultiInputActorCriticPolicy
#from algs.bmg.policy_wrapper import MultiInputActorCriticPolicy
#from stable_baselines3.common.policies import MultiInputActorCriticPolicy

from gsnake.env import GoogleSnakeEnv
from gsnake.configs import GoogleSnakeConfig
from utils import ensure_dir


def main(n_env=10, K=7, L=9, n_steps=128, max_ep_steps=1000, max_steps=10_000_000, postfix='base', device="cuda"):
    register(
        id='GoogleSnake-v1',
        entry_point=GoogleSnakeEnv,
        max_episode_steps=max_ep_steps,
    )
    rootpath = 'ckpt/GoogleSnake-v1'
    name = f'BMG_max{max_ep_steps}_step{max_steps / 1e+6:.1f}M_{n_env}env_{postfix}'
    savepath = os.path.join(rootpath, name)
    ensure_dir(savepath)

    config = GoogleSnakeConfig(
        # reward_mode='basic',
        multi_channel=True,
        seperate_direction=True,
        reward_mode='time_constrained_and_food',
        reward_scale=1,
        n_foods=3
    )
    torch.save(config, os.path.join(savepath, 'config.pt'))
    run = wandb.init(
        job_type='train', config=config.__dict__,
        project='RL2',
        tags=[name.split('_')[0], 'gsnake'],
        name=name,
        sync_tensorboard=True,
        monitor_gym=False
    )
    # Parallel environments
    env = make_vec_env(
        "GoogleSnake-v1", n_envs=n_env,
        env_kwargs={'config': config},
        monitor_kwargs=dict(info_keywords=("food_taken",))
    )
    policy_kwargs = {
        'normalize_images': False,
        'optimizer_class': torchopt.MetaSGD,
    }
    model = BMG(
        MultiInputActorCriticPolicy, env, K=K, L=L,
        meta_window_size=10,
        learning_rate=3e-4,
        n_steps=n_steps,
        meta_learning_rate=1e-5,
        policy_kwargs=policy_kwargs,
        verbose=0, tensorboard_log=f'runs/{run.id}',
        device=device
    )
    model.learn(
        total_timesteps=max_steps,
        callback=WandbCallback(model_save_path=savepath, verbose=2)
    )
    run.finish()

main()
