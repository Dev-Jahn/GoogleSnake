import os
from argparse import ArgumentParser

import torch
import torchopt
from gym.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import wandb
from wandb.integration.sb3 import WandbCallback

from algs import BMG
from algs.bmg.policy_wrapper import MultiInputActorCriticPolicy
from gsnake.env import GoogleSnakeEnv
from gsnake.configs import GoogleSnakeConfig
from utils import ensure_dir, WandbCallbackWithFood

parser = ArgumentParser()
parser.add_argument('--runname', type=str, help='Name of the run', required=True)
parser.add_argument('--method', choices=['bmg', 'ppo'], help='Training method', required=True)

# Shared arguments
parser.add_argument('--env', type=str, default='GoogleSnake-v1')
parser.add_argument('--n_env', type=int, default=10, help='Number of parallel environments')
parser.add_argument('--n_steps', type=int, default=16, help='Number of steps per rollout')
parser.add_argument('--max_steps', type=int, default=1_000_000, help='Max total steps to train')
parser.add_argument('--batch_size', type=int, default=16, help='Train batch size')
parser.add_argument('--lr', type=float, help='Learning rate for the policy')
parser.add_argument('--multi_channel', action='store_true', help='Use multi-channel observation')
parser.add_argument('--seperate_direction', action='store_true', help='Use seperate direction channel')

# BMG arguments
parser.add_argument('--K', type=int, default=7, help='Number of inner update steps')
parser.add_argument('--L', type=int, default=9, help='Number of bootstrapping steps')
parser.add_argument('--metaoptim', choices=['MetaSGD', 'MetaAdam'], default='MetaSGD', help='Meta-optimizer')
parser.add_argument('--metalr', type=float, default=1e-4, help='Learning rate of outer model')
parser.add_argument('--meta-window-size', type=int, default=10, help='Number of rollouts to use for metalearner input')
parser.add_argument('--momentum', type=float, default=0.0, help='Momentum for MetaSGD')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for MetaSGD and MetaAdam')
parser.add_argument('--dampening', type=float, default=0.0, help='Dampening for MetaSGD')
parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum for MetaSGD')

# Misc
parser.add_argument('--savepath', type=str, default='./ckpt')
parser.add_argument('--device', type=str, default='cuda')


def main(args):
    runpath = os.path.join(args.savepath, args.env, args.runname)
    ensure_dir(os.path.dirname(runpath))
    ensure_dir(runpath)

    # create and save config
    kwargs = build_kwargs(args)
    torch.save(kwargs, os.path.join(runpath, 'vec_env_kwargs.pt'))

    # Parallel environments
    env = make_vec_env(args.env, n_envs=args.n_env, **kwargs)

    runconfig = args.__dict__
    runconfig.update({'envconfig': kwargs['env_kwargs']['config'].__dict__, })
    run = wandb.init(
        job_type='train', config=runconfig,
        project='RL2',
        tags=[args.env.replace('-', '_'), args.method],
        name=args.runname,
        sync_tensorboard=True,
        monitor_gym=False
    )

    if args.method.casefold() == 'bmg':
        policy_kwargs = {
            'normalize_images': False,
        }
        if args.metaoptim.casefold() == 'metasgd':
            if args.lr is None:
                args.lr = 1e-2
            policy_kwargs.update({
                'optimizer_class': torchopt.MetaSGD,
                # Currently not working for some reason
                'optimizer_kwargs': dict(
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    dampening=args.dampening,
                    nesterov=args.nesterov,
                )
            })
        elif args.metaoptim.casefold() == 'metaadam':
            if args.lr is None:
                args.lr = 1e-3
            policy_kwargs.update({
                # Currently not working for some reason
                'optimizer_class': torchopt.MetaAdam,
                'optimizer_kwargs': dict(
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=args.weight_decay,
                    eps_root=0.0,
                    moment_requires_grad=False,
                    use_accelerated_op=False
                )
            })
        model = BMG(
            MultiInputActorCriticPolicy, env, K=args.K, L=args.L,
            meta_window_size=args.meta_window_size,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            meta_learning_rate=args.metalr,
            policy_kwargs=policy_kwargs,
            verbose=0, tensorboard_log=f'runs/{run.id}',
            device=args.device
        )
    elif args.method.casefold() == 'ppo':
        policy_kwargs = {'normalize_images': False}
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            policy_kwargs=policy_kwargs,
            verbose=0, tensorboard_log=f'runs/{run.id}',
            device=args.device
        )
    else:
        raise ValueError(f'Unknown method {args.method}')
    if args.env == 'GoogleSnake-v1':
        callback = WandbCallbackWithFood(model_save_path=runpath, model_save_freq=int(1e+5), verbose=2)
    else:
        callback = WandbCallback(model_save_path=runpath, model_save_freq=int(1e+5), verbose=2)
    model.learn(
        total_timesteps=args.max_steps,
        callback=callback,
        progress_bar=True
    )
    run.finish()


def build_kwargs(args):
    if args.env == 'GoogleSnake-v1':
        config = GoogleSnakeConfig(
            # reward_mode='basic',
            multi_channel=args.multi_channel,
            seperate_direction=args.seperate_direction,
            reward_mode='time_constrained_and_food',
            reward_scale=1,
            n_foods=3
        )
        return dict(
            env_kwargs={'config': config},
            monitor_kwargs=dict(info_keywords=("food_taken",))
        )
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main(parser.parse_args())
