import argparse
import traceback

import pygame

from gsnake.env import GoogleSnakeEnv
from gsnake.configs import GoogleSnakeConfig
from gsnake.utils import SnakeAction

parser = argparse.ArgumentParser()

# Basics
parser.add_argument("--width", default=15, help="Width of the game map")
parser.add_argument("--height", default=10, help="Height of the game map")
parser.add_argument("--multi_channel", action="store_true", default=False, help="Use multichannel one-hot observation")

# Reproducibility
parser.add_argument("--seed", default=42, help="Random seed")

# Game settings
parser.add_argument("--n_foods", default=5, help="Number of foods generated")
parser.add_argument("--wall", action="store_true", help="Generate wall if snake eats food")
parser.add_argument("--portal", action="store_true", help="Foods work as portals pairwise")
parser.add_argument("--cheese", action="store_true", help="")
parser.add_argument("--loop", action="store_true", help="Make the map boundary loop")
parser.add_argument("--reverse", action="store_true", help="Reverse the snake if snake eats food")
parser.add_argument("--moving", action="store_true", help="Food moves")
parser.add_argument("--yinyang", action="store_true", help="Put another snake symmetric moves")
parser.add_argument("--key", action="store_true", help="Food is locked and can be eaten only after taking a key")
parser.add_argument("--box", action="store_true", help="Sokoban mode")
parser.add_argument("--poison", action="store_true", help="Generate poison food with normal food")
parser.add_argument("--transparent", action="store_true", help="Body cells become penetrable for a while")
parser.add_argument("--flag", action="store_true", help="Eating foods generates a flag obstacle")
parser.add_argument("--slough", action="store_true", help="Eating foods generates a slough obstacle")
parser.add_argument("--peaceful", action="store_true", help="Snake Never dies")

# Render options
parser.add_argument("--ui", choices=['gui', 'tui', 'rgb_array', 'ansi'], help="Render methods")
# Development
parser.add_argument("--debug", action="store_true", help="Debug mode")

args = parser.parse_args()

if __name__ == '__main__':
    pygame.init()
    config = GoogleSnakeConfig(**dict(args._get_kwargs()))
    env = GoogleSnakeEnv(config, seed=args.seed, ui=args.ui)
    env.reset()
    env.render()
    done = False
    try:
        while not done:
            action = None
            while action is None:
                pygame.event.clear()
                event = pygame.event.wait()
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = SnakeAction.LEFT
                    elif event.key == pygame.K_RIGHT:
                        action = SnakeAction.RIGHT
                    elif event.key == pygame.K_UP:
                        action = SnakeAction.FORWARD
                    elif event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt
            obs, reward, done, info = env.step(action)
            print(obs)
            print(reward)
            env.render()
            if done:
                print("Game Over")
                break
    except KeyboardInterrupt:
        print('User terminated the game')
    except Exception:
        traceback.print_exc()
    finally:
        env.close()
