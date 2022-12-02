import sys
sys.path.append("../../../../..")

from functools import reduce
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import torchopt
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
import matplotlib.pyplot as plt
from gsnake.env import GoogleSnakeEnv
from gsnake.configs import GoogleSnakeConfig

from gym.envs.registration import register
from stable_baselines3.common.env_util import make_vec_env

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

class ActorCritic(nn.Module):

    def __init__(self, input_channel, height, width, input_node, n_actions, alpha, fc0_dims=2560, fc1_dims=1024, fc2_dims=512, gamma=0.99):
        super(ActorCritic, self).__init__()
        self.chkpt_file = os.path.join("todo" + '_bmg')

        hidden_channel = [16, 32, 64, 64]
        hidden_nodes = [256, 256, 256]

        self.grid_convolution = nn.Sequential(
            nn.Conv2d(input_channel, hidden_channel[0], kernel_size=1),
            nn.Conv2d(hidden_channel[0], hidden_channel[1], kernel_size=5, groups=hidden_channel[0], padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channel[1], hidden_channel[1], kernel_size=1),
            nn.Conv2d(hidden_channel[1], hidden_channel[2], kernel_size=5, groups=hidden_channel[1], padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channel[2], hidden_channel[2], kernel_size=1),
            nn.Conv2d(hidden_channel[2], hidden_channel[3], kernel_size=5, groups=hidden_channel[2], padding=1),
            nn.ReLU()
        )

        self.node_linear = nn.Sequential(
            nn.Linear(input_node, hidden_nodes[0]),
            nn.Linear(hidden_nodes[0], hidden_nodes[1]),
            nn.Linear(hidden_nodes[1], hidden_nodes[2])
        )

        self.pi1 = nn.Linear(fc0_dims, fc1_dims)
        self.v1 = nn.Linear(fc0_dims, fc1_dims)
        self.pi2 = nn.Linear(fc1_dims, fc2_dims)
        self.v2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)
        self.v = nn.Linear(fc2_dims, 1)

        self.optim = torchopt.MetaSGD(self, lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, grid, nodes):

        x1 = T.flatten(self.grid_convolution(grid), start_dim=1)
        x2 = self.node_linear(nodes)

        x = T.cat((x1, x2), dim=1)

        pi = F.relu(self.pi1(x))
        v = F.relu(self.v1(x))

        pi = F.relu(self.pi2(pi))
        v = F.relu(self.v2(v))

        pi = self.pi(pi)
        v = self.v(v)

        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)

        return dist, v

    def choose_action(self, grid, nodes):
        dist, v = self.forward(grid, nodes)
        action = dist.sample().numpy()[0]

        return action

    def save_checkpoint(self):
        print("...Saving Checkpoint...")
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        print("...Loading Checkpoint...")
        self.load_state_dict(T.load(self.chkpt_file))


class MetaMLP(nn.Module):
    def __init__(self, alpha, betas=(0.9, 0.999), eps=1e-4, input_dims=10, fc1_dims=64):
        super(MetaMLP, self).__init__()
        self.chkpt_file = os.path.join("todo" + '_bmg')

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, 1)

        self.optim = T.optim.Adam(self.parameters(), lr=alpha, betas=betas, eps=eps)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = T.sigmoid(self.fc2(out))
        return out

    def save_checkpoint(self):
        print("...Saving Checkpoint...")
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        print("...Loading Checkpoint...")
        self.load_state_dict(T.load(self.chkpt_file))


class Agent:
    def __init__(self, input_channel, height, width, input_node, n_actions, gamma, alpha, m_alpha, betas, eps, name,
                 env, steps, K_steps, L_steps, rollout_steps, random_seed):
        super(Agent, self).__init__()

        self.actorcritic = ActorCritic(input_channel, height, width, input_node, n_actions, alpha)
        self.ac_k = ActorCritic(input_channel, height, width, input_node, n_actions, alpha)
        self.meta_mlp = MetaMLP(m_alpha, betas, eps, input_dims=10, fc1_dims=64)

        self.node_name = ['head_col', 'head_row', 'head_direction', 'tail_col', 'tail_row', 'tail_direction', 'portal_row', 'portal_col']

        self.env = env
        self.name = f"agent_{name}"
        self.n_actions = n_actions
        self.input_channel = input_channel
        self.height = height
        self.width = width
        self.input_node = input_node
        self.steps = steps
        self.K_steps = K_steps
        self.L_steps = L_steps
        self.rollout_steps = rollout_steps
        self.random_seed = random_seed
        self.gamma = gamma

        # stats
        self.avg_reward = [0 for _ in range(10)]
        self.accum_reward = 0
        self.cum_reward = []
        self.entropy_rate = []

    def rollout(self, bootstrap=False):
        log_probs, values, rewards, masks, states = [], [], [], [], []
        rollout_reward, entropy = 0, 0
        obs = self.env.reset()
        done = False
        for _ in range(self.rollout_steps):
            grid_obs = T.tensor(obs['grid'], dtype=T.float)
            node_obs = T.cat(tuple([T.tensor(obs[name], dtype=T.float) for name in self.node_name]), dim=1)
            dist, v = self.actorcritic(grid_obs, node_obs)

            action = dist.sample()

            obs_, reward, done, _ = self.env.step(action.cpu().numpy())

            log_prob = dist.log_prob(action)
            entropy += -dist.entropy()

            states.append(obs)
            values.append(v)
            log_probs.append(log_prob.unsqueeze(0).to(self.actorcritic.device))
            rewards.append(T.tensor([reward]).to(self.actorcritic.device))

            # non-episodic, (i.e use all rewards)
            # masks.append(T.tensor([1-int(done)], dtype=T.float).to(self.actorcritic.device))
            # rollout_reward += reward*(1-int(done))
            rollout_reward += reward[0]
            self.accum_reward += reward
            self.cum_reward.append(self.accum_reward)

            obs = obs_

            # No need, since non-episodic
            '''
            if done:
                break
            '''

        grid_obs = T.tensor(obs_['grid'], dtype=T.float)
        node_obs = T.cat(tuple([T.tensor(obs_[name], dtype=T.float) for name in self.node_name]), dim=1)
        _, v = self.actorcritic(grid_obs, node_obs)

        # Calc discounted returns
        R = v
        discounted_returns = []
        for step in reversed(range(len(rewards))):
            # R = rewards[step] + self.gamma * R * masks[step]
            R = rewards[step] + self.gamma * R
            discounted_returns.append(R)
        discounted_returns.reverse()

        self.avg_reward = self.avg_reward[1:]
        self.avg_reward.append(rollout_reward / self.rollout_steps)
        ar = T.tensor(self.avg_reward, dtype=T.float).to(self.actorcritic.device)
        eps_en = self.meta_mlp(ar)

        entropy = entropy / self.rollout_steps
        self.entropy_rate.append(eps_en.item())

        log_probs = T.cat(log_probs)
        values = T.cat(values)
        returns = T.cat(discounted_returns)
        advantage = returns - values

        # Compute losses
        actor_loss = -T.mean((log_probs * advantage.detach()))
        critic_loss = 0.5 * T.mean(advantage.pow(2))

        if bootstrap:
            return actor_loss, states
        else:
            return actor_loss + critic_loss + eps_en * entropy

    def kl_matching_function(self, ac_k, tb, states, ac_k_state_dict):
        with T.no_grad():
            dist_tb = []
            for i in range(len(states)):
                grid_obs = T.tensor(states[i]['grid'], dtype=T.float)
                node_obs = T.cat(tuple([T.tensor(states[i][name], dtype=T.float) for name in self.node_name]), dim=1)
                dist_tb.append(tb(grid_obs, node_obs)[0])

        torchopt.recover_state_dict(ac_k, ac_k_state_dict)

        dist_k = []
        for i in range(len(states)):
            grid_obs = T.tensor(states[i]['grid'], dtype=T.float)
            node_obs = T.cat(tuple([T.tensor(states[i][name], dtype=T.float) for name in self.node_name]), dim=1)
            dist_k.append(ac_k(grid_obs, node_obs)[0])

        # KL Div between dsitributions of TB and AC_K, respectively
        kl_div = sum([kl_divergence(dist_tb[i], dist_k[i]) for i in range(len(states))])

        return kl_div

    def plot_results(self):

        cr = plt.figure(figsize=(10, 10))
        plt.plot(self.cum_reward)
        plt.xlabel('Steps')
        plt.ylabel('Cumulative Reward')
        plt.savefig('res/cumulative_reward')
        plt.close(cr)

        er = plt.figure(figsize=(10, 10))
        plt.plot(list(range(1, 18_750 * 16 + 1, 16)), self.entropy_rate[-18_750:])
        plt.xlabel('Steps (Last 300,000 of 4.8M steps)')
        plt.ylabel('Entropy Rate')
        plt.savefig('res/entropy_rate')
        plt.close(er)

    def run(self):

        outer_range = self.steps // self.rollout_steps
        outer_range = outer_range // (self.K_steps + self.L_steps)
        ct = 0

        count = 0
        print(outer_range)
        for _ in range(outer_range):
            count += 1
            print(count)

            for _ in range(self.K_steps):
                loss = self.rollout()
                self.actorcritic.optim.step(loss)
            k_state_dict = torchopt.extract_state_dict(self.actorcritic)

            for _ in range(self.L_steps - 1):
                loss = self.rollout()
                self.actorcritic.optim.step(loss)
            k_l_m1_state_dict = torchopt.extract_state_dict(self.actorcritic)

            bootstrap_loss, states = self.rollout(bootstrap=True)
            self.actorcritic.optim.step(bootstrap_loss)

            # KL-Div Matching loss
            kl_matching_loss = self.kl_matching_function(self.ac_k, self.actorcritic, states, k_state_dict)

            # MetaMLP update
            self.meta_mlp.optim.zero_grad()
            kl_matching_loss.backward()
            self.meta_mlp.optim.step()

            # Use most recent params and stop grad
            torchopt.recover_state_dict(self.actorcritic, k_l_m1_state_dict)
            torchopt.stop_gradient(self.actorcritic)
            torchopt.stop_gradient(self.actorcritic.optim)

            ct += self.rollout_steps * ((self.K_steps + self.L_steps))

            # print stats
            if ct % 1000 == 0:
                print(f"CR and ER, step# {ct}:")
                print(self.cum_reward[-1])
                print(self.entropy_rate[-1])
                print("###")

    def save_models(self):
        self.actorcritic.save_checkpoint()
        self.meta_mlp.save_checkpoint()

    def load_models(self):
        self.actorcritic.load_checkpoint()
        self.meta_mlp.load_checkpoint()




if __name__ == "__main__":
    '''Driver code'''

    node_name = ['head_col', 'head_row', 'head_direction', 'tail_col', 'tail_row', 'tail_direction', 'portal_row', 'portal_col']

    steps = 4_800_000
    K_steps = 3
    L_steps = 5
    rollout_steps = 16
    random_seed = 5
    env = make_vec_env("GoogleSnake-v1", n_envs=1, env_kwargs={'config':config})
    n_actions = 3
    input_channel = env.observation_space['grid'].shape[0]
    height, width = env.observation_space['grid'].shape[1:]
    input_node = 0
    for i in range(len(node_name)):
        input_node += env.observation_space[node_name[i]].shape[0]
    gamma = 0.99
    alpha = 3e-4
    m_alpha = 1e-4
    betas = (0.9, 0.999)
    eps = 1e-4
    name = 'meta_agent_bmg'

    # set seed
    T.cuda.manual_seed(random_seed)
    T.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    agent = Agent(input_channel, height, width, input_node, n_actions, gamma, alpha, m_alpha, betas, eps, name, env,
                  steps, K_steps, L_steps, rollout_steps, random_seed)
    agent.run()
    print("done")
    agent.plot_results()



