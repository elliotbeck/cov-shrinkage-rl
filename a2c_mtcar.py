# taken from https://github.com/zzzxxxttt/pytorch_simple_RL/blob/master/a2c_mtcar.py

# import libraries
import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F

import sklearn
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler


from drawnow import drawnow, figure
import matplotlib.pyplot as plt

from src.actor import Actor
from src.critic import Critic

# chose device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using: ", device)

last_score_plot = [0]
avg_score_plot = [0]

def draw_fig():
    plt.title('reward')
    plt.plot(last_score_plot, '-')
    plt.plot(avg_score_plot, 'r-')


parser = argparse.ArgumentParser(description='PyTorch A2C solution of MountainCarContinuous-V0')
parser.add_argument('--gamma', type=float, default=0.986)
parser.add_argument('--actor_lr', type=float, default=1e-5)
parser.add_argument('--critic_lr', type=float, default=5e-4)  
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--max_episode', type=int, default=100)
parser.add_argument('--seed', type=int, default=8)

cfg = parser.parse_args()

env = gym.make('MountainCarContinuous-v0')

torch.manual_seed(cfg.seed)

observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

featurizer = sklearn.pipeline.FeatureUnion([
    ("rbf1", RBFSampler(gamma=5, n_components=100)),
    ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
    ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
    ("rbf4", RBFSampler(gamma=0.5, n_components=100))
])
featurizer.fit(scaler.transform(observation_examples))

def process_state(state):
  scaled = scaler.transform([state])
  featurized = featurizer.transform(scaled)
  return featurized[0]


def get_action(state, actor):
    state = torch.from_numpy(process_state(state)).float().to(device)
    action_mu, action_sigma = actor(state)
    action_dist = torch.distributions.normal.Normal(action_mu, action_sigma)
    action = action_dist.sample()
    action = torch.clamp(action, float(env.action_space.low[0]), float(env.action_space.high[0]))
    return action.item()


def get_state_value(state, critic):
    state = torch.from_numpy(process_state(state)).float().to(device)
    state_value = critic(state)
    return state_value.item()


def update_actor(state, actor, action, advantage):
    state = torch.from_numpy(process_state(state)).float().to(device)
    action_mu, action_sigma = actor(state)
    action_dist = torch.distributions.normal.Normal(action_mu, action_sigma)
    act_loss = -action_dist.log_prob(torch.tensor(action).to(device)) * advantage
    entropy = action_dist.entropy()
    loss = act_loss - 1e-4 * entropy
    actor_optimizer.zero_grad()
    loss.backward()
    actor_optimizer.step()
    return


def update_critic(state, target, critic):
    state = torch.from_numpy(process_state(state)).float().to(device)
    state_value = critic(state)
    loss = F.mse_loss(state_value, torch.tensor([target]).to(device))
    critic_optimizer.zero_grad()
    loss.backward()
    critic_optimizer.step()
    return

actor = Actor(input_dim = featurizer.transform(observation_examples).shape[1],  
              hidden_dim = cfg.hidden_dim, 
              output_dim = 1).to(device)
critic = Critic(input_dim = featurizer.transform(observation_examples).shape[1], 
                hidden_dim = cfg.hidden_dim,
                output_dim = 1).to(device)
actor_optimizer = optim.Adam(actor.parameters(), lr=cfg.actor_lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=cfg.critic_lr)


def main():
    stats = []
    for i_episode in range(cfg.max_episode):
        state = env.reset()
        episode_score = 0
        for t in count():
            if isinstance(state, tuple):  
                state = state[0]
            else: 
                state = state
            action = get_action(state, actor)
            next_state, reward, done, _ = env.step([action])[0:4]
            episode_score += reward

            # env.render()

            target = reward + cfg.gamma * get_state_value(next_state, critic)
            td_error = target - get_state_value(state, critic)
            
            update_actor(state, actor, action, advantage=td_error)
            update_critic(state, target, critic)

            if done:
                avg_score_plot.append(
                    (avg_score_plot[-1] * len(avg_score_plot) + episode_score)/(len(avg_score_plot)+1))
                last_score_plot.append(episode_score)
                drawnow(draw_fig)
                break

            state = next_state

        stats.append(episode_score)
        if np.mean(stats[-100:]) > 90 and len(stats) >= 100:
            print(np.mean(stats[-50:]))
            print("Solved")
        print("Episode: {}, reward: {}.".format(i_episode, episode_score))
    return np.mean(stats[-50:])


if __name__ == '__main__':
    main()
    plt.pause(0)