#!/usr/bin/env python
# -*- coding: utf-8 -*-
# idea taken from https://github.com/zzzxxxttt/pytorch_simple_RL/blob/master/a2c_mtcar.py

# import libraries
import pandas as pd
import numpy as np
import argparse
from itertools import count
from sklearn.covariance import LedoitWolf

import torch
import torch.optim as optim

import matplotlib.pyplot as plt

from src.actor import Actor
from src.critic import Critic
from env.portfolio_shrink_env import portfolio_shrink_env
from src.util import get_action, update_actor, update_critic, get_state_value, import_factor_data, import_stock_data

# import matplotlib
# matplotlib.use('TkAgg')

# load the data
stock_returns = import_stock_data()
factor_returns = import_factor_data()

# get vintages
vintages = stock_returns.query('vintage_train != "NaT"')[
    'vintage_train'].unique()  # first one is nan

# chose device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using: ", device)

# initialize lists for plotting
last_score_plot = [0.]
avg_score_plot = [0.]
last_score_benchmark_plot = [0.]
avg_score_benchmark_plot = [0.]

# define arguments
parser = argparse.ArgumentParser(
    description='PyTorch A2C for Covariance Shrinkage')
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--actor_lr', type=float, default=8e-3)
parser.add_argument('--critic_lr', type=float, default=8e-3)
parser.add_argument('--hidden_dim', type=int, default=100)
parser.add_argument('--max_episode', type=int, default=1000)
parser.add_argument('--seed', type=int, default=1)

# parse arguments
cfg = parser.parse_args()

# set up game environment
env = portfolio_shrink_env(return_data=stock_returns,
                           factor_data=factor_returns,
                           vintages=vintages)

# set up actor and critic
actor = Actor(hidden_dim=cfg.hidden_dim,
              input_dim=factor_returns.shape[1],
              output_dim=100).to(device)
critic = Critic(hidden_dim=cfg.hidden_dim,
                input_dim=factor_returns.shape[1],
                output_dim=1).to(device)

# set up optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=cfg.actor_lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=cfg.critic_lr)

# define the main function to run the game iterartively


def main():
    stats = []
    # plt.ion()
    # plt.grid()
    # plt.show()
    for i_episode in range(cfg.max_episode):

        # set up a new game
        state = env.reset()

        # set up the score
        episode_score = 0
        episode_score_benchmark = 0

        # save actions for plots
        actions = []
        actions_benchmark = []

        # iterate through the steps of the game
        for t in count():

            # # make artificial permnos TODO: replace with real permnos
            # permnos = np.arange(0, 100)
            # permnos_vector = np.repeat(permnos, 1260)

            # # replace permnos
            # state_stock_returns = state[0].assign(permno=permnos_vector)

            #  transform to wide format
            state_stock_returns_wide = pd.pivot(state[0],
                                                columns='permno',
                                                values='stock_return',
                                                index='date')

            # get shrinkage intensities
            benchmark_shrinkage = [LedoitWolf().fit(
                state_stock_returns_wide.iloc[-252*i:, :]).shrinkage_ for i in range(1, 6)]
            benchmark_shrinkage = np.array(benchmark_shrinkage)

            # get the action of the agent
            action = get_action(
                factor_returns=state[1], actor=actor, device=device)
            actions.append(action/100)
            actions_benchmark.append(benchmark_shrinkage[0])

            # get the next state, reward and status
            next_state, reward, done, _ = env.step(action/100)  # type: ignore

            # get the next state, reward and status for benchmark
            reward_benchmark = -1 * \
                env.get_portfolio_volatility(
                    benchmark_shrinkage[0])  # type: ignore

            # print action
            print(str(t) + ", " +
                  str(action/100) + ", " +
                  str(benchmark_shrinkage[0]) + ", " +
                  str(reward_benchmark) + ", " + str(reward))

            # update episode score
            episode_score += reward  # type: ignore
            episode_score_benchmark += reward_benchmark  # type: ignore

            # render the game
            # env.render()

            #  calculate the target
            target = reward + cfg.gamma * \
                get_state_value(next_state[1], critic, device)

            # calculate the td error
            td_error = target - \
                get_state_value(next_state[1], critic, device)

            # only update the actor and critic every 10 steps to reduce variance
            #  of gradient descent steps. Furthermore, only update the actor for the
            # first 200 episodes
            # if t%2==0 and i_episode < 200:
            # update actor (gradient descent)
            update_actor(state[1],
                         actor,
                         action,
                         td_error,
                         actor_optimizer,
                         device)

            # update critic (gradient descent)
            update_critic(state[1],
                          target,
                          critic,
                          critic_optimizer,
                          device)

            if done:

                # calculate average score and append to list
                avg_score_plot.append(
                    (avg_score_plot[-1] * len(avg_score_plot) +
                     episode_score)/(len(avg_score_plot)+1))  # type: ignore
                # append last score to list
                last_score_plot.append(float(episode_score))
                avg_score_benchmark_plot.append(
                    (avg_score_benchmark_plot[-1] * len(avg_score_benchmark_plot) +
                     episode_score_benchmark)/(len(avg_score_benchmark_plot)+1))  # type: ignore
                # append last score to list
                last_score_benchmark_plot.append(
                    float(episode_score_benchmark))
                print(last_score_plot)
                print(last_score_benchmark_plot)
                # # plot intermediate results
                # plt.title('reward')
                # plt.xlabel('episode')
                # plt.plot(last_score_plot, 'c-', legend='agent')
                # plt.plot(avg_score_plot, 'g-', legend='agent average')
                # plt.plot(last_score_benchmark_plot, 'r-', legend='benchmark')
                # plt.plot(avg_score_benchmark_plot, 'y-', legend='benchmark average')
                # plt.draw()
                # plt.pause(0.001)
                plt.plot(actions, 'c-', label='agent')
                plt.plot(actions_benchmark, 'r-', label='benchmark')
                plt.title('actions over time')
                plt.xlabel('episode')
                plt.legend()
                plt.savefig('plots/actions_a2c_cov_shrink_episode_' +
                            str(i_episode) + '.png')
                plt.clf()

                # go to next episode
                break

            state = next_state

        stats.append(episode_score)

        # print episode results
        print("Episode: {}, reward: {}, steps: {}.".format(
            i_episode, episode_score, t))  # type: ignore

        # check if game is solved successfully
        if np.mean(stats[-100:]) > 90 and len(stats) >= 100:
            print(np.mean(stats[-100:]))
            print("Solved successfully!")

    # return the average score of the last 100 episodes
    return np.mean(stats[-100:])


if __name__ == '__main__':
    main()
    plt.savefig('plots/reward_a2c_cov_shrink.png')
