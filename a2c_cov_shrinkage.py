#!/usr/bin/env python
# -*- coding: utf-8 -*-
# idea taken from https://github.com/zzzxxxttt/pytorch_simple_RL/blob/master/a2c_mtcar.py

# import libraries
import pandas as pd
import numpy as np    
import hdf5storage
import argparse
from itertools import count
from sklearn.covariance import LedoitWolf

import torch
import torch.optim as optim

import matplotlib.pyplot as plt

from src.actor import Actor
from src.critic import Critic
from env.portfolio_shrink_env import portfolio_shrink_env
from src.util import get_action, update_actor, update_critic, get_state_value

# import matplotlib
# matplotlib.use('TkAgg')

# load the data
filepath = 'data/Data_p100_n1260.mat'
data = hdf5storage.loadmat(filepath)
data = pd.DataFrame(data['Data_p100_n1260'], 
                    columns=['vintage_train', 
                             'vintage_test', 
                             'date', 
                             'permno', 
                             'stock_return'])

# first row has only nans
data = data.iloc[1:, :]


# convert date to datetime
data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
data['vintage_train'] = pd.to_datetime(data['vintage_train'], format='%Y%m%d')
data['vintage_test'] = pd.to_datetime(data['vintage_test'], format='%Y%m%d')

# get vintages
vintages = data.query('vintage_train != "NaT"')['vintage_train'].unique() # first one is nan

# chose device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using: ", device)

# initialize lists
last_score_plot = [0.]
avg_score_plot = [0.]
last_score_benchmark_plot = [0.]
avg_score_benchmark_plot = [0.]

# define arguments
parser = argparse.ArgumentParser(description='PyTorch A2C for Covariance Shrinkage')
parser.add_argument('--gamma', type=float, default=0.98)
parser.add_argument('--actor_lr', type=float, default=1e-3)
parser.add_argument('--critic_lr', type=float, default=5e-2)  
parser.add_argument('--hidden_dim', type=int, default=100)
parser.add_argument('--max_episode', type=int, default=1000)
parser.add_argument('--seed', type=int, default=1)

# parse arguments
cfg = parser.parse_args()

# set up game environment
env = portfolio_shrink_env(data, vintages)

# set up actor and critic
actor = Actor(hidden_dim = cfg.hidden_dim, 
              input_dim = 5, # TODO: Make this dynamic
              output_dim = 1).to(device)
critic = Critic(hidden_dim = cfg.hidden_dim,
                input_dim = 5, # TODO: Make this dynamic
                output_dim = 1).to(device)

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
                
            # make artificial permnos TODO: replace with real permnos
            permnos = np.arange(0, 100)
            permnos_vector = np.repeat(permnos, 1260)
            
            # replace permnos
            state = state.assign(permno=permnos_vector)

            # transform to wide format
            state_wide = pd.pivot(state,
                                  columns='permno',
                                  values='stock_return',
                                  index='date')
            
            # get shrinkage intensities
            state_shrinkage = [LedoitWolf().fit(state_wide.iloc[-252*i:, :]).shrinkage_ for i in range(1, 6)]
            state_shrinkage = np.array(state_shrinkage)
                        
            # get the action of the agent
            action = get_action(state_shrinkage, actor, env, device)
            actions.append(action)
            actions_benchmark.append(state_shrinkage[0])
            
            # # print action
            # print(str(t) + " " + str(action-state_shrinkage[0]))
            # print(str(t) + " " + str(action*10) + " " + str(state_shrinkage[0]*10))
            
            # get the next state, reward and status
            next_state, reward, done, _ = env.step(action) # type: ignore    
            
            # get the next state, reward and status for benchmark
            reward_benchmark = -1*env.get_portfolio_volatility(state_shrinkage[0]) # type: ignore    
                    
            # update episode score
            episode_score += reward # type: ignore  
            episode_score_benchmark += reward_benchmark # type: ignore
                      
            # replace permnos
            next_state = next_state.assign(permno=permnos_vector)

            # transform to wide format
            next_state_wide = pd.pivot(next_state,
                                       columns='permno',
                                       values='stock_return',
                                       index='date')
            
            # get the covariance matrix
            # get shrinkage intensities next state
            next_state_shrinkage = [LedoitWolf().fit(next_state_wide.iloc[-252*i:, :]).shrinkage_ for i in range(1, 6)]
            next_state_shrinkage = np.array(state_shrinkage)

            # render the game
            # env.render()

            # calculate the target
            target = reward + cfg.gamma * get_state_value(next_state_shrinkage, critic, device)
            
            # calculate the td error
            td_error = target - get_state_value(state_shrinkage, critic, device)
            
            # only update the actor and critic every 10 steps to reduce variance
            # of gradient descent steps. Furthermore, only update the actor for the 
            # first 200 episodes
            # if t%2==0 and i_episode < 200: 
            # update actor (gradient descent) 
            update_actor(state_shrinkage,
                         actor, 
                         action, 
                         td_error,
                         actor_optimizer, 
                         device)
            
            # update critic (gradient descent) 
            update_critic(state_shrinkage, 
                          target, 
                          critic, 
                          critic_optimizer, 
                          device)

            if done:
                
                # calculate average score and append to list
                avg_score_plot.append(
                    (avg_score_plot[-1] * len(avg_score_plot) + episode_score)/(len(avg_score_plot)+1)) # type: ignore                
                # append last score to list
                last_score_plot.append(float(episode_score))
                avg_score_benchmark_plot.append(
                    (avg_score_benchmark_plot[-1] * len(avg_score_benchmark_plot) + episode_score_benchmark)/(len(avg_score_benchmark_plot)+1)) # type: ignore                
                # append last score to list
                last_score_benchmark_plot.append(float(episode_score_benchmark))
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
                plt.savefig('plots/actions_a2c_cov_shrink_episode_' + str(i_episode) + '.png')
                
                # go to next episode
                break

            state = next_state

        stats.append(episode_score)
            
        # print episode results
        print("Episode: {}, reward: {}, steps: {}.".format(i_episode, episode_score, t)) # type: ignore
        
        # check if game is solved successfully
        if np.mean(stats[-100:]) > 90 and len(stats) >= 100:
            print(np.mean(stats[-100:]))
            print("Solved successfully!")

        
    # return the average score of the last 100 episodes
    return np.mean(stats[-100:])


if __name__ == '__main__':
    main()
    plt.savefig('plots/reward_a2c_cov_shrink.png')