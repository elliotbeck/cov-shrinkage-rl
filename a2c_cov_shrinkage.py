# idea taken from https://github.com/zzzxxxttt/pytorch_simple_RL/blob/master/a2c_mtcar.py

# import libraries
import pandas as pd
import numpy as np    
import hdf5storage

import argparse
from itertools import count

import torch
import torch.optim as optim

import matplotlib.pyplot as plt

from src.actor import Actor
from src.critic import Critic
from env.portfolio_shrink_env import portfolio_shrink_env
from src.util import get_action, update_actor, update_critic, get_state_value

import matplotlib
import tkinter
matplotlib.use('TkAgg')

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

# define arguments
parser = argparse.ArgumentParser(description='PyTorch A2C solution of MountainCarContinuous-V0')
parser.add_argument('--gamma', type=float, default=0.98)
parser.add_argument('--actor_lr', type=float, default=5e-4)
parser.add_argument('--critic_lr', type=float, default=5e-3)  
parser.add_argument('--hidden_dim', type=int, default=200)
parser.add_argument('--max_episode', type=int, default=1000)
parser.add_argument('--seed', type=int, default=1)

# parse arguments
cfg = parser.parse_args()

# set up game environment
env = portfolio_shrink_env(data, vintages)

# set up actor and critic
actor = Actor(hidden_dim = cfg.hidden_dim, 
              output_dim = 1).to(device)
critic = Critic(hidden_dim = cfg.hidden_dim,
                output_dim = 1).to(device)

# set up optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=cfg.actor_lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=cfg.critic_lr)

# define the main function to run the game iterartively
def main():
    stats = []
    plt.ion()
    plt.grid()
    plt.show()
    for i_episode in range(cfg.max_episode):
        
        # set up a new game
        state = env.reset()
        
        # set up the score
        episode_score = 0
        
        # iterate through the steps of the game
        for t in count():
                
            # make artificial permnos TODO: replace with real permnos
            permnos = np.arange(0, 100)
            permnos_vector = np.repeat(permnos, 1260)
            
            # replace permnos
            state.loc[:, 'permno'] = permnos_vector

            # transform to wide format
            state_wide = pd.pivot(state,
                                  columns='permno',
                                  values='stock_return',
                                  index='date')
            
            # get the covariance matrix
            state_covariance = state_wide.cov()
            
            # get the action of the agent
            action = get_action(state_covariance, actor, env, device)
            
            # print action
            print(str(t) + " " + str(action))
            
            # get the next state, reward and status
            next_state, reward, done, _ = env.step(action)
            
            # update episode score
            episode_score += reward # type: ignore  
                      
            # replace permnos
            next_state.loc[:, 'permno'] = permnos_vector

            # transform to wide format
            next_state_wide = pd.pivot(next_state,
                                       columns='permno',
                                       values='stock_return',
                                       index='date')
            
            # get the covariance matrix
            next_state_covariance = next_state_wide.cov()

            # render the game
            # env.render()

            # calculate the target
            target = reward + cfg.gamma * get_state_value(next_state_covariance, critic, device)
            
            # calculate the td error
            td_error = target - get_state_value(state_covariance, critic, device)
            
            # only update the actor and critic every 10 steps to reduce variance
            # of gradient descent steps. Furthermore, only update the actor for the 
            # first 200 episodes
            # if t%2==0 and i_episode < 200: 
            # update actor (gradient descent) 
            update_actor(state_covariance,
                            actor, 
                            action, 
                            td_error,
                            actor_optimizer, 
                            device)
            
            # update critic (gradient descent) 
            update_critic(state_covariance, 
                            target, 
                            critic, 
                            critic_optimizer, 
                            device)

            if done:
                
                # calculate average score and append to list
                avg_score_plot.append(
                    (avg_score_plot[-1] * len(avg_score_plot) + episode_score)/(len(avg_score_plot)+1))
                
                # append last score to list
                last_score_plot.append(episode_score)
                
                # plot intermediate results
                plt.title('reward')
                plt.xlabel('episode')
                plt.plot(last_score_plot, 'b-')
                plt.plot(avg_score_plot, 'g-')
                plt.draw()
                plt.pause(0.001)
                
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