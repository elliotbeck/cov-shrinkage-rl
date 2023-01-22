# idea taken from https://github.com/zzzxxxttt/pytorch_simple_RL/blob/master/a2c_mtcar.py

# import libraries
import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.optim as optim

import sklearn
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler


import matplotlib.pyplot as plt

from src.actor import Actor
from src.critic import Critic
from src.util import process_state, get_action, update_actor, update_critic, get_state_value

# chose device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using: ", device)

# initialize lists
last_score_plot = [0.]
avg_score_plot = [0.]

# define arguments
parser = argparse.ArgumentParser(description='PyTorch A2C solution of MountainCarContinuous-V0')
parser.add_argument('--gamma', type=float, default=0.986)
parser.add_argument('--actor_lr', type=float, default=1e-5)
parser.add_argument('--critic_lr', type=float, default=5e-4)  
parser.add_argument('--hidden_dim', type=int, default=200)
parser.add_argument('--max_episode', type=int, default=1000)
parser.add_argument('--seed', type=int, default=1)

# parse arguments
cfg = parser.parse_args()

# set up game environment
env = gym.make('MountainCarContinuous-v0')

# # set the seed
# torch.manual_seed(cfg.seed)

# generate some data to fit scaler and featurizer
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# feature engineering
featurizer = sklearn.pipeline.FeatureUnion([
    ("rbf1", RBFSampler(gamma=5, n_components=100)),
    ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
    ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
    ("rbf4", RBFSampler(gamma=0.5, n_components=100))
])
featurizer.fit(scaler.transform(observation_examples))

# set up actor and critic
actor = Actor(input_dim = featurizer.transform(observation_examples).shape[1],  
              hidden_dim = cfg.hidden_dim, 
              output_dim = 1).to(device)
critic = Critic(input_dim = featurizer.transform(observation_examples).shape[1], 
                hidden_dim = cfg.hidden_dim,
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
            
            # weird behaviour of the environment (first state a tuple, then a list)
            if isinstance(state, tuple):  
                state = state[0]
            else: 
                state = state
            
            # process the state to get the features    
            proc_state = process_state(state, scaler, featurizer)
            
            # get the action of the agent
            action = get_action(state, actor, env, device, proc_state)
            
            # get the next state, reward and status
            next_state, reward, done, _ = env.step([action])[0:4]
            
            # update episode score
            episode_score += reward # type: ignore  
                      
            # process the next state to get the features
            proc_state_next = process_state(next_state, scaler, featurizer) 

            # render the game
            # env.render()

            # calculate the target
            target = reward + cfg.gamma * get_state_value(proc_state_next, critic, device)
            
            # calculate the td error
            td_error = target - get_state_value(proc_state, critic, device)
            
            # only update the actor and critic every 10 steps to reduce variance
            # of gradient descent steps
            if t%10==0: 
                # update actor (gradient descent) 
                update_actor(state, 
                            actor, 
                            action, 
                            td_error, 
                            actor_optimizer, 
                            device, 
                            proc_state)
                
                # update critic (gradient descent) 
                update_critic(state, 
                            target, 
                            critic, 
                            critic_optimizer, 
                            device, 
                            proc_state)

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
                plt.hlines(90, 0, len(last_score_plot)-1, colors='r', linestyles='dashed') # type: ignore      
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
    plt.savefig('plots/reward_a2c_mtcar_cont.png')