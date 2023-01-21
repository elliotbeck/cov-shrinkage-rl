# taken from 
# https://github.com/BY571/Deep-Reinforcement-Learning-Algorithm-Collection/blob/master/ContinousControl/A2C_conti_seperate_networks.ipynb

# import packages
import numpy as np
import pandas as pd
import gym
import math
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from env.portfolio_shrink_env import portfolio_shrink_env
from src.actor import Actor
from src.critic import Critic
from src.util import sample, test_net, run_optimization

# # get dataset
# data_daily = pd.read_csv('data/CRSPdaily.csv')

# # convert date to datetime
# data_daily['date'] = pd.to_datetime(data_daily['date'], format='%Y%m%d')
# data_daily = data_daily.set_index(['date'])

# # drop missing values and convert returns to float
# df = data_daily[(data_daily.RET != "C") & (data_daily.RET != "")]
# df['RET'] = df['RET'].astype(float)

# # take only data after 1980
# df = df.loc["1980-01-01":]

# set environment TODO: mamke custom env for our problem
env_name = "MountainCarContinuous-v0" #"MountainCarContinuous-v0" 
env = gym.make(env_name)

# # alternative env
# env = portfolio_shrink_env(df)

# let's see what we have in the env
print("action space: ", env.action_space.shape[0])
print("observation space: ", env.observation_space.shape[0])
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using: ", device)

# set hyperparameters
GAMMA = 0.9
ENTROPY_BETA = 0.01  
CLIP_GRAD = .3
LR_c = 0.01
LR_a = 0.01
HIDDEN_SIZE = 64


# define input shape and output shape
input_shape  = env.observation_space.shape[0]
output_shape = env.action_space.shape[0]

critic = Critic(input_shape, hidden_size=HIDDEN_SIZE).to(device)
actor = Actor(input_shape, output_shape, hidden_size=HIDDEN_SIZE).to(device)

# set the optimizers and hyperparameters
c_optimizer = optim.RMSprop(params = critic.parameters(),lr = LR_c)
a_optimizer = optim.RMSprop(params = actor.parameters(),lr = LR_a)

max_episodes = 500

actor_loss_list = []
critic_loss_list = []
entropy_list = []


average_100 = []
plot_rewards = []
steps = 0
max_steps = 1000

for ep in range(max_episodes):
    state = env.reset()
    done = False

    logprob_batch = []
    entropy_batch = []
    values_batch = []
    rewards_batch = []
    masks = []
    for step in range(max_steps):
        if isinstance(state, tuple):
            state = torch.from_numpy(state[0]).float()
        else:
            state = torch.from_numpy(state).float()
        mean, variance = actor(state.unsqueeze(0).to(device))   
        action, logprob, entropy = sample(mean.cpu(), variance.cpu())
        value = critic(state.unsqueeze(0).to(device))
        next_state, reward, done, _ = env.step(action[0].numpy())[0:4]

        logprob_batch.append(logprob)
        entropy_batch.append(entropy)
        values_batch.append(value)
        rewards_batch.append(reward)  
        masks.append(1 - done)

        state = next_state

        if done:
          break

    actor_loss, critic_loss = run_optimization(entropy,
                                            logprob_batch, 
                                            entropy_batch, 
                                            values_batch, 
                                            rewards_batch, 
                                            masks, 
                                            actor_optim=a_optimizer,
                                            critic_optim=c_optimizer,
                                            actor=actor,
                                            critic=critic,
                                            clip_grad=CLIP_GRAD,
                                            gamma=GAMMA,
                                            entropy_beta=ENTROPY_BETA)
        
    
    actor_loss_list.append(actor_loss)
    critic_loss_list.append(critic_loss)
    
    if ep != 0 and ep % 10 == 0:
        test_rewards, test_entropy, test_steps = test_net(actor=actor, env=env)
        entropy_list.append(test_entropy)
        plot_rewards.append(test_rewards)

        average_100.append(np.mean(plot_rewards[-100:]))
        print("\rEpisode: {} | Ep_Reward: {:.2f} | Average_100: {:.2f}".format(ep, test_rewards, np.mean(plot_rewards[-100:])), end = "", flush = True)
        
        
# PLOTTING RESULTS
plt.figure(figsize = (20,7))
plt.subplot(1,5,1)
plt.title("actor_loss")
plt.plot(torch.stack(actor_loss_list).detach().numpy())
plt.subplot(1,5,2)
plt.title("critic_loss")
plt.plot(torch.stack(critic_loss_list).detach().numpy())
plt.subplot(1,5,3)
plt.title("entropy")
plt.plot(entropy_list)
plt.subplot(1,5,4)
plt.title("rewards")
plt.plot(plot_rewards)
plt.subplot(1,5,5)
plt.title("Average100")
plt.plot(average_100)
plt.savefig("plots/results.png")
# plt.show()