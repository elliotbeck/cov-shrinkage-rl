# libraries
import torch
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import numpy as np

# set up device
device = "cuda" if torch.cuda.is_available() else "cpu"

# define test function
def test_net(actor, env, count = 10):
    rewards = 0.0
    steps = 0
    entropys = 0.0
    for _ in range(count):
        obs = env.reset()
        i = 0
        while True:
            i += 1
            if isinstance(obs, tuple):  
                obs_v = torch.from_numpy(obs[0]).unsqueeze(0).float()
            else: 
                obs_v = torch.from_numpy(obs).unsqueeze(0).float()
            mean_v, var_v = actor(obs_v.to(device))
            action, _, entropy = sample(mean_v[0].cpu(), var_v[0].cpu()) #[0]
            obs, reward, done, info = env.step(action.numpy())[0:4]
            if reward>0:
                print("WINNNNNN: " + str(reward))
            rewards += reward
            entropys += np.mean(entropy.detach().numpy())
            steps += 1
            if done:
                break  
            elif i > 1000:
                break

    return rewards/count, entropys/count, steps/count

# define the return function
def compute_returns(rewards,masks, gamma):
    R = 0 #pred.detach()
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return torch.FloatTensor(returns).reshape(-1).unsqueeze(1)

# function to sample from normal distribution
def sample(mean, variance):
  """
  Calculates the actions, log probs and entropy based on a normal distribution by a given mean and variance.
  
  ====================================================
  
  calculate log prob:
  log_prob = -((action - mean) ** 2) / (2 * var) - log(sigma) - log(sqrt(2 *pi))
  
  calculate entropy:
  entropy =  0.5 + 0.5 * log(2 *pi * sigma) 
  entropy here relates directly to the unpredictability of the actions which an agent takes in a given policy.
  The greater the entropy, the more random the actions an agent takes.
  
  """
  sigma = torch.sqrt(variance)
  m = Normal(mean, sigma)
  actions = m.sample()
  actions = torch.clamp(actions, -1, 1) # TODO: Will be torch.clamp(actions, 0, 1) for us
  logprobs = m.log_prob(actions)
  entropy = m.entropy()  # Equation: 0.5 + 0.5 * log(2 *pi * sigma)
    
  return actions, logprobs, entropy


# define the loss function
def run_optimization(entropy,
                     logprob_batch, 
                     entropy_batch, 
                     values_batch, 
                     rewards_batch, 
                     masks, 
                     actor,
                     critic, 
                     actor_optim,
                     critic_optim, 
                     gamma,
                     clip_grad,
                     entropy_beta):
    """
    Calculates the actor loss and the critic loss and backpropagates it through the Network
    
    ============================================
    Critic loss:
    c_loss = -logprob * advantage
    
    a_loss = 
    
    """
    
    log_prob_v = torch.cat(logprob_batch).to(device)
    entropy_v = torch.cat(entropy_batch).to(device)
    value_v = torch.cat(values_batch).to(device)
    
    

    rewards_batch = torch.FloatTensor(rewards_batch)
    masks = torch.FloatTensor(masks)
    discounted_rewards = compute_returns(rewards_batch, masks, gamma=gamma).to(device)
    
    # critic_loss
    critic_optim.zero_grad()
    critic_loss = 0.5 * F.mse_loss(value_v, discounted_rewards) #+ ENTROPY_BETA * entropy.detach().mean()
    critic_loss.backward()
    clip_grad_norm_(critic.parameters(),clip_grad)
    critic_optim.step()
    
    # A(s,a) = Q(s,a) - V(s)
    advantage = discounted_rewards - value_v.detach() 

    #actor_loss
    actor_optim.zero_grad()
    actor_loss = (-log_prob_v * advantage).mean() + entropy_beta * entropy.detach().mean()
    actor_loss.backward()
    clip_grad_norm_(actor.parameters(),clip_grad)
    actor_optim.step()
    
    return actor_loss, critic_loss