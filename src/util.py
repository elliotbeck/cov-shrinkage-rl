# libraries
import torch
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import numpy as np

def process_state(state, scaler, featurizer):
  scaled = scaler.transform([state])
  featurized = featurizer.transform(scaled)
  return featurized[0]


def get_action(state, actor, env, device, process_state):
    state = torch.from_numpy(process_state).float().to(device)
    action_mu, action_sigma = actor(state)
    action_dist = torch.distributions.normal.Normal(action_mu, action_sigma)
    action = action_dist.sample()
    action = torch.clamp(action, float(env.action_space.low[0]), float(env.action_space.high[0]))
    return action.item()


def get_state_value(state, critic, process_state, device):
    state = torch.from_numpy(process_state).float().to(device)
    state_value = critic(state)
    return state_value.item()


def update_actor(state, actor, action, advantage, actor_optimizer, device, process_state):
    state = torch.from_numpy(process_state).float().to(device)
    action_mu, action_sigma = actor(state)
    action_dist = torch.distributions.normal.Normal(action_mu, action_sigma)
    act_loss = -action_dist.log_prob(torch.tensor(action).to(device)) * advantage
    entropy = action_dist.entropy()
    loss = act_loss - 1e-4 * entropy
    actor_optimizer.zero_grad()
    loss.backward()
    actor_optimizer.step()
    return


def update_critic(state, target, critic, critic_optimizer, device, process_state):
    state = torch.from_numpy(process_state).float().to(device)
    state_value = critic(state)
    loss = F.mse_loss(state_value, torch.tensor([target]).to(device))
    critic_optimizer.zero_grad()
    loss.backward()
    critic_optimizer.step()
    return