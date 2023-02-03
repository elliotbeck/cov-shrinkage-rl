# libraries
import torch
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np

def get_action(state, actor, env, device):
    state = np.expand_dims(np.array(state.values), axis=0)
    state = torch.from_numpy(state).float().to(device)
    action_mu, action_sigma = actor(state)
    action_dist = torch.distributions.normal.Normal(action_mu, action_sigma)
    action = action_dist.sample()
    action = torch.clamp(action, float(env.action_space.low[0]), float(env.action_space.high[0]))
    return action.item()


def get_state_value(state, critic, device):
    state = np.expand_dims(np.array(state.values), axis=0)
    state = torch.from_numpy(state).float().to(device)
    state_value = critic(state)
    return state_value.item()


def update_actor(state, actor, action, advantage, actor_optimizer, device):
    state = np.expand_dims(np.array(state.values), axis=0)
    state = torch.from_numpy(state).float().to(device)
    action_mu, action_sigma = actor(state)
    action_dist = torch.distributions.normal.Normal(action_mu, action_sigma)
    act_loss = -action_dist.log_prob(torch.tensor(action).to(device)) * advantage
    entropy = action_dist.entropy()
    loss = act_loss - 1e-4 * entropy
    actor_optimizer.zero_grad()
    loss.backward()
    actor_optimizer.step()
    return


def update_critic(state, target, critic, critic_optimizer, device):
    state = np.expand_dims(np.array(state.values), axis=0)
    state = torch.from_numpy(state).float().to(device)
    state_value = critic(state)
    loss = F.mse_loss(state_value.float(), torch.tensor([target]).float().to(device))
    critic_optimizer.zero_grad()
    loss.backward()
    critic_optimizer.step()
    return
