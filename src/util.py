# libraries
import torch
from torch.distributions import Normal
import torch.nn.functional as F
import pandas as pd
import numpy as np
import hdf5storage


def get_action(factor_returns, actor, device):
    factor_returns = np.expand_dims(factor_returns, axis=0)
    factor_returns = torch.from_numpy(factor_returns).float().to(device)
    action_mu, action_sigma = actor(factor_returns)
    print(action_mu, action_sigma)
    action_dist = torch.distributions.beta.Beta(action_mu, action_sigma)
    action = action_dist.sample()
    return action.item()


def get_state_value(factor_returns, critic, device):
    factor_returns = np.expand_dims(factor_returns, axis=0)
    factor_returns = torch.from_numpy(factor_returns).float().to(device)
    state_value = critic(factor_returns)
    return state_value.item()


def update_actor(factor_returns, actor, action, advantage, actor_optimizer, device):
    factor_returns = np.expand_dims(factor_returns, axis=0)
    factor_returns = torch.from_numpy(factor_returns).float().to(device)
    action_mu, action_sigma = actor(factor_returns)
    action_dist = torch.distributions.beta.Beta(action_mu, action_sigma)
    act_loss = - \
        action_dist.log_prob(torch.tensor(action).to(device)) * advantage
    entropy = action_dist.entropy()
    loss = act_loss - 1e-4 * entropy
    actor_optimizer.zero_grad()
    loss.backward()
    actor_optimizer.step()
    return


def update_critic(factor_returns, target, critic, critic_optimizer, device):
    factor_returns = np.expand_dims(factor_returns, axis=0)
    factor_returns = torch.from_numpy(factor_returns).float().to(device)
    state_value = critic(factor_returns)
    loss = F.mse_loss(state_value.float(),
                      torch.tensor([[target]]).float().to(device))
    critic_optimizer.zero_grad()
    loss.backward()
    critic_optimizer.step()
    return


def import_factor_data(file_path='data/global_factors_kelly.csv'):
    factor_data = pd.read_csv(file_path)
    factor_data['date'] = pd.to_datetime(factor_data['date'])
    factor_data_wide = factor_data.pivot(
        index='date', columns='name', values='ret')
    return factor_data_wide


def import_stock_data(file_path='data/Data_p100_n1260.mat'):
    stock_data = hdf5storage.loadmat(file_path)
    stock_data = pd.DataFrame(stock_data['Data_p100_n1260'],
                              columns=['vintage_train',
                                       'vintage_test',
                                       'date',
                                       'permno',
                                       'stock_return'])

    # first row has only nans
    stock_data = stock_data.iloc[1:, :]

    # convert date to datetime
    stock_data['date'] = pd.to_datetime(stock_data['date'], format='%Y%m%d')
    stock_data['vintage_train'] = pd.to_datetime(
        stock_data['vintage_train'], format='%Y%m%d')
    stock_data['vintage_test'] = pd.to_datetime(
        stock_data['vintage_test'], format='%Y%m%d')
    return stock_data
