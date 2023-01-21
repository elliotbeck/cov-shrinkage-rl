import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
import matplotlib
matplotlib.use('Agg')
from typing import Union
from itertools import chain

class portfolio_shrink_env(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.stock_returns = df
        stock_num = self.stock_returns.shape[1]
        self.action_space = gym.spaces.Box(low = 0.0, 
                                           high = 1.0, 
                                           shape=(1, ), 
                                           dtype=np.float32)
        self.observation_space = gym.spaces.Box(low = -100, 
                                                high = 100, 
                                                shape=(1, stock_num + 1), 
                                                dtype=np.float32)
        self.done = False
        self.state = None
        self.reward = None
        self.info = None
        self.reward_type = 'volatility'
        self.date = df.index[0]
        self.date_range = df.index
        self.portfolio = []
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        '''
        Reset environment status to initial point.
        Return the initial state.

        Returns:
            state: np.array
        '''
        self.state = self.stock_returns.iloc[0]
        self.done = False
        self.reward = None
        action = self.action_space.sample()
        
        state, reward, done, info = self.step(action)

        return state


    def step (self, action):
        '''
        Core function in environment. Take action as input, and 
        respond to agent.
        Args:
            action: np.array
                    shrinkage intensity
        Returns:
            state: np.array
            reward: float
            done: Bool
        '''        
        
        # get the reward (negative volatility)
        self.reward = -self.get_portfolio_volatility(action)
        
        # check if we are done
        self.done = self.is_done()
        
        # get the next state
        self.date += 1 # TODO: change this to next date
        self.state = self.get_state()

        return self.state, self.reward, self.done, {}

    def get_state(self):
        '''
        Take agent's action and get back env's next state
        Args:
            action: a number (shrinkage intensity)
        Return:
            None
        '''
        if not self.done:
            return self.stock_returns[self.date]
        else:
            print('The end of period\n')
            # exit()

    def get_portfolio(self, action):
        # shrink the covariance matrix
        covariance_matrix = self.state.cov()
        covariance_shrunk = self.get_shrank_cov(
            covariance_matrix = covariance_matrix,
            shrink_target=np.identity(covariance_matrix.shape[1]),
            a=action)
        # get the portfolio weights
        portfolio = self.get_GMVP(covariance_matrix = covariance_shrunk)
        return portfolio
    
    def is_done(self):
        '''
        Check whether agent arrive to the endpoint of the epoc
        '''
        if self.date != self.date[-1]:
            self.done = False
        else:
            self.done = True
            
        return self.done
    
    def get_portfolio_returns(self, action):
        '''
        Get portfolio daily returns time series for one period
        '''
        stocks_returns = self.state
        portfolio = self.get_portfolio(action)
        self.portfolio.append(portfolio) # record
        portfolio_returns = stocks_returns @ portfolio
        portfolio_returns = portfolio_returns.values.tolist()
        portfolio_returns = list(chain.from_iterable(portfolio_returns))

        return portfolio_returns
    
    def get_portfolio_volatility(self, action) -> np.float_:
        '''Given portfolio daily returns to calculate Volatility

        Return:
            Volatility of portfolio
        '''
        # initialization
        portfolio_returns = self.get_portfolio_returns(action)
        portfolio_returns = np.array(portfolio_returns)

        # calculate volatility
        volatility = portfolio_returns.std()* np.sqrt(252) # annualized        
        return volatility
        
    def render (self):
        s = "position: {}  reward: {}  info: {}"
        print(s.format(self.state, self.reward, self.info))
        
    def get_shrank_cov(
                    self, 
                    shrink_target: Union[pd.DataFrame, np.array],
                    a: int,
                    covariance_matrix: Union[pd.DataFrame, np.array] = None
                    ) -> Union[pd.DataFrame, np.array]:
        '''
        Calculate shrank covariance matrix given shrink target and 
        shrink intensity.
        '''
        # initialization
        R_1 = covariance_matrix
        R_2 = shrink_target

        # cov calculation
        H = (1 - a) * R_1 + a * R_2 # new shrank covariance matrix

        return H

    def get_GMVP(self, 
                 covariance_matrix):
        '''Get Global Minimum Variance Portfolio
        Args:
            covariance_matrix: covariance_matrixused to build GMVP
        Returns:
            GMVP(column vector)
        '''
        # initialization
        H = covariance_matrix

        # pre calculation
        one = np.ones(H.shape[0]) # vector of 1s
        H_inv = np.linalg.inv(H)
        numerator = H_inv @ one
        denominator = one.T @ H_inv @ one

        # GMV porfolio
        x = numerator / denominator

        # reshape to column vector
        return x.reshape((len(x), 1))