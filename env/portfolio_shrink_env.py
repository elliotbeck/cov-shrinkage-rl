import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from itertools import chain

class portfolio_shrink_env(gym.Env):
    def __init__(self, df, vintages):
        super().__init__()
        self.stock_returns = df
        self.action_space = gym.spaces.Box(low = 0.0, 
                                           high = 1.0, 
                                           shape=(1, ), 
                                           dtype=np.float32)
        self.done = False
        self.state = None
        self.reward = None
        self.info = None
        self.vintages = vintages
        self.portfolio = []

    def reset(self):
        '''
        Reset environment status to initial point.
        Return the initial state.

        Returns:
            state: np.array
        '''
        self.date = self.vintages[0]
        self.state = self.stock_returns.loc[self.stock_returns['vintage_train'] == self.vintages[0], :]
        self.done = False
        self.reward = None
        action = self.action_space.sample()
        
        state, _, _, _ = self.step(action)

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
        self.date = self.vintages[np.where(self.date==self.vintages)[0]+1]
        self.state = self.get_state()

        return self.state, self.reward, self.done, {}
    
    def render(self):
        s = "position: {}  reward: {}  info: {}"
        print(s.format(self.state, self.reward, self.info))

    def get_state(self):
        '''
        Take agent's action and get back env's next state
        Return:
            next state
        '''
        data = self.stock_returns.loc[self.stock_returns['vintage_train'].values == self.date, :]
        return data


    def get_portfolio(self, action):
        
        # make artificial permnos TODO: replace with real permnos
        permnos = np.arange(0, 100)
        permnos_vector = np.repeat(permnos, 1260)
        
        # replace permnos
        self.state.loc[:, 'permno'] = permnos_vector

        # transform to wide format
        self.state_wide = pd.pivot(self.state,
                             columns='permno',
                             values='stock_return',
                             index='date')

        # get the covariance matrix
        covariance_matrix = self.state_wide.cov()
        
        # shrink the covariance matrix
        covariance_shrunk = self.get_shrank_cov(
            covariance_matrix = covariance_matrix,
            a=action)
        
        # get the portfolio weights
        portfolio = self.get_GMVP(covariance_matrix = covariance_shrunk)
        return portfolio
    
    def is_done(self):
        '''
        Check whether agent arrive to the endpoint of the epoc
        '''
        if self.date != self.vintages[-2]:
            self.done = False
        else:
            self.done = True
            
        return self.done
    
    def get_portfolio_returns(self, action):
        '''
        Get portfolio daily returns time series for one period
        '''
        portfolio = self.get_portfolio(action)
        self.portfolio.append(portfolio) # record
        portfolio_returns = self.state_wide @ portfolio
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
        volatility = portfolio_returns.std() * np.sqrt(252) # annualized        
        return volatility
        
    def get_shrank_cov(
                    self, 
                    a: int,
                    covariance_matrix):
        '''
        Calculate shrank covariance matrix given shrink target and 
        shrink intensity.
        '''
        # get the shrink target
        variances = np.diag(covariance_matrix) 
        shrink_target = np.eye(covariance_matrix.shape[0]) * np.mean(variances)
        
        # initialization
        R_1 = covariance_matrix
        R_2 = shrink_target

        # cov calculation
        H = (1 - a) * R_1.values + a * R_2 # new shraxnk covariance matrix

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
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]