# libraries
import pandas as pd
import scipy.io as sio


import h5py
import numpy as np
filepath = 'data/Data_p100_n1260.mat'    
import hdf5storage
mat = hdf5storage.loadmat(filepath)
mat = pd.DataFrame(mat)
mat.iloc[(1260):(1260+10), 0:5]


# load data
data_daily = pd.read_csv('data/Data_p100_n1260.csv')
data_daily.shape
data_daily.iloc[0:500, 0:5]

# convert date to datetime
data_daily['date'] = pd.to_datetime(data_daily['date'], format='%Y%m%d')
data_daily = data_daily.set_index(['date'])

# drop missing values and convert returns to float
data_daily = data_daily[(data_daily.RET != "C") & (data_daily.RET != "")]
data_daily['RET'] = data_daily['RET'].astype(float)

# get 5 years of data
min_date = "1980-01-01"
max_date = "1985-01-01"
data_subset = data_daily.loc[min_date:max_date] # type: ignore

# long to wide
data_subset_wide = data_subset.pivot(columns='PERMNO', values='RET')
data_subset_wide.dropna(axis=1, inplace=True)

# artificially use only 100 stocks
stocks = data_subset_wide.iloc[:, 0:100]
stocks_cov = stocks.cov()
