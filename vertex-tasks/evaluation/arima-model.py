import argparse
from statistics import mean, stdev

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

import warnings
warnings.filterwarnings("ignore")

from graphesn.dataset import chickenpox_dataset, twitter_tennis_dataset, pedalme_dataset, wiki_maths_dataset
from stocks_loader import *

dataset_dict = {"chickenpox": chickenpox_dataset(target_lags=1),
                "tennis": twitter_tennis_dataset(feature_mode='encoded', target_offset=1),
                "pedalme": pedalme_dataset(target_lags=1),
                "wikimath": wiki_maths_dataset(target_lags=1),
                "stocks": stocks_dataset(target_offset=1)       
                }

def fn_arima(data, n, step_size):
    # Perform the rolling forecast
    data_forecast = []

    for i in range(n, len(data), step_size):
        
        # update input data
        input_data = data[max(0,i-n):i]
        
        # learn
        model = ARIMA(input_data, order=(1,0,0))
        model_fit = model.fit()
        
        # forecast
        f = model_fit.forecast(steps=step_size)
        data_forecast.append(f)
        
        # print(data_forecast)
        
    return np.array(data_forecast).flatten()

def arima_on_stocks():
    file_path = '/home/chri6578/Documents/GG_SPP/dataset/stocks_pd.pickle'

    # Loading dataset from pickle
    with open(file_path, 'rb') as f:
        df = pickle.load(f)

    # Sampling
    sample_interval = 14
    m = 100
    d_sample = df[0:-1:sample_interval]
    df_ = pd.DataFrame(d_sample)
    X = np.array(df_)

    mse_list = []
    
    n_back = m
    n_ahead = 1
    for v in range(X.shape[1]):
        y = X.T[v]
        try:
            y_pred = fn_arima(y, n_back, n_ahead)
            mse = np.mean( np.square( y[n_back: n_back+ len(y_pred) ] - y_pred ) )
            mse_list.append(mse)
            
        except:
            pass
        
    return mse_list
            

mse_list= arima_on_stocks()
# print(len(mse_list))

print(f'arima:stocks',
    f'{mean(mse_list):.3f} ± {stdev(mse_list):.3f}',
    sep='\t')