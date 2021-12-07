import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.signal import lfilter
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
import itertools
import statsmodels.api as sm
import matplotlib
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import LinearRegression
import datetime
from ETL_and_converting_data import *


def create_linear_regression_model(switch=1, n_days=6):
    """
    6 days was selected based on bootstrapping data around a week
    from 1 to 14 days were tested and best fit is 6 days
    """
    ready_ts = Etl(switch).adjust_random_walks()
    x_list = []
    for i in range(2, n_days + 2):
        name = 'SMA_{}'.format(i)
        x_list.append(name)
        ready_ts[name] = ready_ts['Values'].shift(i)
    ready_ts = ready_ts.dropna()
    ready_ts.set_index('Index')
    X = ready_ts[x_list]
    model = LinearRegression()
    model.fit(X, ready_ts['Values'])
    r_sq = model.score(X, ready_ts['Values'])
    print('RSquared is equal to:{}'.format(r_sq))
    return model, ready_ts, x_list, r_sq


def make_trend_prediction(dta, switch=1, n_days=14):
    model, ts, x_list, r_sq = create_linear_regression_model(dta, switch, n_days)
    for i in range(30):
        input = ts.iloc[-1][x_list]
        pred = ts.iloc[-1]['Values']
        for i in range(n_days + 1, 2, -1):
            name_1 = 'SMA_{}'.format(i)
            name_2 = 'SMA_{}'.format(i-1)
            val_1 = input[name_2]
            input[name_1] = val_1
        input['SMA_2'] = pred
        input = np.array(input).reshape(1, n_days)
        pred = np.ndarray.tolist(model.predict(input))
        input = np.ndarray.tolist(input)[0]
        input.insert(0, pred[0])
        max_index = np.max(ts['Index'])
        input.insert(0, max_index + datetime.timedelta(days=1))
        input = pd.DataFrame([input], columns=ts.columns)
        ts = pd.concat([ts, input])
        ts.reset_index(inplace=True, drop=True)
    plt.plot(ts['Values'])
    plt.show()
    return ts, r_sq


def detrend_data(dt_ts_seasonal):
    """ This function detrends the data using linear Regression ..."""
    x = [i for i in range(np.shape(dt_ts_seasonal)[0])]
    x = np.reshape(x, (len(x), 1))
    y = dt_ts_seasonal.values
    model = LinearRegression()
    model.fit(x, y)
    predicted_y = model.predict(x)
    detrended_y = y - predicted_y
    return x, y, detrended_y


def remove_noise(y, n):
    """ remove noise from any given Ys n is the level of smoothing the higher n will cause lower noise"""
    b = [1.0 / n] * n
    a = 1
    yy = lfilter(b, a, y)
    return yy
