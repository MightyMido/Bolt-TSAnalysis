import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import warnings
from ETL_and_converting_data import Etl, LaggedTimeSeries
from sklearn.linear_model import LinearRegression


warnings.filterwarnings("ignore", category=UserWarning)


class DecomposedTimeSeries:
    def __init__(self, ts, country, decomposition_type=1):
        """
        country switch 1 => Portugal and 2 => Ghana ...
        decomposition type  1 => additive and 2 => multiplicative ...
        """
        self.ts = ts
        self.decomposition_type = decomposition_type
        self.country_name = country

    def select_decomposition_type(self):
        if self.decomposition_type == 1:
            return "additive"
        if self.decomposition_type == 2:
            return "multiplicative"

    def decompose_components(self):
        """ Decomposes into two different Multiplicative and Additive data for TimeSeries Analysis !!! """
        decomposition_type = self.select_decomposition_type()
        ts = self.ts
        ts.set_index('Index', inplace=True)
        result = seasonal_decompose(ts, model=decomposition_type, extrapolate_trend='freq')
        return result

    def using_decompose(self, _add=''):
        if _add == '':
            _add = self.decompose_components()
        _seasonal = _add.seasonal
        _trend = _add.trend
        _resid = _add.resid
        total = pd.DataFrame({'Index': list(_resid.index),
                              'X': [i for i in range(0, len(list(_resid.index)))],
                              '_trend': list(_trend),
                              '_seasonal': list(_seasonal),
                              '_resid': list(_resid.values)})
        total.set_index('Index', inplace=True)
        return total

    def using_residuals(self, with_figures=0):
        total = self.using_decompose()
        total_resid = total['_resid']
        result_add = seasonal_decompose(total_resid, model='additive', extrapolate_trend='freq')
        plt.rcParams.update({'figure.figsize': (10, 10)})
        if with_figures == 1:
            result_add.plot().suptitle('Additive Decompose({})'.format(self.country_name), x=0.15, y=0.99, fontsize=15)
            plt.show()
        total = self.using_decompose(_add=result_add)
        return total


def multiple_residual_extraction(n, x_ts, with_figure=0):
    trends = pd.DataFrame()
    seasonals = pd.DataFrame()
    for i in range(n):
        dcts = DecomposedTimeSeries(ts=x_ts, country='Portugal', decomposition_type=1)
        x_obj = dcts.using_residuals(with_figures=with_figure)
        x_ts = x_obj['_resid']
        colname_t = 'trend_{}'.format(i)
        colname_s = 'seasonal_{}'.format(i)
        trends[colname_t] = x_obj['_trend']
        seasonals[colname_s] = x_obj['_seasonal']
        index = list(x_ts.index)
        values = list(x_ts.values)
        x_ts = pd.DataFrame({'Index': index, 'Values': values})
    return seasonals, trends, x_ts


def model_seasonal():
    return

def extract_trend(Y):
    trend_model = LinearRegression()
    x = np.array([i for i in range(0, len(Y))])
    trend_model.fit(x.reshape(-1, 1), Y)
    trend_of_data = trend_model.predict(x.reshape(-1, 1))
    detrended_data = Y - trend_of_data
    return trend_model, x, trend_of_data, detrended_data


def convert_row_number_to_season(seasonal_tbl, row_id):
    x = np.array([i for i in range(0, len(seasonal_tbl))])
    seasonal['row'] = x
    x_list = [col for col in seasonal_tbl.columns if 'seasonal_' in col]
    # Todo must select unique values based on the amount
    real_row_id = row_id % 7
    seasonal_variable = seasonal.loc[seasonal['row'] == real_row_id]
    seasonal_var = seasonal_variable[x_list]
    return seasonal_var


def make_prediction(r_id, trend_model, seasonal_tbl, seasonal_model):
    seasonal_data = convert_row_number_to_season(seasonal_tbl=seasonal_tbl, row_id=r_id)
    predicted_seasonal = seasonal_model.predict(seasonal_data)
    s = np.array(r_id)
    s = s.reshape(1, -1)
    predicted_value = trend_model.predict(s) + predicted_seasonal
    return predicted_value


if __name__ == '__main__':
    data = Etl(switch=2).convert_order_value_to_ts()
    x = np.array([i for i in range(0, len(data))])
    model, x, trend_of_data, detrended_data = extract_trend(data['Values'])
    seasonal, trends, residuals = multiple_residual_extraction(n=7, x_ts=data, with_figure=0)
    y_1 = residuals['Values']
    x = np.array([i for i in range(0, 90)])
    pred = model.predict(np.reshape(x, (-1, 1)))

    seasonal_model = LinearRegression()
    seasonal_model.fit(seasonal, detrended_data)
    predicted = seasonal_model.predict(seasonal)
    plt.plot(data)
    forecast_list =[]
    for i in range(0, 59):
        pred = make_prediction(r_id=i, trend_model=model, seasonal_tbl=seasonal, seasonal_model=seasonal_model)
        forecast_list.append(pred[0])
    forecast_list = np.array(forecast_list)
    print(forecast_list)
    data['forecast'] = forecast_list
    plt.plot(data)
    plt.show()
    data['residuals'] = abs(data['Values'] - data['forecast'])
    print(data.sum()/60)
    plt.plot(data['residuals'])
    plt.show()
