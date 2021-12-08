import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import warnings
import itertools
import statsmodels.api as sm
import matplotlib
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import LinearRegression
import datetime
from UsingLinearRegression import *
from Using_Decomposition import DecomposedTimeSeries


warnings.filterwarnings("ignore", category=UserWarning)
url = "https://docs.google.com/spreadsheets/d/1TLC-f0kA55ZrfNX9er_7bCBJrTc_AIn6viK6esU5WqE/edit#gid=1581757308"
file_name = "Central Operations Specialist - Data for Home Task - Data.csv"


def select_country(switch):
    if switch == 1:
        return 'Portugal'
    elif switch == 2:
        return 'Ghana'


def read_data():
    """reads and creates dataframe"""
    data = pd.read_csv(file_name)
    data[['Day', 'Month', 'Year']] = data['Created Date'].str.split('.', 2, expand=True)
    data['Created Date'] = pd.to_datetime(data['Year'].astype(str) + '-' +
                                          data['Month'].astype(str) + '-' +
                                          data['Day'].astype(str))
    data['Order Value € (Gross)'] = data['Order Value € (Gross)'].str.replace('€', '')
    data['Order Value € (Gross)'] = pd.to_numeric(data['Order Value € (Gross)'])
    # extract week from created Date Data
    data['week'] = data['Created Date'].dt.isocalendar().week
    # set and sort indexes for further use in ts analysis
    data.set_index('Created Date')
    data.sort_index(ascending=True)
    return data


def separate_by_country(dta):
    """separate data into two different DataFrame References"""
    # print(dta)
    portugal_data = dta[dta['Country'] == "Portugal"]
    ghana_data = dta[dta['Country'] == "Ghana"]
    return portugal_data, ghana_data


def convert_order_value_to_ts(Dta, group_by_column='Created Date', column_name='Order Value € (Gross)'):
    """ creates data frame timeSeries to work with ..."""
    dta_ts = Dta.groupby([group_by_column]).sum()
    dta_ts_seasonal = dta_ts[column_name].copy()
    dta_ts_seasonal.reindex()
    return dta_ts_seasonal


def decompose_components(dta, switch=1):
    """ Decomposes into two different Multiplicative and Additive data for TimeSeries Analysis !!! """
    # Multiplicative Decomposition
    portugal_data, ghana_data = separate_by_country(dta)
    if switch == 1:
        df = convert_order_value_to_ts(portugal_data)
    elif switch == 2:
        df = convert_order_value_to_ts(ghana_data)
    result_mul = seasonal_decompose(df, model='multiplicative', extrapolate_trend='freq')

    # Additive Decomposition
    result_add = seasonal_decompose(df, model='additive', extrapolate_trend='freq')
    return result_add, result_mul


def using_decompose(dta, switch=1):
    _add, _mul = decompose_components(dta, switch)
    _seasonal = _add.seasonal
    _trend = _add.trend
    _resid = _add.resid
    total = pd.DataFrame({'Index': list(_resid.index),
                          'X': [i for i in range(0, len(list(_resid.index)))],
                          '_trend': list(_trend),
                          '_seasonal': list(_seasonal),
                          '_resid': list(_resid.values)})
    total.set_index('Index', inplace=True)

    _seasonal = _mul.seasonal
    _trend = _mul.trend
    _resid = _mul.resid
    m_total = pd.DataFrame({'Index': list(_resid.index),
                            'X': [i for i in range(0, len(list(_resid.index)))],
                            '_trend': list(_trend),
                            '_seasonal': list(_seasonal),
                            '_resid': list(_resid.values)})
    m_total.set_index('Index', inplace=True)
    return total, m_total


def using_residuals(dta, switch=1):
    total, m_total = using_decompose(dta, switch)
    country = select_country(switch)
    total_resid = total['_resid']
    result_add = seasonal_decompose(total_resid, model='additive', extrapolate_trend='freq')
    plt.rcParams.update({'figure.figsize': (10, 10)})
    result_add.plot().suptitle('Additive Decompose({})'.format(country), x=0.15, y=0.99, fontsize=15)
    plt.show()
    return result_add


def plot_shapes(dta, switch=1):
    if switch == 1:
        country = 'Portugal'
    elif switch == 2:
        country = 'Ghana'
    res_add, res_mul = decompose_components(dta, switch)
    plt.rcParams.update({'figure.figsize': (10, 10)})
    res_mul.plot().suptitle('Multiplicative Decompose({})'.format(country), x=0.16, y=0.99, fontsize=15)
    res_add.plot().suptitle('Additive Decompose({})'.format(country), x=0.15, y=0.99, fontsize=15)
    plt.show()


def adf_test_companion(ts_seasonal_dta, country=''):
    df_test = adfuller(ts_seasonal_dta, autolag='AIC')
    index_list = ['Test Statistic', 'p-value', '#Lags Used', '#Obs Used']
    df_output = pd.Series(df_test[0:4], index=index_list)
    for key, value in df_test[4].items():
        df_output['Critical Value (%s)' % key] = value
    print('Test Parameters for {} ================>\n'.format(country), df_output)
    plot_acf(ts_seasonal_dta)
    matplotlib.pyplot.show()


def adf_test(Dta):
    """Perform Dickey-Fuller test"""
    portugal_data, ghana_data = separate_by_country(Dta)
    ghana_ts_seasonal = convert_order_value_to_ts(ghana_data, 'week')
    portugal_ts_seasonal = convert_order_value_to_ts(portugal_data, 'week')
    print('Results of Dickey-Fuller Test:')
    adf_test_companion(ghana_ts_seasonal, 'Ghana')
    adf_test_companion(portugal_ts_seasonal, 'Portugal')


def make_prediction_sarima(dta):
    """
    this would be the SARIMA model and it works
    but due to lack of data in this case It will not work well
    it would be better top use simple regression
    with the values of 7 days moving average
    """
    portugal_data, ghana_data = separate_by_country(dta)
    ghana_ts_seasonal = convert_order_value_to_ts(ghana_data)
    portugal_ts_seasonal = convert_order_value_to_ts(portugal_data)
    p = range(0, 3)
    d = range(1, 2)
    q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 5) for x in list(itertools.product(p, d, q))]
    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
    for param in pdq:
        try:
            for param_seasonal in seasonal_pdq:
                mod = sm.tsa.statespace.SARIMAX(ghana_ts_seasonal,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except Exception as E:
            print(E)

    mod = sm.tsa.statespace.SARIMAX(ghana_ts_seasonal.values,
                                    order=(2, 1, 2),
                                    seasonal_order=(2, 1, 1, 5))
    results = mod.fit(method='powell')
    print(results.summary().tables[1])
    pred = results.get_prediction('2020-03-20', dynamic=False)
    print(pred)
    return


def create_linear_regression_model(dta, switch=1, n_days=14):
    portugal_data, ghana_data = separate_by_country(dta)
    if switch == 1:
        ts_seasonal = convert_order_value_to_ts(portugal_data)
    elif switch == 2:
        ts_seasonal = convert_order_value_to_ts(ghana_data)
    ready_ts = pd.DataFrame({'Index': list(ts_seasonal.index), 'Values': list(ts_seasonal.values)})
    cntr = 0
    for row in ready_ts.iterrows():
        if cntr > 0:
            ready_ts.iloc[cntr] = row['Values'] * 0.9 + ready_ts.iloc[cntr - 1] * 0.1
            cntr += 1
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


def find_best_lag(dta, switch):
    r_t = 0
    best_shift = 0
    for i in range(1, 15):
        results, r_sq = make_trend_prediction(dta, switch=switch, n_days=i)
        if r_sq - r_t > 0.03 or r_sq < 0.70:
            r_t = r_sq
            total_result = results
            best_shift = i
        else:
            break
    return best_shift, total_result


if __name__ == "__main__":
    data = read_data()
    # print(data.columns)
    # explore data for seasonal Variables
    # explore_for_seasonal(data)
    # Explore data for processing time of delivery
    # r_p = explore_restaurants_retention(Dta=data)
    # print(r_p)
    # explore_delivery_time(data)
    # make_prediction(data)
    # adf_test(data)
    # result = make_trend_prediction(data, switch=1)
    # make_prediction_SARIMA(data)
    # res_add, res_mul = decompose_components(data, switch=1)
    # print(res_add.resid)
    total = using_decompose(data, switch=1)
    # print(total)
    # plt.plot(total)
    # plt.show()
    # plot_shapes(data, switch=2)
    # print(result)
    # (1, 12)
    best_shift, total_result = find_best_lag(data, switch=1)
    ts_a, r_sq_ = make_trend_prediction(data, switch=1, n_days=best_shift)
    # plt.plot(ts_a['Values'])
    # plt.show()
    # print('best number_of_shifts ==> {}'.format(best_shift))
    # print(r_sq_)
    using_residuals(data, 2)
    # print(total)
    # make_prediction_SARIMA(data)
    # print(results)
    print('I am the man')
