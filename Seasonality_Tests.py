import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from ETL_and_converting_data import Etl


def adf_test_companion(ts_seasonal_dta, country=''):
    df_test = adfuller(ts_seasonal_dta, autolag='AIC')
    index_list = ['Test Statistic', 'p-value', '#Lags Used', '#Obs Used']
    df_output = pd.Series(df_test[0:4], index=index_list)
    for key, value in df_test[4].items():
        df_output['Critical Value (%s)' % key] = value
    print('Test Parameters for {} ================> \n'.format(country), df_output)
    plot_acf(ts_seasonal_dta)
    matplotlib.pyplot.show()


def adf_test(switch):
    """Perform Dickey-Fuller test"""
    data = Etl(switch=switch)
    country = data.select_country()
    total_data = data.convert_order_value_to_ts(column_name='week')
    ts_seasonal = total_data
    print('Results of Dickey-Fuller Test:')
    adf_test_companion(ts_seasonal, country)


"""
Customer Scoring platforms and services from consumer scoring platforms
"""
