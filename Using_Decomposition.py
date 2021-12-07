import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import warnings
from ETL_and_converting_data import Etl


warnings.filterwarnings("ignore", category=UserWarning)


class DecomposedTimeSeries:
    def __init__(self, country_switch=1, decomposition_type=1):
        """
        country switch 1 => Portugal and 2 => Ghana ...
        decomposition type  1 => additive and 2 => multiplicative ...
        """
        self.data = Etl(country_switch)
        self.decomposition_type = decomposition_type
        self.country_name = self.data.select_country()

    def select_decomposition_type(self):
        if self.decomposition_type == 1:
            return "additive"
        if self.decomposition_type == 2:
            return "multiplicative"

    def decompose_components(self):
        """ Decomposes into two different Multiplicative and Additive data for TimeSeries Analysis !!! """
        decomposition_type = self.select_decomposition_type()
        df = self.data.convert_order_value_to_ts()
        result = seasonal_decompose(df, model=decomposition_type, extrapolate_trend='freq')
        return result

    def using_decompose(self):
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

    def using_residuals(self):
        total = self.using_decompose()
        total_resid = total['_resid']
        result_add = seasonal_decompose(total_resid, model='additive', extrapolate_trend='freq')
        plt.rcParams.update({'figure.figsize': (10, 10)})
        result_add.plot().suptitle('Additive Decompose({})'.format(self.country_name), x=0.15, y=0.99, fontsize=15)
        plt.show()
        return result_add


    def extract_all_seasonals(self):
        return