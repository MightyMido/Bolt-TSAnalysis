import pandas as pd


class Etl:
    """ manages the reading and transforming Data from file """
    # todo: May be connecting to a database like MsSQL server or Oracle is a good Idea
    def __init__(self, switch):
        self.file_name = "Central Operations Specialist - Data for Home Task - Data.csv"
        self.switch = switch
        self.dta = self.separate_by_country()

    def select_country(self):
        if self.switch == 1:
            return 'Portugal'
        elif self.switch == 2:
            return 'Ghana'

    def read_data(self):
        """reads and creates dataframe"""
        data = pd.read_csv(self.file_name)
        data[['Day', 'Month', 'Year']] = data['Created Date'].str.split('.', 2, expand=True)
        data['Created Date'] = pd.to_datetime(data['Year'].astype(str) + '-' +
                                              data['Month'].astype(str) + '-' +
                                              data['Day'].astype(str))
        data['Order Value € (Gross)'] = data['Order Value € (Gross)'].str.replace('€', '')
        data['Order Value € (Gross)'] = pd.to_numeric(data['Order Value € (Gross)'])
        data['week'] = data['Created Date'].dt.isocalendar().week
        data.set_index('Created Date')
        data.sort_index(ascending=True)
        return data

    def separate_by_country(self):
        """separate data into two different DataFrame References"""
        country = self.select_country()
        dta = self.read_data()
        data = dta[dta['Country'] == country]
        return data

    def convert_order_value_to_ts(self, dta_ts='', group_by_column='Created Date', column_name='Order Value € (Gross)'):
        """ creates data frame timeSeries to work with ..."""
        if dta_ts == '':
            dta = self.dta
            dta_ts = dta.groupby([group_by_column]).sum()
        dta_ts_seasonal = dta_ts[column_name].copy()
        dta_ts_seasonal.reindex()
        ready_ts = pd.DataFrame({'Index': list(dta_ts_seasonal.index), 'Values': list(dta_ts_seasonal.values)})
        return ready_ts

    def adjust_random_walks(self, main_weight=0.8):
        """ main weight means the volume of  customer weights"""
        denoised_ts = self.convert_order_value_to_ts()
        denoised_ts = LaggedTimeSeries(n_days=14, ts_data=denoised_ts)
        return denoised_ts.adjust_random_walks(main_weight=main_weight)

    def create_time_lags(self, n_days):
        """" x_list Is a container for X variables names """
        ts_data = self.convert_order_value_to_ts()
        ready_ts, x_list = LaggedTimeSeries.create_time_lags(n_days=n_days, ts_data=ts_data)
        return ready_ts, x_list


class LaggedTimeSeries:
    def __init__(self, n_days, ts_data):
        """ ts_data must have two columns only an Index column and a value columns """
        self.n_days = n_days
        self.data = ts_data

    def adjust_random_walks(self, main_weight=0.8):
        """ main weight means the volume of  customer weights"""
        lateral_weight = 1 - main_weight
        cntr = 0
        denoised_ts = self.data
        for row in denoised_ts.iterrows():
            if cntr > 0:
                denoised_ts.iloc[cntr] = row['Values'] * main_weight + denoised_ts.iloc[cntr - 1] * lateral_weight
                cntr += 1
        return denoised_ts

    def create_time_lags(self):
        """" x_list Is a container for X variables names """
        n_days = self.n_days
        x_list = []
        ready_ts = self.adjust_random_walks()
        for i in range(2, n_days + 2):
            name = 'SMA_{}'.format(i)
            x_list.append(name)
            ready_ts[name] = ready_ts['Values'].shift(i)
            """ shifts dataframes Values 1 row at a Time ... """
        ready_ts = ready_ts.dropna()
        ready_ts.set_index('Index', inplace=True)
        return ready_ts, x_list
