import pandas as pd
import numpy as np


def explore_restaurants_retention(Dta):
    """ create a valid data based on the number of customer scoring platforms """
    table = pd.pivot_table(Dta, values='Products in Order',
                           index=['Restaurant ID'],
                           columns=['week'], aggfunc='sum', fill_value=0)
    df = pd.DataFrame(table.to_records())
    df.set_index('Restaurant ID')
    recent_activity = df[['7', '8', '9']].sum(axis=1) / 3
    start_activity = df[['1', '2', '3']].sum(axis=1) / 3
    restaurant_performance = pd.concat([df['Restaurant ID'], recent_activity, start_activity], axis=1)
    restaurant_performance['Animal'] = 'Frog'
    df_test = df[['1', '2', '3']]
    df_test_1 = df_test.loc[(df_test['1'] > 0) & (df_test['2'] > 0) & (df_test['3'] > 0)]

    df_test = df[['7', '8', '9']]
    df_test_2 = df_test.loc[(df_test['7'] > 0) & (df_test['8'] > 0) & (df_test['9'] > 0)]
    threshold_1 = sum(df_test_1.quantile(0.45, axis=0)) / 3
    threshold_2 = sum(df_test_2.quantile(0.45, axis=0)) / 3

    restaurant_performance.loc[(restaurant_performance[0] > restaurant_performance[1]) &
                               (restaurant_performance[0] > threshold_1) &
                               (restaurant_performance[1] < threshold_2), 'Animal'] = 'Elephant'

    restaurant_performance.loc[(restaurant_performance[0] < restaurant_performance[1]) &
                               (restaurant_performance[1] > threshold_2), 'Animal'] = 'Lion'

    restaurant_performance.loc[(restaurant_performance[0] < restaurant_performance[1]) &
                               (restaurant_performance[0] < threshold_1) &
                               (restaurant_performance[1] > threshold_2), 'Animal'] = 'Eagle'

    restaurant_performance.loc[(restaurant_performance[0] > restaurant_performance[1]) &
                               (restaurant_performance[0] < threshold_1) &
                               (restaurant_performance[1] < threshold_2), 'Animal'] = 'Frog'
    return restaurant_performance[['Restaurant ID', 'Animal']]


def explore_delivery_time(Dta):
    table = pd.pivot_table(Dta, values='Delivery Time',
                           index=['Restaurant ID'],
                           # columns=['week'],
                           aggfunc=np.average, fill_value=0)
    df = pd.DataFrame(table.to_records())
    threshold = df.quantile(0.35, axis=0)[1]
    laggies_data = df.loc[df['Delivery Time'] > threshold]
    print(laggies_data)
    return laggies_data
