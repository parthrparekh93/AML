# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:54:21 2016

@author: Rohan Kulkarni
@email : rohan.kulkarni@gmail.com

"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

def main():
    # shape (#rows,18)
    train_users_raw = pd.read_csv('train_users_pruned.csv',delimiter=',',encoding='utf-8')
    test_users_raw = pd.read_csv('test_users.csv',delimiter=',',encoding='utf-8')


    del train_users_raw['id']
    del test_users_raw['id']

    train_users_raw=train_users_raw.drop(train_users_raw.columns[[0]], axis=1)
    test_users_raw=test_users_raw.drop(test_users_raw.columns[[0]], axis=1)

    country_destination = train_users_raw['country_destination']
    del train_users_raw['country_destination']

    selector = VarianceThreshold(threshold=2.0)
    selector.fit(train_users_raw)
    selected_col_ind = selector.get_support(indices=True)
    selected_col_ind =  np.append(selected_col_ind, train_users_raw.shape[1]-1)
    #print selected_col_ind
    #print train_users_raw.columns.values
    # shape (#rows,11)
    train_users_downsized = train_users_raw.ix[:,selected_col_ind]
    del train_users_downsized['month_booked']
    del train_users_downsized['date_booked']
    train_users_downsized['country_destination'] = country_destination
    print train_users_downsized.columns.values
    test_users_downsized = test_users_raw.ix[:,selected_col_ind]
    del test_users_downsized['month_booked']
    del test_users_downsized['date_booked']
    print test_users_downsized.columns.values

    train_users_downsized.to_csv('training_data_processed.csv', sep=',', encoding='utf-8')
    test_users_downsized.to_csv('testing_data_processed.csv', sep=',', encoding='utf-8')



if __name__=='__main__':
    main()
