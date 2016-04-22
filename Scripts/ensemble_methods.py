# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 18:27:48 2016

@author: rohan
"""

from sklearn.ensemble import AdaBoostClassifier
from xgboost.sklearn import XGBClassifier
import pandas as pd
import numpy as np


def adaboost_algorithm(XTrain,YTrain,XTest):
    adb = AdaBoostClassifier(n_estimators=100)
    adb.fit(XTrain, YTrain)
    y_pred_adaboost = adb.predict_proba(XTest)
    return y_pred_adaboost
    
def xgboost_algorithm(XTrain,YTrain,XTest):
    xgb = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)                  
    xgb.fit(XTrain, YTrain)
    y_pred_xgboost = xgb.predict_proba(XTest) 
    return y_pred_xgboost
    
def ensemble_methods(algorithm='xgboost'):
    XTrain = pd.read_csv('training_data_processed.csv',delimiter=',',encoding='utf-8')
    XTest= pd.read_csv('testing_data_processed.csv',delimiter=',',encoding='utf-8')  
    user_ids = XTest['id']  
    YTrain = XTrain['country_destination']
    del XTrain['country_destination']
    del XTest['id']
    
    XTrain=XTrain.drop(XTrain.columns[[0]], axis=1)
    XTest=XTest.drop(XTest.columns[[0]], axis=1)
    
    XTest['language'].fillna(XTrain['language'].median(), inplace="True")
    XTest['first_browser'].fillna(XTest['first_browser'].median(), inplace="True")
  
    if algorithm == 'adaboost':
        print 
        print 'Training AdaBoost Classifier...'
        prediction_labels = adaboost_algorithm(XTrain,YTrain,XTest)
    
    elif algorithm == 'xgboost':
        print 
        print 'Training XGBoost Classifier...'
        prediction_labels = xgboost_algorithm(XTrain,YTrain,XTest)
    
    country_mapping = {0:'FR', 1:'NL', 2:'PT', 3:'CA', 4:'DE', 5:'IT', 6:'US', 7:'other', 8:'AU', 9:'GB', 10:'ES', 11:'NDF'}
    ranked_results=np.array([[country_mapping[x] for x in row.argsort()[-5:][::-1]] for row in prediction_labels])
    ranked_results_flattened = ranked_results.flatten()
    
    user_ids_repeat = list()
    for id in user_ids:
        for i in xrange(5):
            user_ids_repeat.append(id)
    
    submission_result = pd.DataFrame(np.column_stack((user_ids_repeat, ranked_results_flattened)), columns=['id', 'country'])
    submission_result.to_csv('../Results/'+algorithm+'Results.csv',index=False)    
    
    
if __name__=='__main__':
    ensemble_methods(algorithm='adaboost')