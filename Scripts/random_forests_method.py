# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 18:27:48 2016

@author: rohan
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pandas as pd
import numpy as np

def ensemble_methods(algorithm='randomforests'):
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

    xTrain = XTrain.as_matrix()
    yTrain = YTrain.as_matrix()
    xTest = XTest.as_matrix()
    (rX,cX) = xTrain.shape
    (rT,cT) = xTest.shape
    X = xTrain [0::, 0:cX-1]
    Y = yTrain [0::,]
    T = xTest [0::, 0:cT-1]

    nX = preprocessing.normalize(X)
    sX = preprocessing.scale(nX)
    nT = preprocessing.normalize(T)
    sT = preprocessing.scale(nT)

    forest = RandomForestClassifier(n_estimators = 11)
    forest = forest.fit(sX,Y)
    prediction_labels =  forest.predict_proba(sT)

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
    ensemble_methods(algorithm='randomforests')
