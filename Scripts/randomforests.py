# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 18:27:48 2016

@author: viral
"""
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

def randomForestsClassificaion(XTrain,YTrain,XTest):
    forest = RandomForestClassifier(n_estimators = 12)
    forest = forest.fit(XTrain,YTrain)
    classification =  forest.predict(XTest)
    return classification

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

    if algorithm == 'randomforests':
        print "Training Random Forests Classifier"
        prediction_labels = randomForestsClassification(XTrain,YTrain,XTest)

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
