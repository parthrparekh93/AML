# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 18:27:48 2016

@author: rohan
"""

from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from xgboost.sklearn import XGBClassifier


def main():
    iris = load_iris()
    clf = AdaBoostClassifier(n_estimators=100)
    print clf
    scores = cross_val_score(clf, iris.data, iris.target)
    print scores.mean()
    
    
if __name__=='__main__':
    main()