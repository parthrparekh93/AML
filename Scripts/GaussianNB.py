import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

train_users_raw = pd.read_csv('training_data_processed.csv',delimiter=',',encoding='utf-8')
test_users_raw = pd.read_csv('testing_data_processed.csv',delimiter=',',encoding='utf-8')

train_users_raw=train_users_raw.drop(train_users_raw.columns[[0]], axis=1)
test_users_raw=test_users_raw.drop(test_users_raw.columns[[0]], axis=1)

#Just to deal with some NAN values in test data
#print test_users_raw['language'].median()
#print test_users_raw['first_browser'].median()
test_users_raw['language'].fillna(test_users_raw['language'].median(), inplace="True")
test_users_raw['first_browser'].fillna(test_users_raw['first_browser'].median(), inplace="True")
#print train_users_raw[:10]
#print test_users_raw[:10]

Y = train_users_raw['country_destination']
del train_users_raw['country_destination']
X = train_users_raw
Xtest = test_users_raw

#print Xtest.isnull().sum()

#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
#Y = np.array([1, 1, 1, 2, 2, 2])
clf = GaussianNB()
clf.fit(X, Y)
#print(clf.predict([0,1.0,2,7,6.0,7,16.0,7,1,3.0]))
print(clf.predict_proba(Xtest))
