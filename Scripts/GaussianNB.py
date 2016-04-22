import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

train_users_raw = pd.read_csv('training_data_processed.csv',delimiter=',',encoding='utf-8')
test_users_raw = pd.read_csv('testing_data_processed.csv',delimiter=',',encoding='utf-8')

country_mapping = {0:'FR', 1:'NL', 2:'PT', 3:'CA', 4:'DE', 5:'IT', 6:'US', 7:'other', 8:'AU', 9:'GB', 10:'ES', 11:'NDF'}

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
user_id = test_users_raw['id']
del test_users_raw['id']
Xtest = test_users_raw

#print Xtest.isnull().sum()

#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
#Y = np.array([1, 1, 1, 2, 2, 2])
clf = GaussianNB()
clf.fit(X, Y)
#print(clf.predict([0,1.0,2,7,6.0,7,16.0,7,1,3.0]))

ranked_result_array = []
result_array = clf.predict_proba(Xtest)

user_id_final = []
for user in user_id:
    user_id_final.append(user)
    user_id_final.append(user)
    user_id_final.append(user)
    user_id_final.append(user)
    user_id_final.append(user)


for test_row in result_array:
    ranked_result_array.append(country_mapping[test_row.argsort()[-5:][::-1][0]])
    ranked_result_array.append(country_mapping[test_row.argsort()[-5:][::-1][1]])
    ranked_result_array.append(country_mapping[test_row.argsort()[-5:][::-1][2]])
    ranked_result_array.append(country_mapping[test_row.argsort()[-5:][::-1][3]])
    ranked_result_array.append(country_mapping[test_row.argsort()[-5:][::-1][4]])

#print len(user_id_final)
#print len(ranked_result_array)
finalcsv = pd.DataFrame(np.column_stack((user_id_final, ranked_result_array)), columns=['id', 'country'])
finalcsv.to_csv('finalcsv.csv',index=False)
print finalcsv[:28]
