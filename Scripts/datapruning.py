import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import math

sns.set_style("white", {'ytick.major.size': 10.0})
sns.set_context("poster", font_scale=1.1)

train_users = pd.read_csv('../Data/train_users_2.csv')
test_users = pd.read_csv('../Data/test_users.csv')
#test = train_users[train_users.country_destination == "NDF"]
#print test.shape

#Data Pruning
train_users_pruned = train_users[train_users.country_destination != "NDF"]
# print train_users_pruned['age'].isnull().sum()
#Average Age
# print train_users.mean(age)
# print train_users_pruned.mean(age)
# print train_users["age"].mean()

#Labeling
train_users_pruned['gender'] = train_users_pruned['gender'].map({'-unknown-': 0, 'MALE': 1, 'FEMALE' : 2})
train_users_pruned['signup_method'] = train_users_pruned['signup_method'].map({'basic': 0, 'google': 1, 'facebook' : 2})
train_users_pruned['language'] = train_users_pruned['language'].map({'el' : 0, 'en' : 1, 'zh' : 2, 'is' : 3, 'it' : 4, 'cs' : 5, 'es' : 6, 'ru' : 7, 'nl' : 8, 'pt' : 9, 'no' : 10, 'tr' : 11, 'th' : 12, 'ca' : 13, 'pl' : 14, 'fr' : 15, 'de' : 16, 'da' : 17, 'fi' : 18, 'hu' : 19, 'ja' : 20, 'ko' : 21, 'sv' : 22})
train_users_pruned['affiliate_channel'] = train_users_pruned['affiliate_channel'].map({'api' : 0, 'remarketing' : 1, 'direct' : 2, 'content' : 3, 'sem-non-brand' : 4, 'other' : 5, 'seo' : 6, 'sem-brand' : 7})
train_users_pruned['affiliate_provider'] = train_users_pruned['affiliate_provider'].map({'email-marketing' : 0, 'facebook-open-graph' : 1, 'google' : 2, 'gsp' : 3, 'baidu' : 4, 'padmapper' : 5, 'bing' : 6, 'direct' : 7, 'yahoo' : 8, 'meetup' : 9, 'craigslist' : 10, 'other' : 11, 'facebook' : 12, 'yandex' : 13, 'naver' : 14, 'vast' : 15, 'daum' : 16})
train_users_pruned['first_affiliate_tracked'] = train_users_pruned['first_affiliate_tracked'].map({'product' : 1, 'omg' : 2, 'tracked-other' : 3, 'marketing' : 4, 'local ops' : 5, 'untracked' : 6, 'linked' : 7})
train_users_pruned['first_affiliate_tracked'].fillna(0, inplace = True)
train_users_pruned['signup_app'] = train_users_pruned['signup_app'].map({'Web' : 0, 'Android' : 1, 'iOS' : 2, 'Moweb' : 3})
train_users_pruned['first_device_type'] = train_users_pruned['first_device_type'].map({'Windows Desktop' : 0, 'iPad' : 1, 'Desktop (Other)' : 2, 'Android Tablet' : 3, 'Other/Unknown' : 4, 'SmartPhone (Other)' : 5, 'Mac Desktop' : 6, 'iPhone' : 7, 'Android Phone' : 8})
train_users_pruned['first_browser'] = train_users_pruned['first_browser'].map({'CometBird':0, '-unknown-' :1, 'Chrome':2, 'Stainless':3, 'SeaMonkey':4, 'Maxthon':5, 'TheWorld Browser':6, 'wOSBrowser':7, 'Apple Mail':8, 'AOL Explorer':9, 'Android Browser':10, 'Yandex.Browser':11, 'Avant Browser':12, 'SiteKiosk':13, 'CoolNovo':14, 'RockMelt':15, 'Mobile Safari':16, 'Camino':17, 'Sogou Explorer':18, 'Safari':19, 'IE Mobile':20, 'Pale Moon':21, 'Silk':22, 'BlackBerry Browser':23, 'Kindle Browser':24, 'Opera Mini':25, 'SlimBrowser':26, 'Opera':27, 'Chrome Mobile':28, 'Palm Pre web browser':29, 'Iron':30, 'IE':31, 'IceWeasel':32, 'Firefox':33, 'Googlebot':34, 'Mozilla':35, 'NetNewsWire':36, 'TenFourFox':37, 'Chromium':38, 'Mobile Firefox':39})
train_users_pruned['country_destination'] = train_users_pruned['country_destination'].map({'FR':0, 'NL':1, 'PT':2, 'CA':3, 'DE':4, 'IT':5, 'US':6, 'other':7, 'AU':8, 'GB':9, 'ES':10})


test_users['gender'] = test_users['gender'].map({'-unknown-': 0, 'MALE': 1, 'FEMALE' : 2})
test_users['signup_method'] = test_users['signup_method'].map({'basic': 0, 'google': 1, 'facebook' : 2})
test_users['language'] = test_users['language'].map({'el' : 0, 'en' : 1, 'zh' : 2, 'is' : 3, 'it' : 4, 'cs' : 5, 'es' : 6, 'ru' : 7, 'nl' : 8, 'pt' : 9, 'no' : 10, 'tr' : 11, 'th' : 12, 'ca' : 13, 'pl' : 14, 'fr' : 15, 'de' : 16, 'da' : 17, 'fi' : 18, 'hu' : 19, 'ja' : 20, 'ko' : 21, 'sv' : 22})
test_users['affiliate_channel'] = test_users['affiliate_channel'].map({'api' : 0, 'remarketing' : 1, 'direct' : 2, 'content' : 3, 'sem-non-brand' : 4, 'other' : 5, 'seo' : 6, 'sem-brand' : 7})
test_users['affiliate_provider'] = test_users['affiliate_provider'].map({'email-marketing' : 0, 'facebook-open-graph' : 1, 'google' : 2, 'gsp' : 3, 'baidu' : 4, 'padmapper' : 5, 'bing' : 6, 'direct' : 7, 'yahoo' : 8, 'meetup' : 9, 'craigslist' : 10, 'other' : 11, 'facebook' : 12, 'yandex' : 13, 'naver' : 14, 'vast' : 15, 'daum' : 16})
test_users['first_affiliate_tracked'] = test_users['first_affiliate_tracked'].map({'product' : 1, 'omg' : 2, 'tracked-other' : 3, 'marketing' : 4, 'local ops' : 5, 'untracked' : 6, 'linked' : 7})
test_users['first_affiliate_tracked'].fillna(0, inplace = True)
test_users['signup_app'] = test_users['signup_app'].map({'Web' : 0, 'Android' : 1, 'iOS' : 2, 'Moweb' : 3})
test_users['first_device_type'] = test_users['first_device_type'].map({'Windows Desktop' : 0, 'iPad' : 1, 'Desktop (Other)' : 2, 'Android Tablet' : 3, 'Other/Unknown' : 4, 'SmartPhone (Other)' : 5, 'Mac Desktop' : 6, 'iPhone' : 7, 'Android Phone' : 8})
test_users['first_browser'] = test_users['first_browser'].map({'CometBird':0, '-unknown-' :1, 'Chrome':2, 'Stainless':3, 'SeaMonkey':4, 'Maxthon':5, 'TheWorld Browser':6, 'wOSBrowser':7, 'Apple Mail':8, 'AOL Explorer':9, 'Android Browser':10, 'Yandex.Browser':11, 'Avant Browser':12, 'SiteKiosk':13, 'CoolNovo':14, 'RockMelt':15, 'Mobile Safari':16, 'Camino':17, 'Sogou Explorer':18, 'Safari':19, 'IE Mobile':20, 'Pale Moon':21, 'Silk':22, 'BlackBerry Browser':23, 'Kindle Browser':24, 'Opera Mini':25, 'SlimBrowser':26, 'Opera':27, 'Chrome Mobile':28, 'Palm Pre web browser':29, 'Iron':30, 'IE':31, 'IceWeasel':32, 'Firefox':33, 'Googlebot':34, 'Mozilla':35, 'NetNewsWire':36, 'TenFourFox':37, 'Chromium':38, 'Mobile Firefox':39})




#Date change
#train_users_pruned['date_account_created'] =  pd.to_datetime(train_users_pruned['date_account_created'], format='%d%b%Y:%H:%M:%S.%f')
train_users_pruned['year_created'] = pd.DatetimeIndex(train_users_pruned['date_account_created']).year
train_users_pruned['month_created'] = pd.DatetimeIndex(train_users_pruned['date_account_created']).month
train_users_pruned['date_created'] = pd.DatetimeIndex(train_users_pruned['date_account_created']).day
train_users_pruned = train_users_pruned.drop('date_account_created', 1)

train_users_pruned['year_booked'] = pd.DatetimeIndex(train_users_pruned['date_first_booking']).year
train_users_pruned['month_booked'] = pd.DatetimeIndex(train_users_pruned['date_first_booking']).month
train_users_pruned['date_booked'] = pd.DatetimeIndex(train_users_pruned['date_first_booking']).day
train_users_pruned = train_users_pruned.drop('date_first_booking', 1)


test_users['year_created'] = pd.DatetimeIndex(test_users['date_account_created']).year
test_users['month_created'] = pd.DatetimeIndex(test_users['date_account_created']).month
test_users['date_created'] = pd.DatetimeIndex(test_users['date_account_created']).day
test_users = test_users.drop('date_account_created', 1)

test_users['year_booked'] = pd.DatetimeIndex(test_users['date_first_booking']).year
test_users['month_booked'] = pd.DatetimeIndex(test_users['date_first_booking']).month
test_users['date_booked'] = pd.DatetimeIndex(test_users['date_first_booking']).day
test_users = test_users.drop('date_first_booking', 1)

train_users_pruned = train_users_pruned.drop('timestamp_first_active', 1)
train_users_pruned.loc[train_users_pruned.age > 95, 'age'] = np.nan
train_users_pruned.loc[train_users_pruned.age < 13, 'age'] = np.nan


test_users = test_users.drop('timestamp_first_active', 1)
test_users.loc[test_users.age > 95, 'age'] = np.nan
test_users.loc[test_users.age < 13, 'age'] = np.nan
#print train_users_pruned['date_account_created']
#print train_users_pruned[:5]

#print set(train_users_pruned['country_destination'])

# train_users_pruned['age_range'] = pd.cut(train_users_pruned['age'], bins = [0,10,20,30,40,50,60,70,80,90,100,110,120], labels=False)
# labels = np.array('1-10 11-20 21-30 31-40 41-50 51-60 61-70 71-80 81-90 91-100 101-110 111-120'.split())
# train_users_pruned['age_range'] = labels[train_users_pruned['age_range']]
# del train_users_pruned['age']
# print train_users_pruned[:50]
new_age = []
for index,row in train_users_pruned.iterrows():
    if not math.isnan(row['age']):
        new_age.append(int(row['age']/10))
    else:
        new_age.append(float('nan'))

new_age_test = []
for index,row in test_users.iterrows():
    if not math.isnan(row['age']):
        new_age_test.append(int(row['age']/10))
    else:
        new_age_test.append(float('nan'))

train_users_pruned['bucket_age'] = new_age
print train_users_pruned['bucket_age'].median()
train_users_pruned['bucket_age'].fillna(train_users_pruned['bucket_age'].median(), inplace="True")
train_users_pruned = train_users_pruned.drop('age', 1)
print train_users_pruned[:50]


test_users['bucket_age'] = new_age_test
print test_users['bucket_age'].median()
test_users['bucket_age'].fillna(test_users['bucket_age'].median(), inplace="True")
test_users = test_users.drop('age', 1)
print test_users[:50]

train_users_pruned.to_csv('train_users_pruned.csv', sep='\t', encoding='utf-8')
test_users.to_csv('test_users.csv', sep='\t', encoding='utf-8')
