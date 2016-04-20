import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

sns.set_style("white", {'ytick.major.size': 10.0})
sns.set_context("poster", font_scale=1.1)

train_users = pd.read_csv('../Data/train_users_2.csv')
#test = train_users[train_users.country_destination == "NDF"]
#print test.shape

#Data Pruning
train_users_pruned = train_users[train_users.country_destination != "NDF"]
#print train_users_pruned.shape

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

train_users_pruned = train_users_pruned.drop('timestamp_first_active', 1)
#print train_users_pruned['date_account_created']
#print train_users_pruned[:5]
print train_users_pruned[:50]
#print set(train_users_pruned['country_destination'])
