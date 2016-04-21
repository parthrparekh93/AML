import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white", {'ytick.major.size': 10.0})
sns.set_context("poster", font_scale=1.1)

train_users = pd.read_csv('train_users_pruned.csv')
test_users = pd.read_csv('test_users.csv')
