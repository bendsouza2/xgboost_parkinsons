import pandas as pd
import numpy as np
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('parkinsons.data')

# print(df.dtypes)

x = df.loc[:, df.columns != "status"].to_numpy()[:, 1:]  # features
y = df.loc[:, 'status'].to_numpy()  # labels

print(y[y == 1].shape[0], y[y == 0].shape[0])  # number of patients with (1) and without (0) Parkinson's

