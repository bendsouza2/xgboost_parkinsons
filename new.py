import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('parkinsons.data')

# print(df.dtypes)

x = df.loc[:, df.columns != "status"].to_numpy()[:, 1:]  # features
y = df.loc[:, 'status'].to_numpy()  # labels

print(y[y == 1].shape[0], y[y == 0].shape[0])  # number of patients with (1) and without (0) Parkinson's

# Normalising the data
scaler = MinMaxScaler((-1, 1))
features = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=7)  #splitting

#  Building the model
xgb = XGBClassifier()
xgb.fit(x_train, y_train)

#  Accuracy
pred_y = xgb.predict(x_test)
print(accuracy_score(y_test, pred_y) * 100)

