from __future__ import division
import math
import pandas as pd
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('TATASTEEL.NS (1).csv')  # import  stock data
df = df[['Open', 'Adj Close', 'Low', 'High', 'Low', 'Volume']]
df['PCT_CHNG'] = (df['Adj Close'] - df['Open']) / df['Open'] * 100.0
df['HL_PCT'] = (df['High'] - df['Adj Close']) / df['Adj Close'] * 100.0
# df['Date'] = pd.to_datetime(df['Date'])  # helping pandas to make out that the date  column represents..date
# df.sort_values(by='Date', inplace=True)


forecast_col = 'Adj Close'
df.fillna(-999999, inplace=True)
forecast_out = int(math.ceil(0.1 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)  # shifting down 10 spaces that's all
df.dropna(inplace=True)
df['PCT_CHNG_label'] = (df['Adj Close'] - df['label']) / df['Adj Close'] * 100.0

print(df.head())

X = np.array(df.drop(['label'], 1))  # I don't know why we dropped label and kept the rest
y = np.array(df['label'])  # I understand that we might need label as a parameter in this case
X = preprocessing.scale(X)
print(len(X), len(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
accuracy = clf.score(X_test, y_test)
print(accuracy)
