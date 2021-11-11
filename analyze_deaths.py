"""
Predicting the number of COVID-19 deaths given time series data.

Dataset being used: https://www.kaggle.com/fireballbyedimyrnmom/us-counties-covid-19-dataset
"""
from datetime import timedelta

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
import tensorflow as tf

# define the datatypes for correct csv parsing
dtypes = {'date': 'str',
          'county': 'category',
          'state': 'category',
          'fips': 'category',
          'cases': 'int32',
          'deaths': 'int32'}
parse_dates = ['date']

# read the csv containing covid data
covid_data = pd.read_csv('us-counties.csv', parse_dates=parse_dates)
covid_data = covid_data.dropna()
covid_data = covid_data.astype(dtypes)
covid_data['date'] = pd.to_datetime(covid_data['date'])

# get only deaths in pierce county, WA
covid_data = covid_data[(covid_data['county'] == 'Pierce') & (covid_data['state'] == 'Washington')]
covid_data.sort_values('date', inplace=True, ascending=True)
covid_data = covid_data.reset_index(drop=True)
covid_data = covid_data.drop(['county', 'cases', 'state', 'fips'], axis=1)

# undo accumulation
covid_data['deaths'] -= covid_data['deaths'].shift(1).fillna(0)
covid_data['deaths'] = covid_data['deaths'].astype(int)
covid_data = covid_data.set_index('date')
covid_data = covid_data.clip(0, covid_data['deaths'].max())


# split into training and testing data
test_cutoff = covid_data.index.max() - timedelta(days=30)
train, test = covid_data[covid_data.index < test_cutoff], \
              covid_data[covid_data.index >= test_cutoff]

# scale the data for faster processing
sc = MinMaxScaler(feature_range=(0, 1))
covid_scaled = sc.fit_transform(covid_data)
train_scaled = sc.transform(train)
test_scaled = sc.transform(test)
max_val = covid_data['deaths'].max()

# set the look-back to 10, create X_train and y_train
n_lag = 28
X_train = []
y_train = []
for i in range(n_lag, len(train)):
    X_train.append(train_scaled[i - n_lag:i, 0])
    y_train.append(train_scaled[i, 0])

# reshape and numpy
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# repeat the process with the tests
X_test = []
y_test = []
for i in range(n_lag, len(test)):
    X_test.append(test_scaled[i - n_lag:i, 0])
    y_test.append(test_scaled[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

X_val = []
y_val = []
time = []
for i in range(n_lag, len(covid_data)):
    X_val.append(covid_scaled[i - n_lag:i, 0])
    time.append(covid_data.index[i])
    y_val.append(covid_scaled[i, 0])

X_val, y_val = np.array(X_val), np.array(y_val)
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

# define a LSTM (predicting deaths)
model = Sequential()
model.add(LSTM(units=n_lag, activation='relu', input_shape=(n_lag, 1), return_sequences=True, kernel_regularizer=l2(0.1), recurrent_regularizer=l2(0.1), bias_regularizer=l2(0.1)))
model.add(Dropout(0.2))
model.add(LSTM(units=30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

# compile on the data (COVID-19 Deaths in Pierce County)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(loss='mse', optimizer=opt, metrics=['mse'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=750, callbacks=[callback])
scaled_predict = model.predict(X_val)
predict = sc.inverse_transform(scaled_predict)
plt.plot(time, predict, label='predicted')
plt.plot(covid_data.index, covid_data['deaths'], label='actual')
plt.title('COVID Deaths in Pierce County')
plt.xlabel('Time')
plt.ylabel('Deaths')
plt.legend()
plt.show()
