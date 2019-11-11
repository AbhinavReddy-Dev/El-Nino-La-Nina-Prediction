
# coding: utf-8

### Import all necessary libraries

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import keras as kr
import sklearn
import math
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import itertools
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('El-Nino.csv', sep = '\t')

data.head()

#### Replace all column names by overwritting on it

cols = ['Year','Janauary','February','March','April','May','June','July','August','September','October','November','December']

data.columns = cols

data.head(10)

#### Set Index as Year

data.set_index('Year', inplace = True)
data.head()

#### Do transpose to know, how many years are present

data1 = data.transpose()
data1

#### Generate the date_range series 

dates = pd.date_range(start = '1950-01', freq = 'MS', periods = len(data1.columns)*12)
dates

#### Convert the dataframe into matrix 

data_np = data1.transpose().as_matrix()

shape = data_np.shape
shape

data_np


#### Let's convert the matrix size of 68 x 12 into column vector 

data_np = data_np.reshape((shape[0] * shape[1], 1))

data_np.shape

#### Convert the data_np into dataframe
# * Here we are merging two series data i.e data_np and dates series into dataframe.
# * As this dataset belongs to timeseries concept, we apply dates series as index to our dataframe.

df = pd.DataFrame({'Mean' : data_np[:,0]})
df.set_index(dates, inplace = True)

df.head()

#### Now Let's plot how our data looks like

# plt.figure(figsize = (8,3))
# plt.plot(df.index, df['Mean'])
# plt.title('Yearly vs Monthly Mean graph')
# plt.xlabel('Year')
# plt.ylabel('Mean across Month')
# plt.show(block = True)

dataset = df.values

dataset.shape

#### Here we are splitting the data into train and test set

train = dataset[0:696,:]
test = dataset[696:,:]

print("Actual dataset: ",dataset.shape)
print("Training dataset: ",train.shape)
print("Testing dataset: ",test.shape)

# Converting the data into MinMax Scaler because to avoid any outliers present in our dataset
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data.shape

#### As we know we use LSTM model to our data then we follow Imporvements over RNN principle
# * To see more inbrief Click [here](https://www.analyticsvidhya.com/blog/2017/12/fundamentals-of-deep-learning-introduction-to-lstm/)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)


# In[24]:


#x_train shape
x_train.shape

#y_train shape
y_train.shape

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

x_train.shape

# Creating and fitting the model

model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1],1)))
model.add(LSTM(units = 50))
model.add(Dense(1))

model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.fit(x_train, y_train, epochs=10, batch_size = 1, verbose = 2)

# Now Let's perform same operations that are done on training dataset
inputs = df[len(df) - len(test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
Mean = model.predict(X_test)
Mean1 = scaler.inverse_transform(Mean)

rms=np.sqrt(np.mean(np.power((test-Mean1),2)))
rms

#plotting the train, test and forecast data
train = df[:696]
test = df[696:]
test['Predictions'] = Mean1

# plt.figure(figsize=(8,3))
# plt.plot(train['Mean'])
# plt.plot(test['Mean'], color = 'black')
# plt.plot(test['Predictions'], color = 'orange')
# plt.xlabel('Years')
# plt.ylabel('Mean')
# plt.title('Fitting Actual graph and Predicted graph')
# plt.show(block = True)

#Here we are taking steps as 2, means we have taken test size as 120 that is step-1 and steps=2 
#means taking 120 test values and 120 future values i.e next 10 year values from test data
trainpred = model.predict(X_test,steps=2)

trainpred.shape

#trainpred
pred = scaler.inverse_transform(trainpred)

# Total predicted values are 240, but now I'm printing only first 24 values
pred[0:24] 

test.head()

test.tail(12)

# Now printing the test Accuracy
testScore = math.sqrt(mean_squared_error(test['Mean'], trainpred[:120,0]))*100
print('Accuracy Score: %.2f' % (testScore))

#Taking input from user to predict according to the year
step_yr = 2017
yr = int(input("Enter the Year that you'd like to Predict: "))
c = yr - step_yr
e = c-1
b = pred[120+(e*12) : 120+(e*12)+12].mean(axis=0)

print(f"Mean of the total months in the year {yr} is {b}")

if b >= 0.5 and b <= 0.9:
    print(yr, ' is predicted to be Weak El-Nino')

elif b >= 1.0 and b <= 1.4:
    print(yr,' is predicted to be Moderate El-Nino')

elif b >= 1.5 and b <= 1.9:
    print(yr, ' is predicted to be Strong El-Nino')

elif b >= 2:
    print(yr, ' is predicted to be Very Strong El-Nino')

elif b <=-0.5 and b >= -0.9:
    print(yr, ' is predicted to be Weak La-Nina')

elif b <= -1 and b >= -1.4:
    print(yr, ' is predicted to be Moderate La-Nina')

elif b <= -1.5:
    print(yr, ' is predicted to be Strong La-Nina')

else:
    print(yr, ' is predicted to be Moderate')

# Now plot the graph of future predicted values for that generate a date range series upto 2027
dates1 = pd.date_range(start = '2008-01', freq = 'MS', end = '2027-12')
dates1


new_df = pd.DataFrame({'Predicted_values':pred[:,0]})

new_df.set_index(dates1, inplace = True)

new_df.head()

new_df.tail()

#Plotting the Final Graph
plt.figure(figsize=(6,4))
plt.plot(train['Mean'])
plt.plot(test['Mean'], color = 'black')
plt.plot(test['Predictions'], color = 'orange')
plt.plot(new_df['Predicted_values'][120:], color = 'red')
plt.xlabel('Years')
plt.ylabel('Mean')
plt.legend(loc = True)
plt.title('Actual vs Testing vs Predicted')
plt.show(block = True)



#2018  is predicted to be Moderate
#2019  is predicted to be Weak El-Nino
#2020  is predicted to be Moderate La-Nina
#2021  is predicted to be Weak La-Nina
#2022  is predicted to be Moderate
#2023  is predicted to be Moderate
#2024  is predicted to be Weak El-Nino
#2025  is predicted to be Strong El-Nino
#2026  is predicted to be Moderate
#2027  is predicted to be Moderate
#2028  is predicted to be Moderate








