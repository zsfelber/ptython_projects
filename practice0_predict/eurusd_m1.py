#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

#for normalizing data
scaler = MinMaxScaler(feature_range=(0, 1))

#read the file
df0 = pd.read_csv('f:/EURUSD_M5_200001030000_202107092350.csv', sep='\t')
df = df0.head(10000)

# looking at the first five rows of the data
print(df.head())

print('\n Shape of the data:')
print(df.shape)

df['Time0'] = df[['<DATE>','<TIME>']].apply(lambda x: ' '.join(x), axis=1)
print('\n ->')
print(df.head())
df['Time'] = pd.to_datetime(df['Time0'],format='%Y%m%d %H:%M:%S')
df.index = df.Time
df.drop(['<DATE>','<TIME>','<OPEN>','<HIGH>','<LOW>','<TICKVOL>','<VOL>','<SPREAD>','Time0','Time'], axis=1, inplace=True)

print('\n ->')
print(df.head())

#creating train and test sets
dataset = df.values

train = dataset[0:9900,:]
valid = dataset[9900:,:]




#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

print('\n Scaled data (0,1):')
print('\n len:' + str(len(scaled_data)))
print(scaled_data.shape)


for i in range(scaled_data.shape[0]-10,scaled_data.shape[0]):
    print(scaled_data[i][0])

train_len = 10
x_train, y_train = [], []
for i in range(train_len,len(train)):
    x_train.append(scaled_data[i-train_len:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

print('\n x_train shape:')
print(x_train.shape)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

#predicting 9900..10000 values, using past <train_len> bars for each one from the train data
inputs = new_data[len(new_data) - len(valid) - train_len:].values
#inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(train_len,inputs.shape[0]):
    X_test.append(inputs[i-train_len:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)



rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
print('\n rms:')
print(rms)




#for plotting
train = df[9000:9900]
valid = df[9900:]
valid['Predictions'] = closing_price
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])



