import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import load_model

# Load and prepare data
data_path = 'data/AzureLLMInferenceTrace_conv.csv'
df = pd.read_csv(data_path)
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
df.set_index('TIMESTAMP', inplace=True)
df['RequestCount'] = 1  # Counting each request

# Resample data to second level
df_seconds = df['RequestCount'].resample('S').sum()

# Function to create features and labels based on actual timestamps
def create_features_labels(df, look_back=60):
    X, Y = [], []
    timestamps = df.index.to_series()
    # Iterate over the timestamps
    for i in range(len(timestamps) - look_back):
        end_time = timestamps.iloc[i + look_back - 1]
        start_time_next_minute = end_time + pd.Timedelta(seconds=1)
        end_time_next_minute = end_time + pd.Timedelta(minutes=1)

        # Ensure we do not go out of bounds
        if end_time_next_minute > timestamps.iloc[-1]:
            break

        # Get the indices for the next minute
        next_minute_mask = (timestamps >= start_time_next_minute) & (timestamps <= end_time_next_minute)
        next_minute_data = df[next_minute_mask].sum()

        # Append data
        X.append(df[i:i + look_back].values)
        Y.append(next_minute_data)

    return np.array(X), np.array(Y)

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
df_seconds_scaled = scaler.fit_transform(df_seconds.values.reshape(-1, 1))

# Prepare data
look_back = 60  
dataX, dataY = create_features_labels(df_seconds, look_back)

# Split into train and test sets (simple split for demonstration)
train_size = int(len(dataX) * 0.67)
trainX, trainY = dataX[:train_size], dataY[:train_size]
testX, testY = dataX[train_size:], dataY[train_size:]

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# Define and compile LSTM model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(look_back, 1)))
model.add(Dropout(0.3))
model.add(LSTM(100, return_sequences=False))  
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
# Load pre-trained model or train a new one
#model = load_model('model/lstm_model.keras')
# Train and save
model.fit(trainX, trainY, epochs=600, batch_size=128, verbose=2)
model.save('model/lstm_model.keras')

# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Calculate and print RMSE
trainScore = np.sqrt(mean_squared_error(trainY, trainPredict))
testScore = np.sqrt(mean_squared_error(testY, testPredict))
print('Train Score: %.2f RMSE' % (trainScore))
print('Test Score: %.2f RMSE' % (testScore))

# Plotting actual vs predicted
plt.figure(figsize=(20, 6))
markerfrequency = 50
plt.plot(np.arange(len(trainY)), trainY, label='Actual Minute-Level Data (train stage)', color='lightsalmon') # marker='o', markevery=markerfrequency)
plt.plot(np.arange(len(trainPredict)), trainPredict, label='Predicted Minute-Level Data (train stage)', color='brown') # marker='x', markevery=markerfrequency)
offset = len(trainY)
plt.plot(offset+np.arange(len(testY)), testY, label='Actual Minute-Level Data (test stage)', color='pink') #marker='o', markevery=markerfrequency)
plt.plot(offset+np.arange(len(testPredict)), testPredict, label='Predicted Minute-Level Data (test stage)', color='purple') # marker='x', markevery=markerfrequency)

plt.title('Comparison of Actual vs. Predicted Requests Aggregated to Minute-Level')
plt.xlabel('Timestamp')
plt.ylabel('Number of Requests')
plt.legend()
plt.grid(True)
plt.savefig('prediction.png')
