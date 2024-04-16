import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, LayerNormalization, MultiHeadAttention, Flatten, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

class CustomAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim):
        super(CustomAttention, self).__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.norm = LayerNormalization(axis=1)

    def call(self, inputs):
        # Using inputs as query, key, and value
        attn_output = self.attention(inputs, inputs, inputs)
        # Apply normalization
        return self.norm(attn_output + inputs)  # Residual connection

# Load and prepare data
data_path = 'data/AzureLLMInferenceTrace_conv.csv'
df = pd.read_csv(data_path)
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
df.set_index('TIMESTAMP', inplace=True)
df['RequestCount'] = 1  # Counting each request

# Resample data to second level
df_seconds = df['RequestCount'].resample('S').sum()

# Function to create features and labels
def create_features_labels(df, look_back=60, window=1):
    X, Y = [], []
    timestamps = df.index.to_series()
    for i in range(len(timestamps) - look_back):
        end_time = timestamps.iloc[i + look_back - 1]
        start_time_next_minute = end_time + pd.Timedelta(seconds=1)
        end_time_next_minute = end_time + pd.Timedelta(minutes=window)
        if end_time_next_minute > timestamps.iloc[-1]:
            break
        next_minute_mask = (timestamps >= start_time_next_minute) & (timestamps <= end_time_next_minute)
        next_minute_data = df[next_minute_mask].sum()
        X.append(df[i:i + look_back].values)
        Y.append(next_minute_data)
    return np.array(X), np.array(Y)

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
df_seconds_scaled = scaler.fit_transform(df_seconds.values.reshape(-1, 1))

# Prepare data
look_back = 300  
window = 5
dataX, dataY = create_features_labels(df_seconds, look_back, window)

# Split into train and test sets
train_size = int(len(dataX) * 0.67)
trainX, trainY = dataX[:train_size], dataY[:train_size]
testX, testY = dataX[train_size:], dataY[train_size:]

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# Define and compile the model with LSTM and attention
model = Sequential([
    LayerNormalization(axis=1),
    LSTM(50, return_sequences=True),  # LSTM layer with return sequences
    CustomAttention(num_heads=5, key_dim=50),  # Custom attention layer
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(trainX, trainY, epochs=100, batch_size=128, verbose=2)
model.save('model/combined_lstm_attention_model.keras')

# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Calculate and print RMSE
trainScore = np.sqrt(mean_squared_error(trainY, trainPredict))
testScore = np.sqrt(mean_squared_error(testY, testPredict))
print(f'Train Score: {trainScore:.2f} RMSE')
print(f'Test Score: {testScore:.2f} RMSE')

# Plotting actual vs predicted
plt.figure(figsize=(20, 6))
plt.plot(np.arange(len(trainY)), trainY, label='Actual Minute-Level Data (train stage)', color='lightsalmon')
plt.plot(np.arange(len(trainPredict)), trainPredict, label='Predicted Minute-Level Data (train stage)', color='brown')
offset = len(trainY)
plt.plot(offset+np.arange(len(testY)), testY, label='Actual Minute-Level Data (test stage)', color='pink')
plt.plot(offset+np.arange(len(testPredict)), testPredict, label='Predicted Minute-Level Data (test stage)', color='purple')

plt.title('Comparison of Actual vs. Predicted Requests Aggregated to Minute-Level')
plt.xlabel('Timestamp')
plt.ylabel('Number of Requests')
plt.legend()
plt.grid(True)
plt.savefig('prediction.png')