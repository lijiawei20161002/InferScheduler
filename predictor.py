import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression

time_labels = ['0s', '10s', '20s', '30s', '60s', '180s', '1000s']
token_labels = [0, 100, 200, 300, 500, 10000]

def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['TIMESTAMP', 'Deadline'])
    return df

def create_features_labels(df, history_window=10, prediction_window=60):
    features = []
    labels = []
    
    # Iterate over the DataFrame at steps of the prediction_window
    for start_time in pd.date_range(start=df['TIMESTAMP'].min().ceil(freq='s'), end=df['TIMESTAMP'].max().floor(freq='s') - timedelta(seconds=prediction_window), freq=f'{prediction_window}S'):
        end_time = start_time + timedelta(seconds=prediction_window)
        history_start_time = start_time - timedelta(seconds=history_window)
        future_end_time = end_time + timedelta(seconds=prediction_window)

        history_data = df[(df['TIMESTAMP']>history_start_time) & (df['TIMESTAMP']<start_time)]
        future_data = df[(df['TIMESTAMP']>end_time) &(df['TIMESTAMP']<future_end_time)]

        # Calculate features from history_data
        feature_vector = [0] * 9
        if not history_data.empty:
            # Basic statistical calculations
            mean_tokens = history_data['GeneratedTokens'].mean()
            std_tokens = history_data['GeneratedTokens'].std(ddof=0)
            max_tokens = history_data['GeneratedTokens'].max()
            min_tokens = history_data['GeneratedTokens'].min()
            sum_tokens = history_data['GeneratedTokens'].sum()
            count_tokens = len(history_data)
            coef_variation = std_tokens / mean_tokens if mean_tokens != 0 else 0

            # Update feature vector with calculated values
            vector = [
                mean_tokens,  # Mean of generated tokens
                std_tokens,   # Standard deviation of generated tokens
                max_tokens,   # Maximum of generated tokens
                min_tokens,   # Minimum of generated tokens
                sum_tokens,   # Sum of generated tokens
                count_tokens, # Count of data points
                coef_variation  # Coefficient of variation
            ]
            feature_vector[:7]=vector

            # Calculate gradient (first derivative)
            time_diffs = (history_data['TIMESTAMP'] - history_data['TIMESTAMP'].shift()).dt.seconds.fillna(0)
            token_diffs = history_data['GeneratedTokens'].diff().fillna(0)
            gradients = (token_diffs / time_diffs.replace(0, np.inf)).fillna(0)
            mean_gradient = gradients.mean()
            feature_vector[7]= mean_gradient

            # Calculate linear trend (slope)
            lr = LinearRegression()
            timestamps = (history_data['TIMESTAMP'] - history_data['TIMESTAMP'].min()).dt.total_seconds().values.reshape(-1, 1)
            lr.fit(timestamps, history_data['GeneratedTokens'].values.reshape(-1, 1))
            trend_slope = lr.coef_[0][0]
            feature_vector[8]=trend_slope  

        features.append(feature_vector)    
        deadline_bins = end_time + pd.to_timedelta(time_labels)  
        token_bins = token_labels

        future_data['DeadlineBucket'] = pd.cut(df['Deadline'], bins=deadline_bins, labels=time_labels[1:])
        future_data['TokenBucket'] = pd.cut(df['GeneratedTokens'], bins=token_bins, labels=token_labels[1:])

        # Calculate label from future_data
        label = [
            future_data.groupby(['DeadlineBucket', 'TokenBucket']).size().unstack(fill_value=0)
        ]
        labels.append(label)

    return np.array(features), np.array(labels)

def plot_heatmap(actual, predicted, filename):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(actual.reshape(-1, 5), xticklabels=token_labels, yticklabels=time_labels, ax=ax[0], cmap="viridis", annot=True, fmt=".0f")  
    sns.heatmap(predicted.reshape(-1, 5), xticklabels=token_labels, yticklabels=time_labels, ax=ax[1], cmap="viridis", annot=True, fmt=".0f")
    ax[0].set_title("Actual Values")
    ax[0].set_xlabel("Token Buckets")
    ax[0].set_ylabel("Deadline Buckets")
    ax[1].set_title("Predicted Values")
    ax[1].set_xlabel("Token Buckets")
    ax[1].set_ylabel("Time Buckets")
    plt.suptitle('Comparison of Actual vs. Predicted Request Distributions')
    plt.savefig(filename+'.png')

df = load_data('data/AzureLLMInferenceTrace_conv.csv')
X, y = create_features_labels(df, history_window=60, prediction_window=60)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

def max_absolute_error(y_true, y_pred):
    return np.max(np.abs(y_true - y_pred))

max_ae_scorer = make_scorer(max_absolute_error, greater_is_better=False)
# Setup the grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1, scoring=max_ae_scorer)
grid_search.fit(X_train, y_train.reshape(y_train.shape[0], -1))
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best MaxAE: {-grid_search.best_score_}")

# Use the best estimator to make predictions
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

if y_test.ndim > 2:
    y_test = y_test.reshape(y_test.shape[0], -1)  # Flatten each sample into a 1D array

if predictions.ndim > 2:
    predictions = predictions.reshape(predictions.shape[0], -1)  # Flatten each sample into a 1D array

max_ae = max_absolute_error(y_test, predictions)    
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"Maximum Absolute Error (MaxAE): {max_ae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (Coefficient of Determination): {r2:.2f}")

for i in range(len(y_test)):
    actual = y_test[i]
    predicted = predictions[i]
    filename = str(i)+'.png'
    plot_heatmap(actual, predicted, filename)
