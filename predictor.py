import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.linear_model import LinearRegression
import joblib

default_history_window = 60
default_prediction_window = 30

class Predictor:
    def __init__(self, model_path='models/model/random_forest.pkl', time_labels=[0, 1, 10, 20, 30, 60, 1000], token_labels=[0, 10, 100, 200, 500, 10000]):
        self.model_path = model_path
        self.model = None
        self.time_labels = time_labels
        self.token_labels = token_labels
        try:
            self.load_model()
        except:
            print("No trained model found. Train the model...")
            self.train('data/AzureLLMInferenceTrace_conv.csv')

    def load_data(self, file_path):
        df = pd.read_csv(file_path, parse_dates=['TIMESTAMP', 'Deadline'])
        return df
    
    def create_feature(self, df):
        feature_vector = [0] * 9  # Initialize feature vector
        if not df.empty:
            mean_tokens = df['GeneratedTokens'].mean()
            std_tokens = df['GeneratedTokens'].std(ddof=0)
            max_tokens = df['GeneratedTokens'].max()
            min_tokens = df['GeneratedTokens'].min()
            sum_tokens = df['GeneratedTokens'].sum()
            count_tokens = len(df)
            coef_variation = std_tokens / mean_tokens if mean_tokens != 0 else 0
            
            # Linear Regression for trend calculation
            lr = LinearRegression()
            timestamps = (df['TIMESTAMP'] - df['TIMESTAMP'].min()).dt.total_seconds().values.reshape(-1, 1)
            lr.fit(timestamps, df['GeneratedTokens'].values.reshape(-1, 1))
            trend_slope = lr.coef_[0][0]

            # Calculate gradients
            time_diffs = (df['TIMESTAMP'] - df['TIMESTAMP'].shift()).dt.seconds.fillna(0)
            token_diffs = df['GeneratedTokens'].diff().fillna(0)
            gradients = (token_diffs / time_diffs.replace(0, np.inf)).fillna(0)
            mean_gradient = gradients.mean()

            # Update the feature vector with calculated values
            feature_vector = [
                mean_tokens, std_tokens, max_tokens, min_tokens,
                sum_tokens, count_tokens, coef_variation,
                mean_gradient, trend_slope
            ]
        return feature_vector

    def create_features_labels(self, df, history_window=default_history_window, prediction_window=default_prediction_window):
        features = []
        labels = []
        
        # Iterate over the DataFrame at steps of the prediction_window
        for start_time in pd.date_range(start=df['TIMESTAMP'].min().ceil(freq='s'), end=df['TIMESTAMP'].max().floor(freq='s') - timedelta(seconds=prediction_window), freq=f'{prediction_window}S'):
            end_time = start_time + timedelta(seconds=prediction_window)
            history_start_time = start_time - timedelta(seconds=history_window)
            future_end_time = end_time + timedelta(seconds=prediction_window)
            
            history_data = df[(df['TIMESTAMP'] > history_start_time) & (df['TIMESTAMP'] < start_time)]
            future_data = df[(df['TIMESTAMP'] > end_time) & (df['TIMESTAMP'] < future_end_time)]
            
            # Calculate features from history_data
            feature_vector = [0] * 9  # Initialize feature vector
            if not history_data.empty:
                mean_tokens = history_data['GeneratedTokens'].mean()
                std_tokens = history_data['GeneratedTokens'].std(ddof=0)
                max_tokens = history_data['GeneratedTokens'].max()
                min_tokens = history_data['GeneratedTokens'].min()
                sum_tokens = history_data['GeneratedTokens'].sum()
                count_tokens = len(history_data)
                coef_variation = std_tokens / mean_tokens if mean_tokens != 0 else 0
                
                # Linear Regression for trend calculation
                lr = LinearRegression()
                timestamps = (history_data['TIMESTAMP'] - history_data['TIMESTAMP'].min()).dt.total_seconds().values.reshape(-1, 1)
                lr.fit(timestamps, history_data['GeneratedTokens'].values.reshape(-1, 1))
                trend_slope = lr.coef_[0][0]

                # Calculate gradients
                time_diffs = (history_data['TIMESTAMP'] - history_data['TIMESTAMP'].shift()).dt.seconds.fillna(0)
                token_diffs = history_data['GeneratedTokens'].diff().fillna(0)
                gradients = (token_diffs / time_diffs.replace(0, np.inf)).fillna(0)
                mean_gradient = gradients.mean()

                # Update the feature vector with calculated values
                feature_vector = [
                    mean_tokens, std_tokens, max_tokens, min_tokens,
                    sum_tokens, count_tokens, coef_variation,
                    mean_gradient, trend_slope
                ]
            
            features.append(feature_vector)
            
            # Prepare the label matrix based on time and token buckets
            deadline_bins = [end_time + pd.Timedelta(seconds=t) for t in self.time_labels]
            token_bins = self.token_labels

            # Assign buckets for deadline and tokens
            future_data['DeadlineBucket'] = pd.cut(future_data['Deadline'], bins=deadline_bins, labels=self.time_labels[1:])
            future_data['TokenBucket'] = pd.cut(future_data['GeneratedTokens'], bins=token_bins, labels=self.token_labels[1:])

            # Calculate label matrix
            label_matrix = future_data.groupby(['DeadlineBucket', 'TokenBucket']).size().unstack(fill_value=0)
            labels.append(label_matrix.values)
        
        return np.array(features), np.array(labels)

    def train(self, file_path, history_window=default_history_window, prediction_window=default_prediction_window):
        df = self.load_data(file_path)
        X, y = self.create_features_labels(df, history_window, prediction_window)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [None, 10, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        model = RandomForestRegressor(random_state=42)
        max_ae_scorer = make_scorer(lambda y_true, y_pred: np.max(np.abs(y_true - y_pred)), greater_is_better=False)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring=max_ae_scorer, verbose=2, n_jobs=-1)
        grid_search.fit(X_train, y_train.reshape(y_train.shape[0], -1))
        self.model = grid_search.best_estimator_
        self.save_model()
        print(f"Training complete. Model saved to {self.model_path}.")

    def predict(self, X):
        if self.model:
            return self.model.predict(X)
        else:
            raise Exception("Model not trained yet.")

    def evaluate(self, y_pred, y_true):
        y_true = y_true.reshape(y_pred.shape[0], -1)
        metrics = {
            'MaxAE': np.max(np.abs(y_true - y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred)
        }
        return metrics

    def plot_heatmap(self, actual, predicted, filename):
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        token_labels = self.token_labels[1:]  
        time_labels = self.time_labels[1:] 
        token_ticks = np.arange(len(token_labels)) + 0.5
        time_ticks = np.arange(len(time_labels)) + 0.5
        sns.heatmap(actual.reshape(-1, 5), xticklabels=token_labels, yticklabels=time_labels, ax=ax[0], cmap="viridis", annot=True, fmt=".0f")  
        sns.heatmap(predicted.reshape(-1, 5), xticklabels=token_labels, yticklabels=time_labels, ax=ax[1], cmap="viridis", annot=True, fmt=".0f")
        ax[0].set_title("Actual Values")
        ax[0].set_xlabel("Token Buckets")
        ax[0].set_ylabel("Deadline Buckets")
        ax[0].set_xticks(token_ticks)
        ax[0].set_yticks(time_ticks)
        ax[1].set_title("Predicted Values")
        ax[1].set_xlabel("Token Buckets")
        ax[1].set_ylabel("Time Buckets")
        ax[1].set_xticks(token_ticks)
        ax[1].set_yticks(time_ticks)
        plt.suptitle('Comparison of Actual vs. Predicted Request Distributions')
        plt.savefig(filename+'.png')

    def save_model(self):
        joblib.dump(self.model, self.model_path)

    def load_model(self):
        self.model = joblib.load(self.model_path)

'''
data_file_path = 'data/AzureLLMInferenceTrace_conv.csv'
predictor = Predictor()
#predictor.train(data_file_path)
df = predictor.load_data(data_file_path)
X, y = predictor.create_features_labels(df)
predictions = predictor.predict(X)
print(predictor.evaluate(predictions, y))
#for i in range(len(y)-10, len(y)):
    #predictor.plot_heatmap(y[i], predictions[i], 'prediction'+str(i)+'.png')'''
