import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, make_scorer
from sklearn.linear_model import LogisticRegression
import joblib

default_history_window = 60
default_prediction_window = 30

class Predictor:
    def __init__(self, model_path='/data/jiawei_li/InferScheduler/models/model/random_forest.pkl', token_model_path='/data/jiawei_li/InferScheduler/models/model/token_logit.pkl', time_labels=[0, 1, 10, 20, 30, 60, 1000], token_labels=[0, 10, 100, 200, 500, 10000]):
        self.model_path = model_path
        self.token_model_path = token_model_path
        self.model = None
        self.token_model = None
        self.time_labels = time_labels
        self.token_labels = token_labels
        try:
            self.load_model()
        except:
            print("No trained model found. Train the model...")
            self.train('/data/jiawei_li/InferScheduler/data/AzureLLMInferenceTrace_conv.csv')
        try:
            self.load_token_model()
        except:
            print("No trained token model found. Train the model...")
            self.train_token_model('/data/jiawei_li/InferScheduler/data/AzureLLMInferenceTrace_conv.csv')

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
            feature_vector = self.create_feature(history_data)
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
        model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
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
            'Accuracy': accuracy_score(y_true, y_pred),
            'Classification Report': classification_report(y_true, y_pred)
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

    def save_token_model(self):
        joblib.dump(self.token_model, self.token_model_path)

    def load_model(self):
        self.model = joblib.load(self.model_path)

    def load_token_model(self):
        self.token_model = joblib.load(self.token_model_path)

    def predict_generated_token(self, context_token_count):
        if self.token_model is None:
            raise Exception("Token model not trained yet.")
        return int(self.token_model.predict([[context_token_count]])[0])

    def train_token_model(self, file_path):
        df = self.load_data(file_path)
        X = df['ContextTokens'].values.reshape(-1, 1)
        y = df['GeneratedTokens'].values
        y = pd.cut(y, bins=self.token_labels, labels=self.token_labels[1:])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        token_model = LogisticRegression(max_iter=200)
        token_model.fit(X_train, y_train)
        self.token_model = token_model
        y_pred = token_model.predict(X_test)
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Classification Report': classification_report(y_test, y_pred)
        }
        print(f"Token model trained. Metrics: {metrics}")
        self.save_token_model()

predictor = Predictor()
predictor.predict_generated_token(330)
