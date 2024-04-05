import pandas as pd
import boto3
import sagemaker
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.session import Session
from sagemaker.s3 import S3Uploader, S3Downloader
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the dataset
csv_file = 'data/AzureLLMInferenceTrace_conv.csv'
data = pd.read_csv(csv_file, parse_dates=['TIMESTAMP'])

# Pre-process the data: create training and test sets
train_data = data.iloc[:int(len(data) * 0.2)]  # Use 20% for training
test_data = data.iloc[int(len(data) * 0.2):]  # Use the remaining 80% for testing

def convert_to_jsonlines(df, filename):
    time_series = []
    for _, row in df.iterrows():
        series = {
            "start": row['TIMESTAMP'].isoformat(),
            "target": [row['ContextTokens']],
            # "dynamic_feat": [[row['GeneratedTokens']]]  # Uncomment if using dynamic features
        }
        time_series.append(series)
    
    with open(filename, 'w') as f:
        for series in time_series:
            f.write(json.dumps(series) + '\n')

# Convert and save training and test sets
convert_to_jsonlines(train_data, 'train.json')
convert_to_jsonlines(test_data, 'test.json')

input_data = {
    'train': 'data/train.json',
    'test': 'data/test.json'
}

# Configure the estimator for local mode
estimator = Estimator(
    image_uri='amazon image uri for DeepAR', # Use the appropriate DeepAR Docker image URI
    role='SageMakerRole', # Specify the role
    instance_count=1,
    instance_type='local', # Specify 'local' for local mode
    hyperparameters={
        "time_freq": 'minutes',
        "context_length": '30',
        "prediction_length": '10',
        "epochs": '20',
        "early_stopping_patience": '10',
        "learning_rate": "0.001",  
        "mini_batch_size": "64",  
        "num_cells": "40",  
        "num_layers": "3",  
        "likelihood": "gaussian",  
        "dropout_rate": "0.1", 
        "embedding_dimension": "10",  
        "num_dynamic_feat": "auto"
    } 
)

# Train in local mode
estimator.fit(input_data)

# Deploy the model
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='local',
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer()
)

# Evalute the model
test_series = convert_to_jsonlines(test_data, 'AzureLLMInference_test_series.json')
with open('AzureLLMInference_test_series.json', 'r') as f:
    test_jsonlines = f.read()
predictions = predictor.predict(test_jsonlines)
def process_predictions(predictions):
    forecasted_values = []
    for pred in predictions['predictions']:
        forecasted_value = pred['quantiles']['0.5']  
        forecasted_values.append(forecasted_value)
    return forecasted_values

# Process predictions to get forecasted values
forecasted_values = process_predictions(predictions)

# Convert forecasted and actual values to binary outcomes based on a threshold
threshold = 100  # Example threshold, adjust based on your use case
y_true_binary = [1 if x > threshold else 0 for x in test_data['ContextTokens'].values]
y_pred_binary = [1 if x > threshold else 0 for x in forecasted_values]

# Calculate accuracy, precision, and recall
accuracy = accuracy_score(y_true_binary, y_pred_binary)
precision = precision_score(y_true_binary, y_pred_binary)
recall = recall_score(y_true_binary, y_pred_binary)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")


