import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DeepARModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.5):
        super(DeepARModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        batch_size = x.size(0)
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  
        out = self.bn(out[:, -1, :])

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out

class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length):
        """
        Args:
            data: A Pandas DataFrame containing the time series data.
            sequence_length: The length of the input sequences (number of time steps).
        """
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        # Subtract sequence_length to avoid overflow
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        """
        Args:
            idx: The index of the item.

        Returns:
            A tuple of (sequence, target), where sequence is a sequence of
            historical data and target is the value at the next time step.
        """
        sequence = self.data.iloc[idx:idx+self.sequence_length][['ContextTokens', 'GeneratedTokens']].values
        target = self.data.iloc[idx+self.sequence_length][['ContextTokens', 'GeneratedTokens']].values
        sequence = sequence.astype(np.float32)  
        target = target.astype(np.float32)
        return torch.tensor(sequence, dtype=torch.float), torch.tensor(target, dtype=torch.float)

# Load the dataset
csv_file_path = 'data/AzureLLMInferenceTrace_conv.csv'
data = pd.read_csv(csv_file_path, parse_dates=['TIMESTAMP'])
data['TIMESTAMP'] = data['TIMESTAMP'].dt.floor('s') 
aggregated_data = data.groupby(['TIMESTAMP'], as_index=False)[['ContextTokens', 'GeneratedTokens']].sum()
sequence_length = 60
split_idx = int(len(aggregated_data) * 0.8)
train_data = aggregated_data[:split_idx]
test_data = aggregated_data[split_idx:]
train_dataset = TimeSeriesDataset(train_data, sequence_length)
test_dataset = TimeSeriesDataset(test_data, sequence_length)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepARModel(input_size=2, hidden_size=50, num_layers=2, output_size=2, dropout_rate=0.3).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.025)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'model/predictor.pth')
model = DeepARModel(input_size=2, hidden_size=50, num_layers=2, output_size=2).to(device)
model.load_state_dict(torch.load('model/predictor.pth'))
model.eval()
predictions = []
actual_values = []
with torch.no_grad():
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        
        # Model predictions
        outputs = model(data)
        predictions.extend(outputs.view(-1).cpu().numpy())
        actual_values.extend(labels.cpu().numpy())

# For visualization, let's consider a subset of the test set for clarity
num_points_to_plot = 1000  
predictions = np.array(predictions).reshape(-1, 2)  
print(predictions)
actual_values = np.array(actual_values).reshape(-1, 2)  
subset_size = min(num_points_to_plot, len(predictions))
predictions_subset = predictions[:subset_size]
actual_values_subset = actual_values[:subset_size]
plt.figure(figsize=(10, 6))
plt.plot(actual_values_subset[:, 0], 'b-', label='Actual ContextTokens')
plt.plot(predictions_subset[:, 0], 'r--', label='Predicted ContextTokens')
plt.plot(actual_values_subset[:, 1], 'g-', label='Actual GeneratedTokens')
plt.plot(predictions_subset[:, 1], 'y--', label='Predicted GeneratedTokens')
plt.xlabel('seconds')
plt.ylabel('Token Length')
plt.legend()
plt.grid()
plt.title('Actual vs. Predicted Values')
plt.savefig('prediction.png')

'''
# Predict number of requests arrival
data['TIMESTAMP'] = data['TIMESTAMP'].dt.floor('')  # Round down to the nearest day
aggregated_data = data.groupby('TIMESTAMP').sum().reset_index()
sequence_length = 5
split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
test_data = data[split_idx:]
train_dataset = TimeSeriesDataset(train_data, sequence_length)
test_dataset = TimeSeriesDataset(test_data, sequence_length)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepARModel(input_size=2, hidden_size=50, num_layers=2, output_size=2).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'model/predictor.pth')
model = DeepARModel(input_size=2, hidden_size=50, num_layers=2, output_size=2).to(device)
model.load_state_dict(torch.load('model/predictor_day.pth'))
model.eval()
predictions = []
actual_values = []
with torch.no_grad():
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        
        # Model predictions
        outputs = model(data)
        predictions.extend(outputs.view(-1).cpu().numpy())
        actual_values.extend(labels.cpu().numpy())

# For visualization, let's consider a subset of the test set for clarity
num_points_to_plot = 14*24  
predictions = np.array(predictions).reshape(-1, 2)  
actual_values = np.array(actual_values).reshape(-1, 2)  
subset_size = min(num_points_to_plot, len(predictions))
predictions_subset = predictions[:subset_size]
actual_values_subset = actual_values[:subset_size]
plt.figure(figsize=(10, 6))
plt.plot(actual_values_subset[:, 0], 'b-', label='Actual ContextTokens')
plt.plot(predictions_subset[:, 0], 'r--', label='Predicted ContextTokens')
plt.plot(actual_values_subset[:, 1], 'g-', label='Actual GeneratedTokens')
plt.plot(predictions_subset[:, 1], 'y--', label='Predicted GeneratedTokens')
plt.xlabel('Hour')
plt.ylabel('Token Length')
plt.legend()
plt.grid()
plt.title('Actual vs. Predicted Values')
plt.savefig('prediction.png')'''

