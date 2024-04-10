import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DeepARModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(DeepARModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
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

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
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
        # Fetch the sequence from idx to idx+sequence_length
        sequence = self.data.iloc[idx:idx+self.sequence_length]['ContextTokens'].values
        # Target is the next value following the sequence
        target = self.data.iloc[idx+self.sequence_length]['GeneratedTokens']
        sequence = torch.tensor(sequence, dtype=torch.float).unsqueeze(-1)  
        target = torch.tensor(target, dtype=torch.float)
        return torch.tensor(sequence, dtype=torch.float), torch.tensor(target, dtype=torch.float)

# Load the dataset
csv_file_path = 'data/AzureLLMInferenceTrace_conv.csv'
data = pd.read_csv(csv_file_path, parse_dates=['TIMESTAMP'])
sequence_length = 5
split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
test_data = data[split_idx:]
train_dataset = TimeSeriesDataset(train_data, sequence_length)
test_dataset = TimeSeriesDataset(test_data, sequence_length)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepARModel(input_size=1, hidden_size=50, num_layers=2, output_size=1).to(device)
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
num_points_to_plot = 100  
plt.figure(figsize=(10, 6))
plt.plot(actual_values[:num_points_to_plot], 'b-', label='Actual')
plt.plot(predictions[:num_points_to_plot], 'r--', label='Predicted')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.title('Actual vs. Predicted Values')
plt.savefig('prediction.png')