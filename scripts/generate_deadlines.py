import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Load the CSV file
file_path = '../data/AzureLLMInferenceTrace_conv.csv'
data = pd.read_csv(file_path)

# Define the function to convert timestamps and calculate deadlines with randomness
def calculate_deadline(row):
    import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Load the CSV file
file_path = '../data/AzureLLMInferenceTrace_conv.csv'
data = pd.read_csv(file_path)

def parse_timestamp(timestamp): # Truncate fractional seconds to six digits if longer
    main_part, fractional_seconds = timestamp.strip().split('.')
    fractional_seconds = fractional_seconds[:6]
    truncated_timestamp = f"{main_part}.{fractional_seconds}"
    format_str = '%Y-%m-%d %H:%M:%S.%f'
    try:
        return datetime.strptime(truncated_timestamp, format_str)
    except ValueError:
        print(f"Failed to parse timestamp even after truncation: '{truncated_timestamp}'")
        return None

def calculate_deadline(row):
    timestamp = parse_timestamp(row['TIMESTAMP'])
    if timestamp is None:
        return None  # Skip rows where timestamp could not be parsed

    inference_time = row['GeneratedTokens'] * 100  # 100 ms inference time per token
    if np.random.rand() > 0.5:  
        additional_time = np.random.randint(3000, 10001)  
    else:
        additional_time = 1000 + np.random.exponential(scale=10000)  
    total_time = inference_time + additional_time
    deadline = timestamp + timedelta(milliseconds=total_time)
    return deadline

# Apply the function to calculate deadlines
data['Deadline'] = data.apply(calculate_deadline, axis=1)

# Overwrite the original CSV file with the updated data
data.to_csv(file_path, index=False)

print("Original CSV file has been updated with new deadlines with added randomness.")
