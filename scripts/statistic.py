import pandas as pd

def summarize_trace_data(filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Convert timestamp columns to datetime objects
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    df['Deadline'] = pd.to_datetime(df['Deadline'])

    # Calculate the latency in seconds as the difference between deadline and timestamp
    df['Latency'] = (df['Deadline'] - df['TIMESTAMP']).dt.total_seconds()

    # Calculate and print summary statistics for ContextTokens, GeneratedTokens, and Latency
    summary = {
        'ContextTokens': df['ContextTokens'].describe(),
        'GeneratedTokens': df['GeneratedTokens'].describe(),
        'Latency (s)': df['Latency'].describe()
    }

    # Print the summary statistics
    for key, stats in summary.items():
        print(f"\nStatistics for {key}:")
        print(stats)

# Example usage
if __name__ == "__main__":
    # Replace with the path to your trace data CSV file
    trace_data_file = "../data/AzureLLMInferenceTrace_conv.csv"
    summarize_trace_data(trace_data_file)
