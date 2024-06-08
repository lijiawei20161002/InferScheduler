import pandas as pd

# Load data from CSV file
file_path = 'AzureLLMInferenceTrace_conv.csv'
df = pd.read_csv(file_path, parse_dates=['TIMESTAMP', 'Deadline'])

# Calculate statistics
context_tokens_stats = df['ContextTokens'].describe()
generated_tokens_stats = df['GeneratedTokens'].describe()

# Display statistics
print("ContextTokens Statistics:")
print(context_tokens_stats)
print("\nGeneratedTokens Statistics:")
print(generated_tokens_stats)

# Additional statistics if needed
print("\nAdditional Statistics:")
print("ContextTokens Variance:", df['ContextTokens'].var())
print("GeneratedTokens Variance:", df['GeneratedTokens'].var())
print("ContextTokens Median:", df['ContextTokens'].median())
print("GeneratedTokens Median:", df['GeneratedTokens'].median())

# If you need to save these statistics to a CSV file
statistics_df = pd.DataFrame({
    "ContextTokens": context_tokens_stats,
    "GeneratedTokens": generated_tokens_stats
})
print(statistics_df)
statistics_df.to_csv('token_statistics.csv')