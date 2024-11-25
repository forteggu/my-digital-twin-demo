import pandas as pd
import random

# Paths to the input CSV files
csv_file_1 = "../datasets/labelless/normal_queries_with_prefix.csv"
csv_file_2 = "../datasets/labelless/final_injection_queries_with_prefix.csv"
# Load both CSV files into dataframes
df1 = pd.read_csv(csv_file_1)
df2 = pd.read_csv(csv_file_2)

# Concatenate the dataframes
combined_df = pd.concat([df1, df2], ignore_index=True)

# Shuffle the rows randomly
shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)

# Save the shuffled dataframe to a new CSV
output_file = '../datasets/sqli_full_testing_dataset.csv'
shuffled_df.to_csv(output_file, index=False)

print(f"Fused and shuffled CSV saved to {output_file}")
