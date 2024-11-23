import os
import pandas as pd
import sys

def fuse_csvs(input_dir, output_file):
    # List to hold data from all CSV files
    all_data = []

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        input_file = os.path.join(input_dir, filename)
        if os.path.isfile(input_file) and filename.endswith(".csv"):
            print(f"Processing file: {input_file}")
            try:
                # Read the current CSV into a DataFrame
                data = pd.read_csv(input_file)
                all_data.append(data)
            except Exception as e:
                print(f"An error occurred while reading {input_file}: {e}")

    # Concatenate all DataFrames into one
    if all_data:
        fused_data = pd.concat(all_data, ignore_index=True)
        # Save the fused data to the output file
        fused_data.to_csv(output_file, index=False)
        print(f"All CSVs have been fused into {output_file}")
    else:
        print("No CSV files found in the directory.")

if __name__ == "__main__":
    # Check for correct usage
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_directory> <output_file>")
        sys.exit(1)

    # Get parameters from command line
    input_dir = sys.argv[1]
    output_file = sys.argv[2]

    # Check if the input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    # Fuse the CSVs
    fuse_csvs(input_dir, output_file)
