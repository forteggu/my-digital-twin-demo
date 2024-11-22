import pandas as pd
import sys

def add_column_to_csv(input_file, output_file, column_name, column_value):
    """
    Adds a new column to the CSV with the specified value.
    
    :param input_file: Path to the input CSV file
    :param output_file: Path to the output CSV file
    :param column_name: Name of the new column
    :param column_value: Value to populate in the new column
    """
    try:
        # Load the CSV
        df = pd.read_csv(input_file)
        
        # Add the new column with the specified value
        df[column_name] = column_value
        
        # Save the modified DataFrame back to CSV
        df.to_csv(output_file, index=False)
        
        print(f"New column '{column_name}' added with value '{column_value}'. Saved to {output_file}.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Command-line arguments: script.py input.csv output.csv column_name column_value
    if len(sys.argv) != 5:
        print("Usage: script.py <input_file> <output_file> <column_name> <column_value>")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        column_name = sys.argv[3]
        column_value = sys.argv[4]
        
        add_column_to_csv(input_file, output_file, column_name, column_value)
