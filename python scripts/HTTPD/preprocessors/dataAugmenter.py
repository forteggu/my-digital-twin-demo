import pandas as pd
import numpy as np
import argparse

def augment_anomalous_data(data, column_name, column_value, n_copies=5):
    """
    Augment data by adding noise to numeric features and randomizing the IP field.
    Only rows where data[column_name] == column_value are modified.
    If column_value is 'ALL', all rows are augmented.
    
    Args:
        data (pd.DataFrame): Input dataset.
        column_name (str): Name of the column to filter.
        column_value (str): Value of the column to filter.
        n_copies (int): Number of augmented copies to generate for each anomalous row.

    Returns:
        pd.DataFrame: Augmented dataset.
    """
    # Filter rows with specified column value or use all rows if column_value is 'ALL'
    if column_value == 'ALL':
        anomalous_data = data.copy()
    else:
        anomalous_data = data[data[column_name] == column_value].copy()
    
    augmented_data = []
    
    for _ in range(n_copies):
        # Copy the data
        augmented = anomalous_data.copy()
        
        # Add noise to numeric columns
        if 'time_delta' in augmented.columns:
            augmented['time_delta'] = augmented['time_delta'] * (1 + np.random.uniform(-0.1, 0.1, len(augmented)))
        
        # Randomize the IP field
        augmented['ip'] = [
            f"{np.random.randint(1, 255)}.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}"
            for _ in range(len(augmented))
        ]
        
        # Keep the original endpoint unchanged
        augmented_data.append(augmented)
    
    # Concatenate augmented data
    augmented_data = pd.concat(augmented_data, ignore_index=True)
    
    # Combine with original data
    combined_data = pd.concat([data, augmented_data], ignore_index=True)
    
    return combined_data

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Augment anomalous data in a CSV file.")
    parser.add_argument('input_file', type=str, help="Path to the input CSV file.")
    parser.add_argument('output_file', type=str, help="Path to save the augmented CSV file.")
    parser.add_argument('column_name', type=str, help="Name of the column to filter.")
    parser.add_argument('column_value', type=str, help="Value of the column to filter. Use 'ALL' to augment all rows.")
    parser.add_argument('--n_copies', type=int, default=5, help="Number of augmented copies to generate for each anomalous row (default: 5).")
    
    args = parser.parse_args()
    
    # Load the dataset
    data = pd.read_csv(args.input_file)
    
    # Augment the data
    augmented_data = augment_anomalous_data(data, args.column_name, args.column_value, n_copies=args.n_copies)
    
    # Save the augmented dataset
    augmented_data.to_csv(args.output_file, index=False)
    print(f"Augmented dataset saved as '{args.output_file}'")
