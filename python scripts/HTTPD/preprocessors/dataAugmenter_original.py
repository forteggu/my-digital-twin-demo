import pandas as pd
import numpy as np

def augment_anomalous_data(data, n_copies=5):
    """
    Augment data by adding noise to numeric features and randomizing the IP field.
    Only rows where data['flag'] == anomaly are modified.
    
    Args:
        data (pd.DataFrame): Input dataset.
        n_copies (int): Number of augmented copies to generate for each anomalous row.

    Returns:
        pd.DataFrame: Augmented dataset.
    """
    # Filter rows with flag == anomaly
    anomalous_data = data[data['flag'] == "anomaly"].copy()
    
    augmented_data = []
    
    for _ in range(n_copies):
        # Copy the data
        augmented = anomalous_data.copy()
        
        # Add noise to numeric columns
        if 'time_delta' in augmented.columns:
            augmented['time_delta'] = augmented['time_delta'] * (1 + np.random.uniform(-0.1, 0.1, len(augmented)))
        
        # if 'size' in augmented.columns:
        #     augmented['size'] = augmented['size'] * (1 + np.random.uniform(-0.1, 0.1, len(augmented)))
        
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

# Example usage
# Load the dataset
data = pd.read_csv('../datasets/parsed/httpd_logs_with_behavior_parsed.csv')

# Augment the data
augmented_data = augment_anomalous_data(data, n_copies=500)

# Save the augmented dataset
output_dir='../datasets/parsed/augmented_httpd_logs_with_behavior_parsed.csv'
augmented_data.to_csv(output_dir, index=False)
print("Augmented dataset saved as '"+output_dir+"'")
