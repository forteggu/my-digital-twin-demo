import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# File paths
model_save_path = '../models/httpd_enhanced_training_model.h5'
test_data_path = '../datasets/parsed/labelless/HTTPD_FULL_TRAINING_DATA_LABELLESS.csv'
output_predictions_path = '../model_outputs/enhanced_model_testing_results.csv'

# 1. Load the Trained Model
print("Loading the trained model...")
model = load_model(model_save_path)

# 2. Load and Preprocess Test Data
print("Loading test data...")
test_data = pd.read_csv(test_data_path)

# Add timestamp and time-based features
test_data['timestamp'] = pd.to_datetime(test_data['date'], format='%d/%b/%Y:%H:%M:%S %z', errors='coerce')
test_data['hour'] = test_data['timestamp'].dt.hour.fillna(0).astype(int)
test_data['hour_sin'] = np.sin(2 * np.pi * test_data['hour'] / 24)
test_data['hour_cos'] = np.cos(2 * np.pi * test_data['hour'] / 24)

# Calculate time delta and request rates
test_data = test_data.sort_values(['ip', 'timestamp'])
test_data['time_delta'] = test_data.groupby('ip')['timestamp'].diff().dt.total_seconds().fillna(0)
test_data['time_delta'] = test_data['time_delta'] / test_data['time_delta'].max()
test_data['requests_per_minute'] = test_data.groupby('ip')['timestamp'].transform(
    lambda x: x.diff().dt.total_seconds().fillna(60).rpow(-1) * 60
)

# Tokenize 'endpoint' for embeddings
test_data['endpoint'] = test_data['endpoint'].fillna('unknown').astype(str)
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(test_data['endpoint'])
endpoint_sequences = tokenizer.texts_to_sequences(test_data['endpoint'])

# Group test data by IP and create sequences
grouped_test = test_data.groupby('ip')
test_sequences = []
test_endpoint_sequences = []

for _, group in grouped_test:
    group = group.sort_values('timestamp')
    seq = group[['time_delta', 'hour_sin', 'hour_cos', 'requests_per_minute', 'size']].values
    test_sequences.append(seq)

    # Correctly process endpoint sequences for each group
    endpoint_seq = tokenizer.texts_to_sequences(group['endpoint'])
    endpoint_seq = [item for sublist in endpoint_seq for item in sublist]  # Flatten
    test_endpoint_sequences.append(endpoint_seq)

# Pad sequences
max_seq_length = 50
X_test_temporal = pad_sequences(test_sequences, maxlen=max_seq_length, padding='post', dtype='float32')
X_test_endpoint = pad_sequences(test_endpoint_sequences, maxlen=max_seq_length, padding='post', dtype='int32')

# 3. Make Predictions
print("Making predictions...")
predictions = model.predict([X_test_temporal, X_test_endpoint])
threshold = 0.5
predicted_flags = (predictions.flatten() > threshold).astype(int)

# Map predictions back to the test data
test_data['predicted_flag'] = np.nan
grouped_logs = test_data.groupby('ip')
for idx, (_, group) in enumerate(grouped_logs):
    test_data.loc[group.index, 'predicted_flag'] = predicted_flags[idx]

# Map 0/1 to normal/anomaly
test_data['predicted_flag'] = test_data['predicted_flag'].map({0: 'normal', 1: 'anomaly'})

# Drop unnecessary columns
columns_to_drop = ['hour', 'hour_sin', 'hour_cos', 'time_delta', 'requests_per_minute']
test_data = test_data.drop(columns=columns_to_drop)

# Save predictions to CSV
test_data.to_csv(output_predictions_path, index=False)
print(f"Predictions saved to {output_predictions_path}")

# Show example predictions
print("\nExample Predictions:")
print(test_data[['ip', 'date', 'endpoint', 'predicted_flag']].head(10))
