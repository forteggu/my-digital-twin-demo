import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# File paths
model_save_path = '../models/hybrid_anomaly_detector.h5'
real_logs_path = '../datasets/parsed/labelless/HTTPD_FULL_TRAINING_DATA_LABELLESS.csv'
output_predictions_path = '../model_outputs/httpd_training_hybrid_model_test_results.csv'

# 1. Load the Trained Model
print("Loading the trained model...")
model = load_model(model_save_path)

# 2. Load and Preprocess Real Logs
print("Loading real logs...")
real_logs = pd.read_csv(real_logs_path)

# Convert date column to timestamp and extract time-based features
real_logs['timestamp'] = pd.to_datetime(real_logs['date'], format='%d/%b/%Y:%H:%M:%S %z', errors='coerce')
real_logs['hour'] = real_logs['timestamp'].dt.hour.fillna(0).astype(int)
real_logs['hour_sin'] = np.sin(2 * np.pi * real_logs['hour'] / 24)
real_logs['hour_cos'] = np.cos(2 * np.pi * real_logs['hour'] / 24)

# Encode categorical columns (ensure the same encoder as during training is used)
endpoint_encoder = LabelEncoder()
real_logs['endpoint_encoded'] = endpoint_encoder.fit_transform(real_logs['endpoint'].fillna('unknown').astype(str))

# Fill missing sizes with 0
real_logs['size'] = real_logs['size'].fillna(0).astype(float)

# Group logs by IP or identifier to create sequences
grouped = real_logs.groupby('ip')
sequences = []

for _, group in grouped:
    group = group.sort_values('timestamp')
    seq = group[['hour_sin', 'hour_cos', 'size']].values  # Temporal features
    endpoint_seq = group['endpoint_encoded'].values  # Endpoint categorical feature
    sequences.append((seq, endpoint_seq))

# Pad sequences to uniform length
max_seq_length = 50  # Use the same max sequence length as during training
temporal_features = 3  # Number of temporal features
X_temporal = [s[0] for s in sequences]
X_temporal = pad_sequences(X_temporal, maxlen=max_seq_length, padding='post', dtype='float32')

X_endpoint = [s[1] for s in sequences]
X_endpoint = pad_sequences(X_endpoint, maxlen=max_seq_length, padding='post', dtype='int32')

# 3. Predict Anomalies
print("Predicting anomalies...")
predictions = model.predict([X_temporal, X_endpoint])

# Apply a threshold to classify predictions
threshold = 0.5  # Default threshold
predicted_flags = (predictions.flatten() > threshold).astype(int)

# 4. Save Predictions
# Add predictions back to the original data
real_logs['predicted_flag'] = np.nan  # Initialize with NaN
grouped_logs = real_logs.groupby('ip')
for idx, (_, group) in enumerate(grouped_logs):
    real_logs.loc[group.index, 'predicted_flag'] = predicted_flags[idx]

# Map predicted_flag to "normal" or "anomaly"
real_logs['predicted_flag'] = real_logs['predicted_flag'].map({0: 'normal', 1: 'anomaly'})

# Drop unnecessary columns
columns_to_drop = ['hour', 'hour_sin', 'hour_cos', 'endpoint_encoded']
real_logs = real_logs.drop(columns=columns_to_drop)

# Save predictions to a CSV
real_logs.to_csv(output_predictions_path, index=False)
print(f"Predictions saved to {output_predictions_path}")

# 5. Show Example Predictions
print("\nExample Predictions:")
print(real_logs[['ip', 'date', 'endpoint', 'predicted_flag']].head(10))
