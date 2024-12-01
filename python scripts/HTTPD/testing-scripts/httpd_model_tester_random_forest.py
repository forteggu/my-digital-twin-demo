import pandas as pd
import numpy as np
import joblib
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model and tokenizer
model_path = '../models/random_forest_model.pkl'
tokenizer_path = '../models/tokenizer.pkl'
print(f"Loading model from {model_path}...")
model = joblib.load(model_path)
print(f"Loading tokenizer from {tokenizer_path}...")
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the real log dataset
real_log_path = '../datasets/parsed/labelless/HTTPD_FULL_TRAINING_DATA_LABELLESS.csv'
print(f"Loading real log data from {real_log_path}...")
data = pd.read_csv(real_log_path)

# Preprocess data
print("Preprocessing real log data...")
data['timestamp'] = pd.to_datetime(data['date'], format='%d/%b/%Y:%H:%M:%S %z', errors='coerce')

# Calculate requests per minute per IP
data = data.sort_values(['ip', 'timestamp'])
data['time_delta'] = data.groupby('ip')['timestamp'].diff().dt.total_seconds().fillna(0)
data['requests_per_minute'] = data.groupby('ip')['timestamp'].transform(
    lambda x: 60 / x.diff().dt.total_seconds().replace([np.inf, -np.inf], 60).fillna(60)
)

# Tokenize 'endpoint' for embedding
data['endpoint'] = data['endpoint'].fillna('unknown').astype(str)
endpoint_sequences = tokenizer.texts_to_sequences(data['endpoint'])
endpoint_padded = pad_sequences(endpoint_sequences, maxlen=50, padding='post')

# Prepare features for prediction
X_real = pd.DataFrame({
    'requests_per_minute': data['requests_per_minute'].replace([np.inf, -np.inf], 0).fillna(0),
})
X_real = np.hstack((X_real.values, endpoint_padded))

# Make predictions
print("Making predictions on real log data...")
predictions = model.predict(X_real)
data['predicted_flag'] = ['anomaly' if pred == 1 else 'normal' for pred in predictions]

# Save predictions to CSV
output_path = '../model_outputs/predicted_httpd_flags_rf_test.csv'
data.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")
