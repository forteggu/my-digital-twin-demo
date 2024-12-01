import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the pre-trained model
model_path = '../models/httpd_enhanced_iterative_training_model_v2.h5'
print(f"Loading model from {model_path}...")
model = load_model(model_path)

# Load the dataset for testing
test_data_path = '../models/parsed/labelless/HTTPD_FULL_TRAINING_DATA_LABELLESS.csv'
print(f"Loading test dataset from {test_data_path}...")
data = pd.read_csv(test_data_path)

# Preprocess data
print("Preprocessing test data...")
data['timestamp'] = pd.to_datetime(data['date'], format='%d/%b/%Y:%H:%M:%S %z', errors='coerce')

# Calculate requests per minute per IP
data = data.sort_values(['ip', 'timestamp'])
data['time_delta'] = data.groupby('ip')['timestamp'].diff().dt.total_seconds().fillna(0)
data['requests_per_minute'] = data.groupby('ip')['timestamp'].transform(
    lambda x: 60 / x.diff().dt.total_seconds().fillna(60)
)

# Tokenize 'endpoint' for embedding
data['endpoint'] = data['endpoint'].fillna('unknown').astype(str)
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(data['endpoint'])
endpoint_sequences = tokenizer.texts_to_sequences(data['endpoint'])
endpoint_padded = pad_sequences(endpoint_sequences, maxlen=50, padding='post')

# Prepare features for prediction
X = pd.DataFrame({
    'requests_per_minute': data['requests_per_minute'],
})
X = np.hstack((X.values, endpoint_padded))

# Make predictions
print("Making predictions...")
predictions = model.predict(X)

# Interpret predictions
predictions_binary = [1 if pred >= 0.5 else 0 for pred in predictions]
data['predicted_flag'] = ['anomaly' if pred == 1 else 'normal' for pred in predictions_binary]

# Save predictions to CSV
output_path = '../models/parsed/labelless/HTTPD_FULL_TRAINING_DATA_PREDICTIONS.csv'
data.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")
