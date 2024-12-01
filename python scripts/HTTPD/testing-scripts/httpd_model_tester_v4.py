import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def preprocess_test_data(data_path, max_seq_length=50, tokenizer=None):
    print(f"Loading test data from {data_path}...")
    data = pd.read_csv(data_path)

    # Add timestamp and time-based features
    data['timestamp'] = pd.to_datetime(data['date'], format='%d/%b/%Y:%H:%M:%S %z', errors='coerce')
    data['hour'] = data['timestamp'].dt.hour.fillna(0).astype(int)
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)

    # Calculate time delta and request rates
    data = data.sort_values(['ip', 'timestamp'])
    data['time_delta'] = data.groupby('ip')['timestamp'].diff().dt.total_seconds().fillna(0)
    data['time_delta'] = data['time_delta'] / data['time_delta'].max()
    data['requests_per_minute'] = data.groupby('ip')['timestamp'].transform(
        lambda x: x.diff().dt.total_seconds().fillna(60).rpow(-1) * 60
    )

    # Normalize numerical features
    scaler = StandardScaler()
    data[['time_delta', 'requests_per_minute']] = scaler.fit_transform(data[['time_delta', 'requests_per_minute']])

    # Tokenize 'endpoint' for embeddings
    data['endpoint'] = data['endpoint'].fillna('unknown').astype(str)
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided for preprocessing test data.")

    # Group data by IP and create sequences
    grouped = data.groupby('ip')
    sequences = []
    endpoint_sequences = []

    for _, group in grouped:
        group = group.sort_values('timestamp')
        seq = group[['time_delta', 'hour_sin', 'hour_cos', 'requests_per_minute']].values
        sequences.append(seq)

        # Tokenize and pad endpoints for each IP group
        endpoint_seq = tokenizer.texts_to_sequences(group['endpoint'])
        endpoint_sequences.append([item for sublist in endpoint_seq for item in sublist])

    # Pad sequences
    X_temporal = pad_sequences(sequences, maxlen=max_seq_length, padding='post', dtype='float32')
    X_endpoint = pad_sequences(endpoint_sequences, maxlen=max_seq_length, padding='post', dtype='int32')

    return X_temporal, X_endpoint, data

def predict_and_save_results(model_path, tokenizer_path, data_path, output_predictions_path, max_seq_length=50):
    # Load the trained model and tokenizer
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    print(f"Loading tokenizer from {tokenizer_path}...")
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Preprocess the test data
    X_temporal, X_endpoint, original_data = preprocess_test_data(data_path, max_seq_length, tokenizer)

    # Make predictions
    print("Making predictions on test data...")
    predictions = model.predict([X_temporal, X_endpoint])
    predicted_flags = np.where(predictions > 0.5, 'anomaly', 'normal')  # Reverted threshold to 0.5 for balanced evaluation

    # Add predictions to the original dataset
    original_data['predicted_flag'] = np.repeat(predicted_flags, [len(g) for _, g in original_data.groupby('ip')])

    # Save the results
    print(f"Saving predictions to {output_predictions_path}...")
    original_data.to_csv(output_predictions_path, index=False)
    print(f"Predictions saved to {output_predictions_path}")

    # Plot prediction distribution
    plt.figure(figsize=(10, 6))
    plt.hist(predictions, bins=50, alpha=0.75, color='blue')
    plt.xlabel('Prediction Scores')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Scores')
    plt.grid(axis='y')
    plt.show()

# Define paths
model_path = '../models/httpd_enhanced_iterative_training_model_v2.h5'
tokenizer_path = '../models/httpd_enhanced_iterative_training_tokenizer_v2.pkl'
data_path = '../datasets/parsed/labelless/HTTPD_FULL_TRAINING_DATA_LABELLESS.csv'
output_predictions_path = '../model_outputs/predicted_httpd_flags_test.csv'

# Run the prediction script
predict_and_save_results(model_path, tokenizer_path, data_path, output_predictions_path)
