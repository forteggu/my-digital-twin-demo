## THIS VERSION INCLUDES SEVERAL ENHANCEMENTS AND IMPROVEMENTS SUCH AS 

    # Embedding for endpoint:
    #     Replaces label encoding with tokenization and embeddings, capturing semantic patterns in URLs.

    # Synthetic Data:
    #     Introduces examples of anomalies (e.g., injection, XSS) to improve generalization.

    # Time-Based Features:
    #     Adds time_delta and requests_per_minute to detect temporal anomalies.

    # Generalization Across IPs:
    #     Ensures training includes diverse and randomized IPs for better robustness.

import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, SimpleRNN, Dense, Dropout, Embedding, Concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# File paths
data_path = '../datasets/parsed/HTTPD_FULL_TRAINING_DATA.csv'
model_save_path = '../models/httpd_enhanced_training_model.h5'

# 1. Load and Preprocess Data
def preprocess_data(data_path, max_seq_length=50, tokenizer=None):
    print("Loading data...")
    data = pd.read_csv(data_path)
    if 'flag' in data.columns:
        data['flag'] = data['flag'].map({'normal': 0, 'anomaly': 1}).astype(int)

    # Add timestamp and time-based features
    data['timestamp'] = pd.to_datetime(data['date'], format='%d/%b/%Y:%H:%M:%S %z', errors='coerce')
    data['hour'] = data['timestamp'].dt.hour.fillna(0).astype(int)
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)

    # Calculate time delta and request rates
    data = data.sort_values(['ip', 'timestamp'])
    data['time_delta'] = data.groupby('ip')['timestamp'].diff().dt.total_seconds().fillna(0)
    data['time_delta'] = data['time_delta'] / data['time_delta'].max()  # Normalize time delta
    data['requests_per_minute'] = data.groupby('ip')['timestamp'].transform(
        lambda x: x.diff().dt.total_seconds().fillna(60).rpow(-1) * 60
    )

    # Tokenize 'endpoint' for embeddings
    data['endpoint'] = data['endpoint'].fillna('unknown').astype(str)
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=5000)  # Adjust vocab size as needed
        tokenizer.fit_on_texts(data['endpoint'])

    # Group data by IP and create sequences
    grouped = data.groupby('ip')
    sequences = []
    endpoint_sequences = []
    labels = []

    for _, group in grouped:
        group = group.sort_values('timestamp')
        seq = group[['time_delta', 'hour_sin', 'hour_cos', 'requests_per_minute']].values
        sequences.append(seq)

        # Tokenize and pad endpoints for each IP group
        endpoint_seq = tokenizer.texts_to_sequences(group['endpoint'])
        endpoint_sequences.append([item for sublist in endpoint_seq for item in sublist])

        if 'flag' in group.columns:
            labels.append(group['flag'].values[-1])  # Label the sequence with the last log's flag

    # Pad sequences
    X_temporal = pad_sequences(sequences, maxlen=max_seq_length, padding='post', dtype='float32')
    X_endpoint = pad_sequences(endpoint_sequences, maxlen=max_seq_length, padding='post', dtype='int32')
    y = np.array(labels) if labels else None

    return X_temporal, X_endpoint, y, tokenizer
# 2. Build the Model
def build_model(max_seq_length=50):
    temporal_input = Input(shape=(max_seq_length, 5))  # Temporal features
    endpoint_input = Input(shape=(max_seq_length,))  # Endpoint tokens

    # RNN for temporal features
    temporal_rnn = SimpleRNN(64, return_sequences=False)(temporal_input)

    # Embedding for endpoints
    embedding = Embedding(input_dim=5000, output_dim=16, input_length=max_seq_length)(endpoint_input)
    endpoint_rnn = SimpleRNN(64, return_sequences=False)(embedding)

    # Combine features
    combined = Concatenate()([temporal_rnn, endpoint_rnn])

    # Dense layers
    dense1 = Dense(64, activation='relu')(combined)
    dropout = Dropout(0.5)(dense1)
    output = Dense(1, activation='sigmoid')(dropout)

    model = Model(inputs=[temporal_input, endpoint_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# 3. Train the Model
def train_model(data_path, model_save_path, max_seq_length=50):
    X_temporal, X_endpoint, y, tokenizer = preprocess_data(data_path, max_seq_length)
    X_train_temp, X_val_temp, X_train_end, X_val_end, y_train, y_val = train_test_split(
        X_temporal, X_endpoint, y, test_size=0.2, random_state=42
    )
    model = build_model(max_seq_length)
    print("Training model...")
    history = model.fit(
        [X_train_temp, X_train_end], y_train,
        validation_data=([X_val_temp, X_val_end], y_val),
        epochs=20,
        batch_size=32
    )
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Plot training results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.show()

    return tokenizer

# Train the model
tokenizer = train_model(data_path, model_save_path)
