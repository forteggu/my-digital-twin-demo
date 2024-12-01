import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, SimpleRNN, Dense, Dropout, Embedding, Concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pickle

def preprocess_training_data(data_path, max_seq_length=50, tokenizer=None):
    print(f"Loading training data from {data_path}...")
    data = pd.read_csv(data_path)

    # Add timestamp and time-based features
    data['timestamp'] = pd.to_datetime(data['date'], format='%d/%b/%Y:%H:%M:%S %z', errors='coerce')
    data['hour'] = data['timestamp'].dt.hour.fillna(0).astype(int)

    # Calculate time delta and request rates
    data = data.sort_values(['ip', 'timestamp'])
    data['time_delta'] = data.groupby('ip')['timestamp'].diff().dt.total_seconds().fillna(0)
    data['time_delta'] = data['time_delta'] / data['time_delta'].max()
    data['requests_per_minute'] = data.groupby('ip')['timestamp'].transform(
        lambda x: x.diff().dt.total_seconds().fillna(60).rpow(-1) * 60
    )

    # Tokenize 'endpoint' for embeddings
    data['endpoint'] = data['endpoint'].fillna('unknown').astype(str)
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=10000)
        tokenizer.fit_on_texts(data['endpoint'])

    # Group data by IP and create sequences
    grouped = data.groupby('ip')
    sequences = []
    endpoint_sequences = []
    labels = []

    for _, group in grouped:
        group = group.sort_values('timestamp')
        seq = group[['time_delta', 'requests_per_minute']].values
        sequences.append(seq)

        # Tokenize and pad endpoints for each IP group
        endpoint_seq = tokenizer.texts_to_sequences(group['endpoint'])
        endpoint_sequences.append([item for sublist in endpoint_seq for item in sublist])

        if 'flag' in group.columns:
            labels.append(1 if group['flag'].values[-1] == 'anomaly' else 0)

    # Pad sequences
    X_temporal = pad_sequences(sequences, maxlen=max_seq_length, padding='post', dtype='float32')
    X_endpoint = pad_sequences(endpoint_sequences, maxlen=max_seq_length, padding='post', dtype='int32')
    y = np.array(labels) if labels else None

    return X_temporal, X_endpoint, y, tokenizer

def build_model(max_seq_length=50):
    temporal_input = Input(shape=(max_seq_length, 2))  # Temporal features
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

def train_model_on_data(model, data_path, max_seq_length=50, tokenizer=None, epochs=10):
    X_temporal, X_endpoint, y, tokenizer = preprocess_training_data(data_path, max_seq_length, tokenizer)
    X_train_temp, X_val_temp, X_train_end, X_val_end, y_train, y_val = train_test_split(
        X_temporal, X_endpoint, y, test_size=0.2, random_state=42
    )
    print(f"Training on dataset: {data_path}...")
    history = model.fit(
        [X_train_temp, X_train_end], y_train,
        validation_data=([X_val_temp, X_val_end], y_val),
        epochs=epochs,
        batch_size=32
    )
    return model, tokenizer, history

def train_iteratively(subset_paths, model_save_path, tokenizer_save_path, max_seq_length=50):
    # Initialize model and tokenizer
    model = build_model(max_seq_length)
    tokenizer = None

    # Fine-tune on each subset
    for subset in subset_paths:
        model, tokenizer, _ = train_model_on_data(model, subset, max_seq_length, tokenizer, epochs=10)

    # Save final model and tokenizer
    model.save(model_save_path)
    with open(tokenizer_save_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Final model saved to {model_save_path}")
    print(f"Tokenizer saved to {tokenizer_save_path}")
    return model

# Paths to datasets
subset_paths = [
    '../datasets/testing/parsed_benign.csv',
    '../datasets/testing/parsed_ffuf-extensions.csv',
    '../datasets/testing/parsed_ffuf-small.csv',
    '../datasets/testing/parsed_search_benign.csv',
    '../datasets/testing/augmented_httpd_logs_with_behavior_parsed.csv',
    '../datasets/testing/HTTPD_FULL_TRAINING_DATA.csv'
]
model_save_path = '../models/httpd_enhanced_iterative_training_model_v2.h5'
tokenizer_save_path = '../models/httpd_enhanced_iterative_training_tokenizer_v2.pkl'

# Train iteratively
model = train_iteratively(subset_paths, model_save_path, tokenizer_save_path)
