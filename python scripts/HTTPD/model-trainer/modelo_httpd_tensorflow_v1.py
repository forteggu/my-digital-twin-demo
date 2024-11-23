import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.models import load_model


# File paths
data_path = '../datasets/parsed/HTTPD_FULL_TRAINING_DATA.csv'
model_save_path = '..models/ip_based_anomaly_detector.h5'
test_data_path = '../datasets/parsed/labelless/HTTPD_FULL_TRAINING_DATA_LABELLESS.csv'
output_predictions_path = '../model_outputs/ip_based_predictions.csv'

# 1. Load and Preprocess Data
print("Loading data...")
data = pd.read_csv(data_path)
if 'flag' in data.columns:
    data['flag'] = data['flag'].map({'normal': 0, 'anomaly': 1}).astype(int)
# Add timestamp and time-based features
data['timestamp'] = pd.to_datetime(data['date'], format='%d/%b/%Y:%H:%M:%S %z', errors='coerce')
data['hour'] = data['timestamp'].dt.hour.fillna(0).astype(int)
data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)

# Encode categorical columns
endpoint_encoder = LabelEncoder()
data['endpoint_encoded'] = endpoint_encoder.fit_transform(data['endpoint'].fillna('unknown').astype(str))

# Fill missing sizes with 0
data['size'] = data['size'].fillna(0).astype(float)

# Calculate time delta between requests for each IP
data = data.sort_values(['ip', 'timestamp'])
data['time_delta'] = data.groupby('ip')['timestamp'].diff().dt.total_seconds().fillna(0)
data['time_delta'] = data['time_delta'] / data['time_delta'].max()  # Normalize time delta

# Group data by IP and create sequences
grouped = data.groupby('ip')
sequences = []
labels = []

for _, group in grouped:
    group = group.sort_values('timestamp')
    seq = group[['time_delta', 'hour_sin', 'hour_cos', 'size', 'endpoint_encoded']].values
    sequences.append(seq)
    if 'flag' in group.columns:
        labels.append(group['flag'].values[-1])  # Label the sequence with the last log's flag

# Pad sequences
max_seq_length = 50  # Adjust based on dataset
X_temporal = pad_sequences(sequences, maxlen=max_seq_length, padding='post', dtype='float32')
if labels:
    y = np.array(labels)

# Split into training and validation sets
if labels:
    X_train, X_val, y_train, y_val = train_test_split(X_temporal, y, test_size=0.2, random_state=42)

# 2. Build the Model
print("Building model...")
sequence_input = Input(shape=(max_seq_length, 5))  # Adjust based on number of features per request
#lstm_out = LSTM(64, return_sequences=False)(sequence_input)
lstm_out = SimpleRNN(64, return_sequences=False)(sequence_input)
dense1 = Dense(64, activation='relu')(lstm_out)
dropout = Dropout(0.5)(dense1)
output = Dense(1, activation='sigmoid')(dropout)

model = Model(inputs=sequence_input, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 3. Train the Model
if labels:
    print("Training model...")
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=20, 
        batch_size=32
    )

    # Save the trained model
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

# 4. Test the Model on New Data
print("Loading test data...")
test_data = pd.read_csv(test_data_path)
if 'flag' in test_data.columns:
    test_data['flag'] = test_data['flag'].map({'normal': 0, 'anomaly': 1}).astype(int)
# Preprocess test data (same as training data)
test_data['timestamp'] = pd.to_datetime(test_data['date'], format='%d/%b/%Y:%H:%M:%S %z', errors='coerce')
test_data['hour'] = test_data['timestamp'].dt.hour.fillna(0).astype(int)
test_data['hour_sin'] = np.sin(2 * np.pi * test_data['hour'] / 24)
test_data['hour_cos'] = np.cos(2 * np.pi * test_data['hour'] / 24)
test_data['endpoint_encoded'] = endpoint_encoder.transform(test_data['endpoint'].fillna('unknown').astype(str))
test_data['size'] = test_data['size'].fillna(0).astype(float)
test_data = test_data.sort_values(['ip', 'timestamp'])
test_data['time_delta'] = test_data.groupby('ip')['timestamp'].diff().dt.total_seconds().fillna(0)
test_data['time_delta'] = test_data['time_delta'] / test_data['time_delta'].max()

# Group test data by IP and create sequences
grouped_test = test_data.groupby('ip')
test_sequences = []

for _, group in grouped_test:
    group = group.sort_values('timestamp')
    seq = group[['time_delta', 'hour_sin', 'hour_cos', 'size', 'endpoint_encoded']].values
    test_sequences.append(seq)

# Pad sequences
X_test = pad_sequences(test_sequences, maxlen=max_seq_length, padding='post', dtype='float32')

# Load the trained model and make predictions
print("Making predictions...")
model = load_model(model_save_path)
predictions = model.predict(X_test)
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
columns_to_drop = ['hour', 'hour_sin', 'hour_cos', 'endpoint_encoded', 'time_delta']
test_data = test_data.drop(columns=columns_to_drop)

# Save predictions to CSV
test_data.to_csv(output_predictions_path, index=False)
print(f"Predictions saved to {output_predictions_path}")

# Show example predictions
print("\nExample Predictions:")
print(test_data[['ip', 'date', 'endpoint', 'predicted_flag']].head(10))
