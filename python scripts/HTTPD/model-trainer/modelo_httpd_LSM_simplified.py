import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import pickle
# Load dataset
data = pd.read_csv('../datasets/testing/HTTPD_FULL_TRAINING_DATA.csv')

# Preprocess data
print("Preprocessing data...")
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

# Prepare features and labels
X = pd.DataFrame({
    'requests_per_minute': data['requests_per_minute'],
})
X = np.hstack((X.values, endpoint_padded))
y = np.array([1 if flag == 'anomaly' else 0 for flag in data['flag']])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
print("Building the model...")
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=64, input_length=X.shape[1]))
model.add(SimpleRNN(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the model...")

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
# Save the tokenizer
tokenizer_save_path = '../models/tokenizer.pkl'
with open(tokenizer_save_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Tokenizer saved to {tokenizer_save_path}")

# Save the model
model.save('../models/LSM_simplified.h5')
print("Model saved.")
