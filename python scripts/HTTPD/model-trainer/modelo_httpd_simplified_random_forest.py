import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pickle

# Load dataset
data = pd.read_csv('../datasets/testing/HTTPD_FULL_TRAINING_DATA.csv')

# Preprocess data
print("Preprocessing data...")
data['timestamp'] = pd.to_datetime(data['date'], format='%d/%b/%Y:%H:%M:%S %z', errors='coerce')

# Calculate requests per second per IP
data = data.sort_values(['ip', 'timestamp'])
data['time_delta'] = data.groupby('ip')['timestamp'].diff().dt.total_seconds().fillna(0)
data['requests_per_second'] = data.groupby('ip')['timestamp'].transform(
    lambda x: x.diff().dt.total_seconds().replace([np.inf, -np.inf], 1).fillna(1).rpow(-1)
)

# Set threshold for bursts of requests (e.g., only label as burst if requests per second > 30)
data['is_burst'] = data['requests_per_second'].apply(lambda x: 1 if x > 30 else 0)

# Tokenize 'endpoint' for embedding
data['endpoint'] = data['endpoint'].fillna('unknown').astype(str)
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(data['endpoint'])
endpoint_sequences = tokenizer.texts_to_sequences(data['endpoint'])
endpoint_padded = pad_sequences(endpoint_sequences, maxlen=50, padding='post')

# Prepare features and labels
X = pd.DataFrame({
    'requests_per_second': data['requests_per_second'].replace([np.inf, -np.inf], 0).fillna(0),
})
X = np.hstack((X.values, endpoint_padded))
y = np.array([1 if flag == 'anomaly' and burst == 1 else 0 for flag, burst in zip(data['flag'], data['is_burst'])])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Random Forest model
print("Building the Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
print("Training the model...")
model.fit(X_train, y_train)

# Evaluate the model
print("Evaluating the model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model
model_save_path = '../models/random_forest_model_updated.pkl'
joblib.dump(model, model_save_path)
print(f"Model saved to {model_save_path}")

# Save the tokenizer
tokenizer_save_path = '../models/tokenizer_updated.pkl'
with open(tokenizer_save_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Tokenizer saved to {tokenizer_save_path}")
