import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
# Paths to files and saved model
model_path = "../models/sqli_detection_model_v2.h5"  # Path to the saved model
real_log_path = "../datasets/sqli_full_testing_dataset.csv"  # Path to the file containing Testing Logs

# Load the trained model
model = load_model(model_path)

# Load Testing Logs (assuming the file has a 'query' column)
testing_logs_df = pd.read_csv(real_log_path, header=None, names=['query'])
print("Loaded Testing Logs:")
print(testing_logs_df.head())

# Preprocessing
def preprocess_logs(logs, tokenizer, max_sequence_length):
    # Tokenize the queries
    sequences = tokenizer.texts_to_sequences(logs)
    # Pad the sequences
    return pad_sequences(sequences, maxlen=max_sequence_length)

# Initialize the tokenizer (must match the one used during training)
# Example: You should use the same word index that was saved or trained earlier
tokenizer = Tokenizer(oov_token="<OOV>")  # Reinitialize tokenizer
# (Rebuild the word index used during training - if saved, load it)
# Replace `word_index_path` with the actual saved tokenizer's word_index file, if applicable.
word_index_path = "../models/tokenizer_word_index.json"
with open(word_index_path, 'r') as f:
    tokenizer.word_index = json.load(f)

# Maximum sequence length (must match training)
max_sequence_length = 100  # Replace with the same value used during training

# Preprocess the Testing Logs
X_real = preprocess_logs(testing_logs_df['query'], tokenizer, max_sequence_length)

# Predict using the model
predictions = model.predict(X_real)

# Convert predictions to labels (0 for normal, 1 for SQL injection)
threshold = 0.5  # You can adjust this threshold based on your needs
testing_logs_df['prediction'] = (predictions > threshold).astype(int)

# Add a label for easier interpretation
testing_logs_df['label'] = testing_logs_df['prediction'].apply(lambda x: 'SQL Injection' if x == 1 else 'Normal')

# Display predictions
print("Predictions:")
print(testing_logs_df[['query', 'label']])

# Save the predictions
testing_logs_df[['query', 'label']].to_csv("testing_log_predictions.csv", index=False)
print("Predictions saved to 'testing_log_predictions.csv'")
