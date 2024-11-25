import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SimpleRNN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

# Paths to your CSV files
normal_csv_path = "../datasets/parsed/normal_queries_with_prefix.csv"
injection_csv_path = "../datasets/parsed/final_injection_queries_with_prefix.csv"

# Load the datasets
normal_df = pd.read_csv(normal_csv_path)
injection_df = pd.read_csv(injection_csv_path)

# Function to remove surrounding quotes from payloads
def remove_surrounding_quotes(line):
    if isinstance(line, str) and line.startswith('"') and line.endswith('"'):
        return line[1:-1]  # Remove the outermost quotes
    return line

# Apply function to the 'query' column only
normal_df['query'] = normal_df['query'].apply(remove_surrounding_quotes)
injection_df['query'] = injection_df['query'].apply(remove_surrounding_quotes)

# Convert the 'label' column to numerical values
# Assuming "normal" corresponds to 0 and "sqli_attack" corresponds to 1
normal_df['label'] = normal_df['label'].apply(lambda x: 0 if x == "normal" else 1)
injection_df['label'] = injection_df['label'].apply(lambda x: 0 if x == "normal" else 1)

# Check the structure to confirm the label column is present and correct
print(injection_df.head())

# Combine the datasets and shuffle
data = pd.concat([normal_df, injection_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# Check and handle missing values
data['query'] = data['query'].fillna("missing_query")  # Replace missing values with placeholder

# Tokenize the queries
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(data['query'])
sequences = tokenizer.texts_to_sequences(data['query'])
word_index = tokenizer.word_index

with open("../models/tokenizer_word_index.json", "w") as f:
    json.dump(word_index, f)

print("Tokenizer saved successfully.")

# Pad sequences to ensure uniform input size
max_sequence_length = 100
X = pad_sequences(sequences, maxlen=max_sequence_length)
y = data['label'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Embedding(input_dim=len(word_index) + 1, output_dim=128, input_length=max_sequence_length),
    SimpleRNN(64, return_sequences=False),  # Replace SimpleRNN with LSTM if your GPU supports it
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64)

# Save the trained model
model_save_path = "../models/sqli_detection_model_v2.h5"
model.save(model_save_path)

# Print the save path
print(f"Model saved to {model_save_path}")
