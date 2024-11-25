import argparse
import json
import pandas as pd
import numpy as np
from kubernetes import client, config, watch
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
import re

# Global variables
EVENT_RECEIVER_URL = "http://34.65.255.107:8000/log-ms-event"
model_path = "models/sqli_detection_model_v2.h5"  # Path to the saved model
word_index_path = "models/tokenizer_word_index.json"  # Path to tokenizer's word index
max_sequence_length = 100  # Must match training
oov_token = "<OOV>"

# Load the trained model
model = load_model(model_path)

# Initialize the tokenizer
tokenizer = Tokenizer(oov_token=oov_token)
with open(word_index_path, 'r') as f:
    tokenizer.word_index = json.load(f)

# Function to preprocess logs
def preprocess_logs(logs):
    sequences = tokenizer.texts_to_sequences(logs)
    return pad_sequences(sequences, maxlen=max_sequence_length)

# Function to predict with the model
def predict_with_model(logs):
    preprocessed_logs = preprocess_logs(logs)
    predictions = model.predict(preprocessed_logs)
    return (predictions.flatten() > 0.5).astype(int)


# Function to process and predict logs
def process_and_predict_log(raw_log):
     # Use regex to extract the timestamp and the actual query
    match = re.match(r"^(\S+)\s+(.*)$", raw_log)
    if match:
        timestamp, query = match.groups()
    else:
        # Handle cases where the log format is unexpected
        timestamp, query = None, raw_log

    fixed_prefix = "Ejecutando consulta: SELECT * FROM users WHERE name = "
    query=query.replace(fixed_prefix,"")
    # Wrap the parsed data into a DataFrame
    df = pd.DataFrame({
        'timestamp': [timestamp],
        'query': [query]
    })
    # print(f'Timestamp: {df["timestamp"].iloc[0]} | Query: {df["query"].iloc[0]}')
    # Preprocess the log and predict
    prediction = predict_with_model(df['query'])
    label = 'SQL Injection' if prediction[0] == 1 else 'Normal'

    # Structure the result
    return {
        "timestamp": df["timestamp"],
        "raw_log": df["query"],
        "predicted_label": label
    }

# Prepare the event object for sending
def prepare_event_object(row):
# Extract scalar values from Series or use defaults
    raw_log = row.get("raw_log")
    if isinstance(raw_log, pd.Series):
        raw_log = raw_log.iloc[0]  # Extract scalar value if it's a Series

    timestamp = row.get("timestamp")
    if isinstance(timestamp, pd.Series):
        timestamp = timestamp.iloc[0]  # Extract scalar value if it's a Series

    predicted_label = row.get("predicted_label", "Unknown")
    log_type = row.get("log_type", "")
    user = row.get("user", "")
    ip = row.get("ip", "")

    # Ensure timestamp is a string
    if isinstance(timestamp, pd.Timestamp):  # If timestamp is a Pandas timestamp
        timestamp = timestamp.isoformat()
    elif isinstance(timestamp, (int, float)):
        timestamp = str(timestamp)

    # Prepare the event payload
    event = {
        "timestamp": timestamp,
        "raw_log": raw_log,
        "predicted_label": predicted_label
    }

    # Debugging: Check the event before sending
    print(f"Prepared Event: {event}")
    return event

# Function to send events to a receiver
def send_event(row):
    preparedEventObject=prepare_event_object(row)
    try:
        response = requests.post(EVENT_RECEIVER_URL, json=preparedEventObject)
        if response.status_code == 200:
            print(f"Event sent successfully: {preparedEventObject}")
        else:
            print(f"Failed to send event: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Error sending event: {e}")

# Function to stream logs from a Kubernetes pod
def stream_pod_logs(namespace, pod_name, container_name=None, only_live=False):
    count=0
    try:
        # Load Kubernetes config
        config.load_kube_config()

        # Create Kubernetes CoreV1 API client
        v1 = client.CoreV1Api()

        # Initialize log stream
        w = watch.Watch()
        print(f"Streaming logs in real-time from pod {pod_name} in namespace {namespace}...")

        for line in w.stream(
            v1.read_namespaced_pod_log,
            name=pod_name,
            namespace=namespace,
            container=container_name,
            follow=not only_live,
            #since_seconds=1 if only_live else None,
            timestamps=True
        ):
            if(line):
                # Process the raw log
                prediction_result = process_and_predict_log(line)
                send_event(prediction_result)
                if(count<=5):
                    count+=1
                else:
                    w.close()
            else:
                exit(1)

    except client.exceptions.ApiException as e:
        print(f"Error streaming logs: {e}")


# Main function
if __name__ == "__main__":
    # Configure script arguments
    parser = argparse.ArgumentParser(description="Stream logs from a Kubernetes pod")
    parser.add_argument("--only-live", action="store_true", help="If set, only fetch live logs")
    args = parser.parse_args()

    # Kubernetes configuration
    namespace = "default"
    pod_name = "my-digital-twin-vulnerable-microservice-deployment-555df5629cbj"
    container_name = None

    # Start streaming logs
    stream_pod_logs(namespace, pod_name, container_name, only_live=args.only_live)
    exit(1)
