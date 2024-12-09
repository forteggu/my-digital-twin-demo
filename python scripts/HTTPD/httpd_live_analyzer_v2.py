import argparse
import pandas as pd
import numpy as np
from kubernetes import client, config, watch
import joblib
import pickle
import requests
import re
import redis
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime

# Global variables
EVENT_RECEIVER_URL = "http://34.65.255.107:8000/log-httpd-event"
model_path = "models/random_forest_model.pkl"  # Path to the saved model
tokenizer_path = "models/tokenizer.pkl"  # Path to tokenizer
redis_client = redis.Redis(host='localhost', port=6379, db=0)  # Redis client
print("Vaciando la base de datos Redis...")
redis_client.flushdb()
print("Base de datos Redis vaciada.")

# Load the trained model
print(f"Loading model from {model_path}...")
model = joblib.load(model_path)

# Load the tokenizer
print(f"Loading tokenizer from {tokenizer_path}...")
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to parse each log line
def parseLine(line):
    # Remove any surrounding quotes from the line
    log_pattern = re.compile(
        r'(?P<ip>[\d\.]+) - - \[(?P<date>[\w:/]+\s[+\-]\d+)\] "(?P<method>[^\s"]+)\s(?P<endpoint>[^\s"]+)\s(?P<protocol>HTTP/[\d\.]+)?" (?P<status>\d+) (?P<size>[\d\-]+)'
    )
    line = line.strip()
    if line.startswith('"') and line.endswith('"'):
        line = line[1:-1]  # Remove the outermost quotes
        line = line.replace('""', '"')

    match = log_pattern.match(line)
    if match:
        log_entry = match.groupdict()
        log_entry.pop("protocol", None)
        # Convert size to integer where applicable
        log_entry["size"] = int(log_entry["size"]) if log_entry["size"].isdigit() else None
        # Add the flag value to the log entry if provided
        return log_entry
    else:
        return False

# Function to preprocess logs
def preprocess_logs(logs):
    endpoint_sequences = tokenizer.texts_to_sequences(logs)
    return pad_sequences(endpoint_sequences, maxlen=50, padding='post')

# Function to predict with the model
def predict_with_model(logs, requests_per_minute):
    endpoint_padded = preprocess_logs(logs)
    X_real = pd.DataFrame({
        'requests_per_minute': requests_per_minute,
    })
    X_real = np.hstack((X_real.values, endpoint_padded))
    predictions = model.predict(X_real)
    return predictions

# Function to calculate requests per minute from Redis
def calculate_requests_per_minute(ip, current_timestamp):
    # Store the current timestamp in Redis
    key = f"requests:{ip}"
    redis_client.zadd(key, {current_timestamp: current_timestamp})
    # Remove timestamps older than 1 minute
    one_minute_ago = current_timestamp - 60
    redis_client.zremrangebyscore(key, '-inf', one_minute_ago)
    # Get the number of requests in the last minute
    requests_count = redis_client.zcard(key)
    return requests_count

def calculate_requests_per_second(ip, current_timestamp):
    # Store the current timestamp in Redis
    key = f"requests:{ip}:second"
    redis_client.zadd(key, {current_timestamp: current_timestamp})
    # Remove timestamps older than 1 second
    one_second_ago = current_timestamp - 1
    redis_client.zremrangebyscore(key, '-inf', one_second_ago)
    # Get the number of requests in the last second
    requests_count = redis_client.zcard(key)
    return requests_count

# Function to process and predict logs
def process_and_predict_log(raw_log):
    # Parse the log line
    parsed_log = parseLine(raw_log)
    
    if not parsed_log:
        print(f"[!] Skipping log: {raw_log}")
        return None

    ip = parsed_log.get('ip')
    timestamp_str = parsed_log.get('date')
    endpoint = parsed_log.get('endpoint')

    # Parse the timestamp manually
    try:
        timestamp = datetime.strptime(timestamp_str, "%d/%b/%Y:%H:%M:%S %z")
        timestamp_unix = timestamp.timestamp()
    except ValueError as e:
        print(f"[!] Skipping log due to timestamp parsing error: {raw_log} - {e}")
        return None

    # Calculate requests per minute
    #requests_per_minute = calculate_requests_per_minute(ip, timestamp_unix)
    requests_per_second = calculate_requests_per_second(ip, timestamp_unix)
    print(f"[?] Requests per second per ip (${ip}): ${requests_per_second}")

    # Preprocess and predict
    prediction = predict_with_model([endpoint], [requests_per_second])
    label = 'anomaly' if prediction[0] == 1 else 'normal'
    
    # Structure the result
    return {
        "timestamp": timestamp_str,
        "raw_log": raw_log,
        "predicted_label": label
    }
# Prepare the event object for sending
def prepare_event_object(row):
    # Prepare the event payload
    event = {
        "timestamp": row["timestamp"],
        "raw_log": row["raw_log"],
        "predicted_label": row["predicted_label"]
    }
    
    # Debugging: Check the event before sending
    print(f"Prepared Event: {event}")
    return event

# Function to send events to a receiver
def send_event(row):
    preparedEventObject = prepare_event_object(row)
    print(preparedEventObject)
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
    counter=0
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
            since_seconds=1 if only_live else None,
            timestamps=False
        ):
            # counter+=1
            # if counter<20:
            if line:
                # Process the raw log
                prediction_result = process_and_predict_log(line)
                if prediction_result:
                    send_event(prediction_result)
            else:
                exit(1)
            # else:
            #     exit(1)
            

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
    pod_name = "my-digital-twin-httpd-server-deployment-85b7bb64f6-f8qxd"
    container_name = None

    # Start streaming logs
    stream_pod_logs(namespace, pod_name, container_name, only_live=args.only_live)
    exit(1)
