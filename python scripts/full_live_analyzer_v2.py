import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException
from sklearn.preprocessing import LabelEncoder
import redis
import re
import requests
import argparse

#EVENT_RECEIVER_URL = "http://localhost:8000/log-event/"
EVENT_RECEIVER_URL = "http://34.65.255.107:8000/log-event/"

# Inicializar entorno y variables
redis_db = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Limpiar Redis al inicio
print("Vaciando la base de datos Redis...")
redis_db.flushdb()
print("Base de datos Redis vaciada.")

# Patrones de log
patterns = {
    "failed_password_log": r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z) Failed password for (?:invalid )?(?:user )?(?P<user>\w+) from (?P<ip>\d+\.\d+\.\d+\.\d+) port (?P<port>\d+)(?: (?P<log_tail>.*))?",
    "invalid_user_log": r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z) Invalid user (?P<user>\w+) from (?P<ip>\d+\.\d+\.\d+\.\d+) port (?P<port>\d+)(?: (?P<log_tail>.*))?",
    "accepted_password": r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z) Accepted password for (?P<user>\w+) from (?P<ip>\d+\.\d+\.\d+\.\d+) port (?P<port>\d+)(?: (?P<log_tail>.*))?",
    "generic_connection_log": r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z) (?P<log_head>[\w\s]+) (?P<ip>\d+\.\d+\.\d+\.\d+) port (?P<port>\d+)(?: (?P<log_tail>.*))?",
    "general_log": r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z) (?P<log_head>[\w\s]+)",
}

# Cargar el modelo
model_path = 'models/sftp_anomaly_detector.h5'
model = load_model(model_path)

# Características necesarias para el modelo
features = ["failed_password_attempt_count", "ip_attempt_count", "log_type_encoded"]


def send_event(row):
    raw_log = row["raw_log"]
    timestamp = row["timestamp"]
    predicted_label = row["predicted_label"]
    log_type = row["log_type"]
    user = row["user"]
    ip = row["ip"]
    
    if isinstance(raw_log, str) and isinstance(timestamp, str) and raw_log.startswith(timestamp):
        raw_log = raw_log[len(timestamp):].strip()  # Quitar el timestamp del inicio del raw_log


    """Enviar un evento al receptor"""
    event = {
        "timestamp": timestamp,
        "raw_log": raw_log,
        "predicted_label": predicted_label
    }
    try:
        response = requests.post(EVENT_RECEIVER_URL, json=event)
        if response.status_code == 200:
            print(f"Evento enviado: {event}")
        else:
            print(f"Error al enviar el evento: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Error al conectar con el receptor: {e}")



# Función para estructurar una línea de log
def structure_line(line):
    line = line.strip()
    structured = {"log_type": None, "timestamp": None, "log_head": None, "user": None, "ip": None, "port": None, "log_tail": None, "raw_log": line}
    for log_type, pattern in patterns.items():
        match = re.match(pattern, line)
        if match:
            structured.update(match.groupdict())
            structured["log_type"] = log_type
            break
    return pd.DataFrame([structured])

# Función para procesar DataFrame de logs
def process_line_dataframe(line_df):
    # Inicializar las columnas requeridas
    line_df['failed_password_attempt_count'] = 0
    line_df['ip_attempt_count'] = 0

    # Procesar cada línea
    for index, row in line_df.iterrows():
        ip = row.get("ip", "unknown")
        user = row.get("user", "unknown")
        log_type = row.get("log_type", "general_log")

        if log_type in ["invalid_user_log", "failed_password_log"]:
            redis_db.hincrby(f"{ip}", "count_ip", 1)
            ip_count = int(redis_db.hget(f"{ip}", "count_ip") or 0)
            line_df.at[index, 'ip_attempt_count'] = ip_count

        if log_type == "failed_password_log":
            redis_db.hincrby(f"{ip}:{user}", "count_user_password", 1)
            password_count = int(redis_db.hget(f"{ip}:{user}", "count_user_password") or 0)
            line_df.at[index, 'failed_password_attempt_count'] = password_count

        elif log_type == "accepted_password":
            redis_db.hdel(f"{ip}:{user}", "count_user_password")

    # Generar log_type_encoded
    if "log_type_encoded" not in line_df.columns:
        label_encoder = LabelEncoder()
        line_df['log_type_encoded'] = label_encoder.fit_transform(line_df['log_type'].astype(str))

    # Rellenar valores NaN si los hay
    line_df = line_df.fillna(0)

    return line_df[features]

# Función para predecir con el modelo
def predict_with_model(features_df):
    predictions = model.predict(features_df)
    predictions = (predictions.flatten() > 0.5).astype(int)
    return predictions

# Función para manejar logs en streaming
def stream_pod_logs(namespace, pod_name, container_name=None,only_live=False):
    try:
        # Cargar configuración de Kubernetes
        config.load_kube_config()

        # Crear cliente de CoreV1Api
        v1 = client.CoreV1Api()

        # Inicializar watch para logs en vivo
        w = watch.Watch()
        print(f"Streaming logs en tiempo real del pod {pod_name} en el namespace {namespace}...")

        for line in w.stream(
            v1.read_namespaced_pod_log,
            name=pod_name,
            namespace=namespace,
            container=container_name,
            follow=True,
            since_seconds=1 if only_live else None,
            timestamps=True  # Incluye timestamps en los logs
        ):
            # Estructurar y procesar la línea
            structured_df = structure_line(line)
            features_df = process_line_dataframe(structured_df)

            # Hacer predicción
            predictions = predict_with_model(features_df)
            structured_df['predicted_label'] = predictions
            structured_df['predicted_label'] = structured_df['predicted_label'].map({0: 'normal', 1: 'anomaly'})

            # Mostrar resultados
            print(structured_df[['raw_log', 'predicted_label']])
            # Enviar el evento al receptor si es anomalia
            for _, row in structured_df.iterrows():
#               if row['predicted_label'] == 'anomaly':
                send_event(row)


    except ApiException as e:
        print(f"Error al obtener logs en streaming: {e}")

# Main
if __name__ == "__main__":

        # Configurar argumentos del script
    parser = argparse.ArgumentParser(description="Stream de logs de un pod en Kubernetes")
    parser.add_argument("--only-live", action="store_true", help="Si se establece, solo obtiene logs en vivo")

    args = parser.parse_args()


    namespace = "default"
    pod_name = "my-digital-twin-sftp-server-deployment-5f98586588-szq7d"
    container_name = None

    # Habilitar streaming en vivo de logs
    stream_pod_logs(namespace, pod_name, container_name, only_live=args.only_live)
