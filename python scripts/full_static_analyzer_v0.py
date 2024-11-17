import numpy as np
#from tensorflow.keras.models import load_model
import subprocess
from sklearn.preprocessing import LabelEncoder
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException
import re
import pandas as pd
import redis
import time
from concurrent.futures import ThreadPoolExecutor

#Initialización entorno y variables
redis_db = redis.Redis(host='localhost', port=6379, decode_responses=True)

## Limpiar Redis al inicio
print("Vaciando la base de datos Redis...")
redis_db.flushdb()
print("Base de datos Redis vaciada.")

# Cargar el modelo
model_path = 'models/sftp_anomaly_detector.h5'
model = load_model(model_path)

# Características necesarias para el modelo
features = ["failed_password_attempt_count", "ip_attempt_count", "log_type_encoded"]

# Patrones de log
patterns = {
    "failed_password_log": r"Failed password for (?:invalid )?(?:user )?(?P<user>\w+) from (?P<ip>\d+\.\d+\.\d+\.\d+) port (?P<port>\d+)(?: (?P<log_tail>.*))?",
    "invalid_user_log": r"Invalid user (?P<user>\w+) from (?P<ip>\d+\.\d+\.\d+\.\d+) port (?P<port>\d+)(?: (?P<log_tail>.*))?",
    "accepted_password": r"Accepted password for (?P<user>\w+) from (?P<ip>\d+\.\d+\.\d+\.\d+) port (?P<port>\d+)(?: (?P<log_tail>.*))?",
    "generic_connection_log": r"(?P<log_head>[\w\s]+) (?P<ip>\d+\.\d+\.\d+\.\d+) port (?P<port>\d+)(?: (?P<log_tail>.*))?",
    "general_log": r"(?P<log_head>[\w\s]+)",
}

# Almacen de logs
structured_logs = []


def get_pod_logs(namespace, pod_name, container_name=None):
    try:
        # Cargar la configuración de Kubernetes desde ~/.kube/config
        config.load_kube_config()

        # Crear instancia del cliente CoreV1Api
        v1 = client.CoreV1Api()

        # Obtener todos los logs anteriores del pod
        logs = v1.read_namespaced_pod_log(
            name=pod_name,
            namespace=namespace,
            container=container_name,
            follow=False  # Obtener logs estáticos (anteriores)
        )
        return logs

    except ApiException as e:
        print(f"Error al obtener logs estáticos: {e}")
        return None

def stream_pod_logs(namespace, pod_name, container_name=None):
    try:
        # Cargar la configuración de Kubernetes desde ~/.kube/config
        config.load_kube_config()

        # Crear instancia del cliente CoreV1Api
        v1 = client.CoreV1Api()

        # Inicializar watch para streaming de logs
        w = watch.Watch()
        print(f"Streaming logs en tiempo real del pod {pod_name} en el namespace {namespace}...")

        # Iniciar el stream de logs en vivo
        for line in w.stream(
            v1.read_namespaced_pod_log,
            name=pod_name,
            namespace=namespace,
            container=container_name
        ):
            # Las líneas ya son cadenas de texto, no necesitan decodificación
           lineDataFrame=structureLine(line)
           processLineDataFrame(lineDataFrame)



    except ApiException as e:
        print(f"Error al obtener logs en streaming: {e}")


# Leer logs desde el archivo
with open('./datasets/sftp.log', 'r') as f:
    log_lines = f.readlines()
    print(f"Total de líneas en el archivo: {len(log_lines)}")


def structureLine(line):
        line = line.strip()  # Eliminar espacios en blanco
        structured = {"log_type": None, "log_head": None, "user": None, "ip": None, "port": None, "log_tail": None, "raw_log": line}
        for log_type, pattern in patterns.items():
            match = re.match(pattern, line)
            if match:
                structured.update(match.groupdict())
                structured["log_type"] = log_type
                break
        # Convertir a DataFrame
        return pd.DataFrame([structured])

def processLineDataFrame(lineDataFrame):
    # Inicializar la columna de conteo de intentos fallidos
    lineDataFrame['failed_password_attempt_count'] = 0
    lineDataFrame['ip_attempt_count'] = 0

    # Procesar después de fusionar los datasets
    for index, row in lineDataFrame.iterrows():
        ip = row.get("ip", "unknown")
        user = row.get("user", "unknown")
        log_type = row.get("log_type", "general_log")

        if log_type in ["ftp_brute"]:
            # Incrementar conteos en Redis
            redis_db.hincrby(f"{ip}", "count_ip", 1)
            ip_count = int(redis_db.hget(f"{ip}", "count_ip") or 0)
            lineDataFrame.at[index,'ip_attempt_count'] = ip_count
                
        elif log_type in ["invalid_user_log"]:
            redis_db.hincrby(f"{ip}", "count_ip", 1)
            ip_count = int(redis_db.hget(f"{ip}", "count_ip") or 0)
            lineDataFrame.at[index, 'ip_attempt_count'] = ip_count

        elif log_type in ["failed_password_log"]:
            redis_db.hincrby(f"{ip}:{user}", "count_user_password", 1)
            ip_count = int(redis_db.hget(f"{ip}", "count_ip") or 0)
            password_count = int(redis_db.hget(f"{ip}:{user}", "count_user_password") or 0)
            lineDataFrame.at[index, 'failed_password_attempt_count'] = password_count
            lineDataFrame.at[index, 'ip_attempt_count'] = ip_count
        elif log_type == "accepted_password":
            # Recuperar conteos actuales
            ip_count = int(redis_db.hget(f"{ip}", "count_ip") or 0)
            password_count = int(redis_db.hget(f"{ip}:{user}", "count_user_password") or 0)
            lineDataFrame.at[index, 'ip_attempt_count'] = ip_count
            redis_db.hdel(f"{ip}:{user}", "count_user") # Reiniciar conteos después de una contraseña aceptada

    # Mostrar resultados
    print(lineDataFrame)




if __name__ == "__main__":
    # Define el namespace y el nombre del pod
    namespace = "default"  # Cambia según tu caso
    pod_name = "my-digital-twin-sftp-server-deployment-5f98586588-szq7d"  # Cambia según tu caso
    container_name = None  # Especifica el contenedor si hay múltiples

    # Obtener y mostrar los logs anteriores
    # print("Logs anteriores del Pod:")
    # logs = get_pod_logs(namespace, pod_name, container_name)
    # if logs:
    #     print(logs)

    # Habilitar el streaming en vivo de logs
    stream_pod_logs(namespace, pod_name, container_name)


