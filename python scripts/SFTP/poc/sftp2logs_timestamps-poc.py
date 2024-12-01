import re
import pandas as pd
import redis
import time
from concurrent.futures import ThreadPoolExecutor

# Configurar Redis
redis_db = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Limpiar Redis al inicio
print("Vaciando la base de datos Redis...")
redis_db.flushdb()
print("Base de datos Redis vaciada.")

# Leer logs desde el archivo
with open('../datasets/sftpd2_timestamps.log', 'r') as f:
    log_lines = f.readlines()
    print(f"Total de líneas en el archivo: {len(log_lines)}")

#maximum authentication attempts exceeded for mydigitaltwinuser from 185.209.120.11 port 41489 ssh2 [preauth]
#Disconnecting authenticating user mydigitaltwinuser 185.209.120.11 port 41489: Too many authentication failures [preauth]
# Patrones de log
patterns = {
    "max_auth_requests": r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z) maximum authentication attempts exceeded for ?(?P<user>\w+) from (?P<ip>\d+\.\d+\.\d+\.\d+) port (?P<port>\d+)(?: (?P<log_tail>.*))?",
    "too_many_failures": r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z) Disconnecting authenticating user ?(?P<user>\w+) (?P<ip>\d+\.\d+\.\d+\.\d+) port (?P<port>\d+)(?: (?P<log_tail>.*))?",
    "failed_password_log": r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z) Failed password for (?:invalid )?(?:user )?(?P<user>\w+) from (?P<ip>\d+\.\d+\.\d+\.\d+) port (?P<port>\d+)(?: (?P<log_tail>.*))?",
    "invalid_user_log": r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z) Invalid user (?P<user>\w+) from (?P<ip>\d+\.\d+\.\d+\.\d+) port (?P<port>\d+)(?: (?P<log_tail>.*))?",
    "accepted_password": r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z) Accepted password for (?P<user>\w+) from (?P<ip>\d+\.\d+\.\d+\.\d+) port (?P<port>\d+)(?: (?P<log_tail>.*))?",
    "generic_connection_log": r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z) (?P<log_head>[\w\s]+) (?P<ip>\d+\.\d+\.\d+\.\d+) port (?P<port>\d+)(?: (?P<log_tail>.*))?",
    "general_log": r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z) (?P<log_head>[\w\s]+)",
}

# Procesar logs
structured_logs = []

for log in log_lines:
    log = log.strip()  # Eliminar espacios en blanco
    structured = {"log_type": None, "timestamp": None, "log_head": None, "user": None, "ip": None, "port": None, "log_tail": None, "raw_log": log}
    for log_type, pattern in patterns.items():
        match = re.match(pattern, log)
        if match:
            structured.update(match.groupdict())
            structured["log_type"] = log_type
            break
    structured_logs.append(structured)

# Convertir a DataFrame
df_logs = pd.DataFrame(structured_logs)
# --------------------------------------------------------------------------------original end


# Ruta del nuevo CSV
ftpBruteDataset = '../datasets/10.csv'

# Carga el nuevo CSV
ftpBrute_dataset = pd.read_csv(ftpBruteDataset, dtype=str)

# Mapear columnas relevantes
mapped_data = ftpBrute_dataset.rename(columns={
    "IP Source": "ip",
    "TCP Source Port": "port",
    "Info": "raw_log"
})

# Añadir columnas faltantes y llenar con valores predeterminados
mapped_data["log_type"] = "ftp_brute"
mapped_data["timestamp"] = None
mapped_data["log_head"] = None
mapped_data["user"] = None
mapped_data["log_tail"] = None
# Reordenar columnas para coincidir con df_logs
mapped_data = mapped_data[["log_type", "log_head", "user", "ip", "port", "log_tail", "raw_log"]]

# Combinar los datos
df_logs = pd.concat([df_logs, mapped_data], ignore_index=True)


#--------------------------------------------------------------------------------original processing

# Inicializar la columna de conteo de intentos fallidos
df_logs['failed_password_attempt_count'] = 0
df_logs['ip_attempt_count'] = 0
df_logs['label'] = 'normal'

# Contar intentos fallidos consecutivos por IP y usuario
current_ip_failed = None
failed_count = 0

current_ip_invalid = None
invalid_count = 0

generic_log_count = 0 

max_user_tries_before_suspcious=3
max_password_tries_before_suspcious=3
max_ip_tries_before_suspcious=5
start_time = time.time()

# Procesar después de fusionar los datasets
for index, row in df_logs.iterrows():
    ip = row.get("ip", "unknown")
    user = row.get("user", "unknown")
    log_type = row.get("log_type", "general_log")

    if log_type in ["max_auth_requests"]:
        df_logs.at[index, 'label'] = 'anomaly'

    elif log_type in ["too_many_failures"]:
        df_logs.at[index, 'label'] = 'anomaly'

    elif log_type in ["ftp_brute"]:
        # Incrementar conteos en Redis
        redis_db.hincrby(f"{ip}", "count_ip", 1)
        ip_count = int(redis_db.hget(f"{ip}", "count_ip") or 0)
        if ip_count >= max_ip_tries_before_suspcious:
            df_logs.at[index, 'label'] = 'anomaly'
        else:
            df_logs.at[index, 'label'] = 'normal'
        df_logs.at[index,'ip_attempt_count'] = ip_count
            
    elif log_type in ["invalid_user_log"]:
        redis_db.hincrby(f"{ip}", "count_ip", 1)
        ip_count = int(redis_db.hget(f"{ip}", "count_ip") or 0)
        df_logs.at[index, 'ip_attempt_count'] = ip_count
        # Aplicar reglas de etiquetado
        if ip_count >= max_ip_tries_before_suspcious:
            df_logs.at[index, 'label'] = 'anomaly'
        else:
            df_logs.at[index, 'label'] = 'normal'

    elif log_type in ["failed_password_log"]:
        redis_db.hincrby(f"{ip}:{user}", "count_user_password", 1)
        password_count = int(redis_db.hget(f"{ip}:{user}", "count_user_password") or 0)
        df_logs.at[index, 'failed_password_attempt_count'] = password_count
        # Aplicar reglas de etiquetado
        if password_count >= max_password_tries_before_suspcious:
            df_logs.at[index, 'label'] = 'anomaly'
        else:
            df_logs.at[index, 'label'] = 'normal'
        df_logs.at[index, 'failed_password_attempt_count'] = password_count
        df_logs.at[index, 'ip_attempt_count'] = ip_count
    elif log_type == "accepted_password":
        # Recuperar conteos actuales
        ip_count = int(redis_db.hget(f"{ip}", "count_ip") or 0)
        password_count = int(redis_db.hget(f"{ip}:{user}", "count_user_password") or 0)
        df_logs.at[index, 'ip_attempt_count'] = ip_count
        # Aplicar reglas de etiquetado
        if ip_count >= max_ip_tries_before_suspcious:
            df_logs.at[index, 'label'] = 'anomaly'
        else:
            df_logs.at[index, 'label'] = 'normal'           
        redis_db.hdel(f"{ip}:{user}", "count_user") # Reiniciar conteos después de una contraseña aceptada

# Mostrar resultados
print(f"Tiempo: {time.time() - start_time} segundos")
print(df_logs)

# Guardar en CSV
df_logs.to_csv('../processed_datasets/structured_sftp_logs_with_counts_timestamps2.csv', index=False)
