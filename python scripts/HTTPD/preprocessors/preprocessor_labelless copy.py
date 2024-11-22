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
with open('../datasets/http.log', 'r') as f:
    print(f"Total de líneas en el archivo: {len(log_lines)}")
    log_lines = f.readlines()
    print(f"Total de líneas en el archivo: {len(log_lines)}")

# Patrón para dividir los logs HTTP
pattern = r'^(?P<ip>\d+\.\d+\.\d+\.\d+) - - \[(?P<timestamp>[^\]]+)\] "(?P<method>\w+) (?P<request>[^\s]+) (?P<protocol>HTTP\/\d\.\d)" (?P<response_code>\d+) (?P<response_body_size>[\d\-]+)'

# Procesar logs
structured_logs = []
for log in log_lines:
    match = re.match(pattern, log.strip())
    if match:
        structured_logs.append(match.groupdict())

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

    if log_type in ["ftp_brute"]:
        # Incrementar conteos en Redis
        redis_db.hincrby(f"{ip}", "count_ip", 1)
        ip_count = int(redis_db.hget(f"{ip}", "count_ip") or 0)
        df_logs.at[index,'ip_attempt_count'] = ip_count
            
    elif log_type in ["invalid_user_log"]:
        redis_db.hincrby(f"{ip}", "count_ip", 1)
        ip_count = int(redis_db.hget(f"{ip}", "count_ip") or 0)
        df_logs.at[index, 'ip_attempt_count'] = ip_count

    elif log_type in ["failed_password_log"]:
        redis_db.hincrby(f"{ip}:{user}", "count_user_password", 1)
        password_count = int(redis_db.hget(f"{ip}:{user}", "count_user_password") or 0)
        df_logs.at[index, 'failed_password_attempt_count'] = password_count
        df_logs.at[index, 'ip_attempt_count'] = ip_count
    elif log_type == "accepted_password":
        # Recuperar conteos actuales
        ip_count = int(redis_db.hget(f"{ip}", "count_ip") or 0)
        password_count = int(redis_db.hget(f"{ip}:{user}", "count_user_password") or 0)
        df_logs.at[index, 'ip_attempt_count'] = ip_count
        redis_db.hdel(f"{ip}:{user}", "count_user") # Reiniciar conteos después de una contraseña aceptada

# Mostrar resultados
print(f"Tiempo: {time.time() - start_time} segundos")
print(df_logs)

# Guardar en CSV
df_logs.to_csv('processed_datasets/labelless_structured_sftp_logs_with_counts.csv', index=False)
