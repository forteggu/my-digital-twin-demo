import re
import pandas as pd

# Leer logs desde el archivo
with open('./datasets/sftp.log', 'r') as f:
    log_lines = f.readlines()
    print(f"Total de líneas en el archivo: {len(log_lines)}")

# Patrones de log
patterns = {
    "failed_password_log": r"Failed password for (?:invalid )?(?:user )?(?P<user>\w+) from (?P<ip>\d+\.\d+\.\d+\.\d+) port (?P<port>\d+)(?: (?P<log_tail>.*))?",
    "invalid_user_log": r"Invalid user (?P<user>\w+) from (?P<ip>\d+\.\d+\.\d+\.\d+) port (?P<port>\d+)(?: (?P<log_tail>.*))?",
    "generic_connection_log": r"(?P<log_head>[\w\s]+) (?P<ip>\d+\.\d+\.\d+\.\d+) port (?P<port>\d+)(?: (?P<log_tail>.*))?",
    "general_log": r"(?P<log_head>[\w\s]+)",
}

# Procesar logs
structured_logs = []

for log in log_lines:
    log = log.strip()  # Eliminar espacios en blanco
    structured = {"log_type": None, "log_head": None, "user": None, "ip": None, "port": None, "log_tail": None, "raw_log": log}
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
ftpBruteDataset = './datasets/10.csv'

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




#TODO: change counter to failed_attempt_count so that it is taken into account for the failed attempts counting


#--------------------------------------------------------------------------------original processing

# Inicializar la columna de conteo de intentos fallidos
df_logs['failed_attempt_count'] = 0
df_logs['invalid_user_count'] = 0
df_logs['label'] = 'normal'

# Contar intentos fallidos consecutivos por IP y usuario
current_ip_failed = None
failed_count = 0

current_ip_invalid = None
invalid_count = 0

generic_log_count = 0 

for index, row in df_logs.iterrows():
    if row['log_type'] == "failed_password_log":
        if row['ip'] == current_ip_failed:
            failed_count += 1
            if failed_count >= 3:
                df_logs.at[index, 'label'] = 'anomaly'

        else:
            current_ip_failed = row['ip']
            failed_count = 1
        df_logs.at[index, 'failed_attempt_count'] = failed_count

    elif row['log_type'] == "invalid_user_log":
        if row['ip'] == current_ip_invalid:
            invalid_count += 1
            if invalid_count >= 3:
                df_logs.at[index, 'label'] = 'anomaly'
        else:
            current_ip_invalid = row['ip']
            invalid_count = 1
        df_logs.at[index, 'invalid_user_count'] = invalid_count

    elif row['log_type'] == "ftp_brute":
        if row['ip'] == current_ip_invalid:
            invalid_count += 1
            if invalid_count >= 3:
                df_logs.at[index, 'label'] = 'anomaly'
        else:
            current_ip_invalid = row['ip']
            invalid_count = 1
        df_logs.at[index, 'failed_attempt_count'] = invalid_count

    else:
        # Reiniciar contadores para otros tipos de logs
        # current_ip_failed = None
        # failed_count = 0
        # current_ip_invalid = None
        # invalid_count = 0
        generic_log_count+=1
        

# Mostrar resultados
print(df_logs)

# Guardar en CSV
df_logs.to_csv('structured_sftp_logs_with_counts.csv', index=False)
