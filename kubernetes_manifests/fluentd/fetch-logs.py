import requests

# Configura la IP y el puerto del servicio Fluentd
FLUENTD_HOST = "34.65.84.23"  # Reemplaza con la IP externa del LoadBalancer
FLUENTD_PORT = 9880  # Puerto configurado en Fluentd

def fetch_logs():
    url = f"http://{FLUENTD_HOST}:{FLUENTD_PORT}"
    
    try:
        response = requests.get(url, stream=True)  # Usa `stream=True` para logs en tiempo real
        if response.status_code == 200:
            print("Conexión exitosa. Logs:")
            for line in response.iter_lines():
                if line:  # Ignora líneas vacías
                    print(line.decode('utf-8'))
        else:
            print(f"Error al conectar con Fluentd: {response.status_code} - {response.reason}")
    except requests.exceptions.RequestException as e:
        print(f"Error al conectar con Fluentd: {e}")

if __name__ == "__main__":
    fetch_logs()
