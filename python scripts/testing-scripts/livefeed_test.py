from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException

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
            container=container_name,
            follow=True
        ):
            # Las líneas ya son cadenas de texto, no necesitan decodificación
            print(line.strip())

    except ApiException as e:
        print(f"Error al obtener logs en streaming: {e}")

if __name__ == "__main__":
    # Define el namespace y el nombre del pod
    namespace = "default"  # Cambia según tu caso
    pod_name = "my-digital-twin-sftp-server-deployment-5f98586588-szq7d"  # Cambia según tu caso
    container_name = None  # Especifica el contenedor si hay múltiples

    # Obtener y mostrar los logs anteriores
    print("Logs anteriores del Pod:")
    logs = get_pod_logs(namespace, pod_name, container_name)
    if logs:
        print(logs)

    # Habilitar el streaming en vivo de logs
    stream_pod_logs(namespace, pod_name, container_name)
