# Estructura del Proyecto

## Raíz del Proyecto
- **`docker-compose.yml`**  
  Archivo que contiene las imágenes y configuración de los diferentes contenedores Docker que se van a crear.

- **`ftp_data`**  
  Datos simulados para el servicio FTP.
  - `FTP_DATA_EXAMPLE.txt`: Archivo de prueba.

- **`sftp_data`**  
  Datos simulados para el servicio SFTP.
  - `SFTP_DATA_EXAMPLE`: Archivo de prueba.

- **`wordpress_data`**  
  Datos para el servicio WooCommerce.

- **`db_data`**  
  Datos de la base de datos para WooCommerce.

- **`product-service`**  
  Código y configuración del microservicio de ejemplo.
  - `Dockerfile`: Archivo de configuración para construir la imagen del contenedor.
  - `server.js`: Código fuente del microservicio.

- **`python_scripts`**  
  Contiene los scripts en Python y PowerShell para el procesamiento de datasets y la creación de modelos de IA.
  - **`Datasets`**: Carpeta que contiene los datasets utilizados.
  - **`model-tester / testing scripts`**: Scripts para probar las predicciones de los modelos generados.
  - **`model-trainer`**: Carpeta que contiene los scripts para generar los modelos de IA.
  - **`models`**: Carpeta de salida con los modelos generados:
    - Archivos `.h5` para modelos RNN.
    - Archivos `.pkl` para modelos Random Forest.
    - Tokenizers generados.
  - **`preprocessors`**: Scripts para el preprocesamiento de los datasets.
  - **`SERVICE_live_analyzer_version.py`**: Script para realizar predicciones de los logs existentes en Kubernetes.  
    **Nota:** Los nombres de los contenedores están hardcodeados.
