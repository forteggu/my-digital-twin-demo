# Proyecto My-Digital-Twin - Entregable Final

## Estructura:

●	Raiz
    ○	docker-compose.yml: Fichero que contiene las imágenes y configuración de los diferentes contenedores docker que se van a crear.
    ○	ftp_data: Datos simulados para el servicio FTP
        ■	FTP_DATA_EXAMPLE.txt: fichero de prueba
    ○	sftp_data: Datos simulados para el servicio SFTP
        ■	SFTP_DATA_EXAMPLE: Fichero de prueba
    ○	wordpress_data: Datos para WooCommerce
    ○	db_data: Datos de la base de datos para WooCommerce
    ○	product-service:  Código y Dockerfile de microservicio de ejemplo
        ■	Dockerfile
        ■	server.js
    ○	python scripts:  Contiene los scripts en python y powershell para el procesamiento de los datasets y la creación de los modelos de IA
         ■	Datasets: datasets
         ■	model-tester / testing scripts: scripts para probar las predicciones de los modelos generados
         ■	model-trainer: contiene los scripts para generar los modelos de IA
         ■	models: output de los modelos (.h5 para RNN, .pkl para random forest y tokenizers) 
         ■	preprocessors: preprocesamiento de los datasets
         ■	SERVICE_live_analyzer_version.py: script a ejecutar para realizar las predicciones de los logs existentes en kubernetes. Los nombres de los contenedores están harcodeados 




#
