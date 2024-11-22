# Usa una imagen base de Node.js
FROM node:16

# Crea el directorio de trabajo
WORKDIR /app

# Instala las dependencias dentro del contenedor
RUN npm install sqlite3
RUN npm install express
RUN npm install cors

# Copia el resto del código de la aplicación
COPY server.js .

# Expone el puerto en el que corre el servicio
EXPOSE 3000

# Ejecuta el servicio
CMD ["node", "server.js"]
