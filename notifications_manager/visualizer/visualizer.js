const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const axios = require('axios');

const app = express();
const server = http.createServer(app);
const io = socketIo(server);

const PORT = 3000;
const EVENT_API_URL = 'http://34.65.255.107:8000/events'; // URL del receptor de eventos
//const EVENT_API_URL = 'http://localhost:8000/events'; // URL del receptor de eventos

// Servir el archivo index.html directamente desde el directorio actual
app.get('/', (req, res) => {
    res.sendFile(__dirname + '/index.html');
});

// Obtener eventos periódicamente y enviarlos a los clientes
setInterval(async () => {
    try {
        const response = await axios.get(EVENT_API_URL);
        io.emit('events', response.data); // Emitir eventos a todos los clientes conectados
    } catch (error) {
        console.error('Error obteniendo eventos:', error.message);
    }
}, 3000); // Consultar eventos cada 3 segundos

// Iniciar el servidor
server.listen(PORT, () => {
    console.log(`Visualizador ejecutándose en http://localhost:${PORT}`);
});
