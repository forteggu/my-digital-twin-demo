const express = require('express');
const bodyParser = require('body-parser');

const app = express();
const PORT = 8000;

// Base de datos en memoria para almacenar eventos
let httpdEvents = [];
let msLogsEvents = [];
let sftpdEvents = [];

// Middleware para parsear JSON
app.use(bodyParser.json());

const MAX_LOGS = 5000; // Limitar a 5000 logs por fuente

function handleEvent(req, res, eventsArray) {
    const event = req.body;
    console.log(event);
    if (event.timestamp && event.raw_log && event.predicted_label) {
        // Agregar evento al arreglo
        eventsArray.push(event);

        // Si excede el límite, eliminar el más antiguo
        if (eventsArray.length > MAX_LOGS) {
            eventsArray.shift(); // Elimina el primer elemento (más antiguo)
        }

        console.log('Evento recibido:', event);
        res.status(200).json({ status: 'received', event });
    } else {
        console.log("invalid event format");
        res.status(400).json({ status: 'error', message: 'Invalid event format' });
    }
}

// Endpoints para los diferentes tipos de eventos
app.post('/log-httpd-event', (req, res) => handleEvent(req, res, httpdEvents));
app.post('/log-ms-event', (req, res) => handleEvent(req, res, msLogsEvents));
app.post('/log-sftpd-event', (req, res) => handleEvent(req, res, sftpdEvents));

// Endpoints para obtener eventos
app.get('/events/httpd', (req, res) => res.status(200).json(httpdEvents));
app.get('/events/ms-vulnerable', (req, res) => res.status(200).json(msLogsEvents));
app.get('/events/sftpd', (req, res) => res.status(200).json(sftpdEvents));

// Endpoints para borrar eventos
app.delete('/events/httpd', (req, res) => {
    httpdEvents = [];
    console.log('Todos los logs de HTTPD han sido eliminados.');
    res.status(200).json({ status: 'success', message: 'Todos los logs de HTTPD han sido eliminados.' });
});

app.delete('/events/ms-vulnerable', (req, res) => {
    msLogsEvents = [];
    console.log('Todos los logs de MS Logs han sido eliminados.');
    res.status(200).json({ status: 'success', message: 'Todos los logs de MS Logs han sido eliminados.' });
});

app.delete('/events/sftpd', (req, res) => {
    sftpdEvents = [];
    console.log('Todos los logs de SFTPD han sido eliminados.');
    res.status(200).json({ status: 'success', message: 'Todos los logs de SFTPD han sido eliminados.' });
});

// Iniciar el servidor
app.listen(PORT, () => {
    console.log(`Servidor ejecutándose en http://localhost:${PORT}`);
});
