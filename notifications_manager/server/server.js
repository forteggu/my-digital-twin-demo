const express = require('express');
const bodyParser = require('body-parser');

const app = express();
const PORT = 8000;

// Base de datos en memoria para almacenar eventos
let events = [];

// Middleware para parsear JSON
app.use(bodyParser.json());

// Endpoint para recibir eventos
app.post('/log-event', (req, res) => {
    const event = req.body;
    if (event.raw_log && event.predicted_label) {
        events.push(event);
        console.log('Evento recibido:', event);
        res.status(200).json({ status: 'received', event });
    } else {
        res.status(400).json({ status: 'error', message: 'Invalid event format' });
    }
});

// Endpoint para obtener todos los eventos
app.get('/events', (req, res) => {
    res.status(200).json(events);
});

// Iniciar el servidor
app.listen(PORT, () => {
    console.log(`Servidor ejecutándose en http://localhost:${PORT}`);
});
