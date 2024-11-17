const express = require('express');
const bodyParser = require('body-parser');

const app = express();
const PORT = 8000;

// Base de datos en memoria para almacenar eventos
let events = [];

// Middleware para parsear JSON
app.use(bodyParser.json());

const MAX_LOGS = 5000; // Limitar a 1000 logs

app.post('/log-event', (req, res) => {
    const event = req.body;

    if (event.timestamp && event.raw_log && event.predicted_label) {
        // Agregar evento al arreglo
        events.push(event);

        // Si excede el límite, eliminar el más antiguo
        if (events.length > MAX_LOGS) {
            events.shift(); // Elimina el primer elemento (más antiguo)
        }

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
