const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const axios = require('axios');

const app = express();
const server = http.createServer(app);
const io = socketIo(server);

const PORT = 3000;
const EVENT_API_URLS = {
    httpd: 'http://34.65.255.107:8000/events/httpd',
    msLogs: 'http://34.65.255.107:8000/events/ms-vulnerable',
    sftpd: 'http://34.65.255.107:8000/events/sftpd',
};

// Serve the index.html file directly from the current directory
app.get('/', (req, res) => {
    res.sendFile(__dirname + '/index.html');
});

let selectedSource = 'httpd'; // Default selected source
let intervalId;

// Listen for changes in the selected source
io.on('connection', (socket) => {
    // When a new client connects, send the current source events
    fetchAndEmitEvents();

    socket.on('changeSource', (source) => {
        console.log(`Source changed to: ${source}`);
        selectedSource = source;
        fetchAndEmitEvents(); // Fetch events immediately for the new source

        // Clear the previous interval and start a new one
        if (intervalId) {
            clearInterval(intervalId);
        }
        intervalId = setInterval(fetchAndEmitEvents, 3000);
    });
});

// Function to fetch and emit events
async function fetchAndEmitEvents() {
    try {
        const response = await axios.get(EVENT_API_URLS[selectedSource]);
        io.emit('events', response.data); // Emit events to all connected clients
    } catch (error) {
        console.error('Error fetching events:', error.message);
    }
}

// Start fetching events at the default interval
intervalId = setInterval(fetchAndEmitEvents, 3000);

// Start the server
server.listen(PORT, () => {
    console.log(`Log viewer running at http://localhost:${PORT}`);
});
