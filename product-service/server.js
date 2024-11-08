const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const app = express();
const port = 3000;

// Conexión a la base de datos en memoria
const db = new sqlite3.Database(':memory:');

// Crear una tabla y agregar datos de ejemplo
db.serialize(() => {
  db.run("CREATE TABLE users (id INT, name TEXT)");
  db.run("INSERT INTO users (id, name) VALUES (1, 'Alice')");
  db.run("INSERT INTO users (id, name) VALUES (2, 'Bob')");
});

// Endpoint vulnerable a inyección SQL
app.get('/user', (req, res) => {
  const userId = req.query.id;

  // Consulta vulnerable a inyección SQL
  const query = `SELECT * FROM users WHERE id = ${userId}`;
  console.log(`Ejecutando consulta: ${query}`);

  db.all(query, (err, rows) => {
    if (err) {
      res.status(500).send("Error en la consulta");
      return;
    }
    res.json(rows);
  });
});

app.listen(port, () => {
  console.log(`Microservice running on port ${port}`);
});