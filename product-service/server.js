const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const app = express();
const port = 3000;
const cors = require('cors'); // Importa CORS
// Conexión a la base de datos en memoria
const db = new sqlite3.Database(':memory:');
app.use(cors());
// Crear una tabla y agregar datos de ejemplo
db.serialize(() => {
  db.run("CREATE TABLE users (id INT, name TEXT)");
  db.run("INSERT INTO users (id, name) VALUES (1, 'admin')");
  db.run("INSERT INTO users (id, name) VALUES (2, 'Alice')");
  db.run("INSERT INTO users (id, name) VALUES (3, 'Bob')");
  db.run("INSERT INTO users (id, name) VALUES (4, 'Bobber')");
  db.run("INSERT INTO users (id, name) VALUES (5, 'Kurwa')");
  db.run("INSERT INTO users (id, name) VALUES (6, 'BobberKurwa')");
});

// Endpoint vulnerable a inyección SQL
app.get('/user', (req, res) => {
  const userName = req.query.name;
  
  // Consulta vulnerable a inyección SQL
  const query = `SELECT * FROM users WHERE name = '${userName}'`;
  console.log(`Ejecutando consulta: ${query}`);
  db.all(query, (err, rows) => {
    if (err) {
      res.status(500).send({error:"Error en la consulta"});
      return;
    }
    res.json(rows);
  });
});

app.listen(port, () => {
  console.log(`Microservice running on port ${port}`);
});