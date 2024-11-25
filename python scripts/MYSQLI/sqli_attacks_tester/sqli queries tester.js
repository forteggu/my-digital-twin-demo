const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const fs = require('fs');
const cors = require('cors');

const app = express();
const port = 3000;

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

// Leer y procesar las consultas maliciosas desde un archivo
const readMaliciousQueries = (filePath) => {
  const fileContent = fs.readFileSync(filePath, 'utf8');
  const lines = fileContent.split('\n');
  return lines
    .filter((line) => line.startsWith('Ejecutando consulta:')) // Filtrar líneas válidas
    .map((line) => line.replace('Ejecutando consulta: ', '').trim()); // Remover el prefijo y limpiar
};

// Cargar consultas desde el archivo
const maliciousQueries = readMaliciousQueries('../datasets/originals/final_injection_queries_with_prefix.csv');

// Función para ejecutar las consultas maliciosas
const runMaliciousQueries = () => {
  let successfulQueries = [];
  let failedQueries = [];

  maliciousQueries.forEach((query) => {
    db.all(query, (err, rows) => {
      if (err) {
        failedQueries.push(query);
      } else {
        successfulQueries.push(query);
      }

      // Log resultados
      if (successfulQueries.length + failedQueries.length === maliciousQueries.length) {
        console.log('Malicious query execution complete:');
        console.log(`Successful queries: ${successfulQueries.length}`);
        console.log(`Failed queries: ${failedQueries.length}`);
        if (successfulQueries.length > 0) {
          console.log('Successful queries:', successfulQueries);
        }
        if (failedQueries.length > 0) {
          console.log('Failed queries:', failedQueries);
        }
      }
    });
  });
};

// Ejecutar las consultas maliciosas después de inicializar el servidor
app.listen(port, () => {
  console.log(`Microservice running on port ${port}`);
  runMaliciousQueries();
});
