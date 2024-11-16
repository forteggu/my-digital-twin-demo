import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configuración inicial
data_path = './processed_datasets/structured_sftp_logs_with_counts.csv'
model_save_path = 'models/sftp_anomaly_detector.h5'

# 1. Cargar y Preprocesar Datos
print("Cargando datos...")
data = pd.read_csv(data_path)

# Generar 'log_type_encoded' si no existe
if 'log_type_encoded' not in data.columns:
    print("Generando 'log_type_encoded'...")
    label_encoder = LabelEncoder()
    data['log_type_encoded'] = label_encoder.fit_transform(data['log_type'].astype(str))
    print("Valores únicos de log_type y sus codificaciones:")
    print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# Seleccionar columnas relevantes
features = ["failed_password_attempt_count", "ip_attempt_count", "log_type_encoded"]
target = "label"

# Codificar etiquetas (normal=0, anomaly=1)
data[target] = data[target].map({'normal': 0, 'anomaly': 1})

# Separar características y etiquetas
X = data[features]
y = data[target]

# Manejar valores NaN
X = X.fillna(0)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Guardar conjuntos preprocesados para reutilizar
X_train.to_csv('./processed_datasets/X_train.csv', index=False)
X_test.to_csv('./processed_datasets/X_test.csv', index=False)
y_train.to_csv('./processed_datasets/y_train.csv', index=False)
y_test.to_csv('./processed_datasets/y_test.csv', index=False)

print("Datos preprocesados y guardados.")

# 2. Crear el Modelo
print("Creando modelo...")
model = Sequential([
    Dense(16, activation='relu', input_shape=(len(features),)),  # Número de características
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')  # Salida binaria (0=normal, 1=anomaly)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 3. Entrenar el Modelo
print("Entrenando modelo...")
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Guardar el modelo
model.save(model_save_path)
print(f"Modelo guardado en {model_save_path}")

# 4. Evaluar el Modelo
print("Evaluando modelo...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Pérdida en prueba: {loss:.4f}, Precisión en prueba: {accuracy:.4f}")

# 5. Visualizar el Entrenamiento
print("Visualizando métricas...")
# Graficar pérdida
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Pérdida Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida Validación')
plt.legend()
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')

# Graficar precisión
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Precisión Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión Validación')
plt.legend()
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')

plt.tight_layout()
plt.show()

# 6. Guardar y Predecir con el Modelo
print("Haciendo predicciones con datos de prueba...")
predictions = model.predict(X_test)
threshold = 0.5  # Umbral para clasificación
y_pred = (predictions.flatten() > threshold).astype(int)

# Agregar predicciones al DataFrame de prueba
X_test['predicted_label'] = y_pred
X_test['true_label'] = y_test.values

# Mostrar resultados
print(X_test[['predicted_label', 'true_label']].head())

# Guardar resultados
X_test.to_csv('model_outputs/sftp_test_results.csv', index=False)
print("Resultados guardados en 'model_outputs/sftp_test_results.csv'.")
