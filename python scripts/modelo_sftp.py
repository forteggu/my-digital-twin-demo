import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Cargar el dataset
df = pd.read_csv('structured_sftp_logs_with_counts.csv')

# Seleccionar las columnas relevantes
features = ['failed_attempt_count', 'invalid_user_count', 'port']  # Añade otras características relevantes
target = 'label'

# Convertir etiquetas en valores numéricos
label_encoder = LabelEncoder()
df[target] = label_encoder.fit_transform(df[target])  # 'normal' -> 0, 'anomaly' -> 1

# Dividir en características (X) y etiquetas (y)
X = df[features].values
y = df[target].values

# Normalizar las características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir los datos en entrenamiento, validación y prueba
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Crear el modelo
# model = Sequential([
#     Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
#     Dropout(0.3),
#     Dense(8, activation='relu'),
#     Dropout(0.3),
#     Dense(1, activation='sigmoid')  # Salida binaria: normal (0) o anomalía (1)
# ])

# Compilar el modelo
# model.compile(optimizer='adam', 
#               loss='binary_crossentropy', 
#               metrics=['accuracy'])


# Entrenar el modelo
# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     epochs=20,  # Ajusta según el tamaño del dataset
#     batch_size=32,
#     verbose=1
# )


# Evaluar el modelo en los datos de prueba
# loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
# print(f"Pérdida: {loss:.4f}, Precisión: {accuracy:.4f}")


# # Cargar el modelo entrenado
# loaded_model = tf.keras.models.load_model('sftp_brute_force_model.h5')

# # Usar el modelo para hacer predicciones
# predictions = loaded_model.predict(new_data)  # Sustituye new_data con los datos en tiempo real
