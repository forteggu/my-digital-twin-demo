import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import subprocess
from sklearn.preprocessing import LabelEncoder


# Llamar a otro script Python
result = subprocess.run(["python", "preprocessor_labelless.py"], capture_output=True, text=True)

# Imprimir la salida del script invocado
print(result.stdout)
print(result.stderr)  # Para errores

# Cargar el modelo
model_path = 'models/sftp_anomaly_detector.h5'
model = load_model(model_path)

# Cargar el nuevo conjunto de datos (sin la columna 'label')
data_path = 'processed_datasets/labelless_structured_sftp_logs_with_counts.csv'  # Reemplazar con tu archivo
new_data = pd.read_csv(data_path)

# Generar 'log_type_encoded' si no existe
if 'log_type_encoded' not in new_data.columns:
    print("Generando 'log_type_encoded'...")
    label_encoder = LabelEncoder()
    new_data['log_type_encoded'] = label_encoder.fit_transform(new_data['log_type'].astype(str))
    print("Valores únicos de log_type y sus codificaciones:")
    print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# Asegurarse de que las columnas sean correctas
features = ["failed_password_attempt_count", "ip_attempt_count", "log_type_encoded"]
X_new = new_data[features]

# Manejar valores NaN si los hay
X_new = X_new.fillna(0)

# Hacer predicciones
predictions = model.predict(X_new)

# Convertir probabilidades en etiquetas (0: normal, 1: anomaly)
threshold = 0.5
new_data['predicted_label'] = (predictions.flatten() > threshold).astype(int)

# Interpretar etiquetas
new_data['predicted_label'] = new_data['predicted_label'].map({0: 'normal', 1: 'anomaly'})

# Guardar resultados
new_data.to_csv('model_outputs/sftp_test_results.csv', index=False)

# Mostrar las primeras filas con predicciones
print(new_data.head())
