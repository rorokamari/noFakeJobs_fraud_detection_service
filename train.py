import numpy as np
import joblib
from model import detector, MODEL_PATH
from sklearn.ensemble import RandomForestClassifier

# 1. Creamos un dataset simulado inicial (en el futuro puedes reemplazar esto cargando un CSV de Kaggle)
training_data = [
    # Legítimos (label: 0)
    {
        "title": "Ingeniero de Software Backend",
        "company": "Kavak",
        "description": "Buscamos desarrollador con experiencia en Python, FastAPI, microservicios y bases de datos relacionales. Ofrecemos esquema 100nomina y prestaciones superiores.",
        "email": "recruiting@kavak.com",
        "label": 0
    },
    {
        "title": "Contador Público Titulado",
        "company": "Grupo Bimbo",
        "description": "Solicitamos contador con experiencia en auditoría fiscal. Zona corporativa Santa Fe, CDMX. Indispensable cédula profesional.",
        "email": "seleccion@grupobimbo.com",
        "label": 0
    },
    # Fraudulentos (label: 1)
    {
        "title": "Trabaja desde casa armando plumas",
        "company": "Sin registrar",
        "description": "Gana hasta 8000 pesos semanales desde la comodidad de tu hogar. No se necesita experiencia previa. Envía mensaje por WhatsApp urgente.",
        "email": "trabajosfaciles99@gmail.com",
        "label": 1
    },
    {
        "title": "Inversión inicial para empleo de empaque",
        "company": "Corporativo Global",
        "description": "Gane dinero rápido. Se requiere un pago único de 300 pesos para el envío del kit de materiales de trabajo. Contacto por Telegram.",
        "email": "empleo.global2024@outlook.com",
        "label": 1
    }
]

print("Preparando datos y extrayendo vectores para entrenamiento...")
jobs_only = [{k: v for k, v in item.items() if k != 'label'} for item in training_data]
y = np.array([item['label'] for item in training_data])

# Vectorizamos usando el mismo motor del modelo
X = detector.vectorize_jobs(jobs_only)

# Entrenamos el clasificador Random Forest
print("Entrenando el clasificador RandomForest...")
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X, y)

# Guardamos el modelo entrenado en disco
joblib.dump(classifier, MODEL_PATH)
print(f"¡Modelo entrenado y guardado exitosamente en '{MODEL_PATH}'!")