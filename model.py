import os
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "trained_model.pkl"


class JobFraudDetector:
    def __init__(self):
        # Cargamos un modelo multilingüe optimizado para similitud y clasificación de texto
        print("Cargando modelo multilingüe de Hugging Face...")
        self.encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.classifier = None
        self.load_model()

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            print("Cargando modelo clasificador existente...")
            self.classifier = joblib.load(MODEL_PATH)
        else:
            print("No se encontró modelo entrenado. Inicializando clasificador base por defecto...")
            # Inicializamos un RandomForest por defecto (en producción se entrena con un dataset histórico)
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    def extract_metadata_features(self, email: str, description: str) -> list:
        """Extrae características heurísticas clave para el contexto de México y estafas comunes"""
        free_domains = ["gmail.com", "outlook.com", "yahoo.com", "hotmail.com", "icloud.com"]

        # 1. ¿Usa correo gratuito corporativo? (1 si es sospechoso, 0 si usa dominio propio)
        email_domain = email.split("@")[-1].lower() if "@" in email else ""
        is_free_email = 1.0 if email_domain in free_domains else 0.0

        # 2. ¿Menciona aplicaciones de mensajería informal típicas de estafas en México?
        desc_lower = description.lower()
        mentions_whatsapp = 1.0 if "whatsapp" in desc_lower or "wa.me" in desc_lower else 0.0
        mentions_telegram = 1.0 if "telegram" in desc_lower else 0.0

        # 3. Longitud del texto (las estafas suelen ser muy cortas o genéricas)
        text_length = float(len(description))

        return [is_free_email, mentions_whatsapp, mentions_telegram, text_length]

    def vectorize_jobs(self, jobs: list[dict]):
        texts = [
            f"Título: {j.get('title', '')}. Empresa: {j.get('company', '')}. Descripción: {j.get('description', '')}"
            for j in jobs]

        # Generar embeddings de texto (vectores de 384 dimensiones)
        text_embeddings = self.encoder.encode(texts, show_progress_bar=False)

        # Extraer metadatos tabulares
        metadata_features = np.array([
            self.extract_metadata_features(j.get('email', ''), j.get('description', ''))
            for j in jobs
        ])

        # Combinar embeddings de texto + metadatos numéricos en una sola matriz de características
        combined_features = np.hstack((text_embeddings, metadata_features))
        return combined_features

    def predict_batch(self, jobs: list[dict]):
        if not jobs:
            return []

        X = self.vectorize_jobs(jobs)

        # Si el modelo no ha sido entrenado formalmente con un dataset, usamos un comportamiento heurístico temporal
        # En cuanto entrenes tu modelo con Kaggle + datos locales, esto usará el clasificador real.
        try:
            predictions = self.classifier.predict(X)
            probabilities = self.classifier.predict_proba(X)
            results = []
            for i, job in enumerate(jobs):
                is_fraud = bool(predictions[i])
                confidence = float(np.max(probabilities[i]))
                results.append({
                    "job_id": job.get("id"),
                    "is_fraudulent": is_fraud,
                    "confidence": confidence
                })
            return results
        except Exception:
            # Fallback temporal si el clasificador no está ajustado con datos reales aún
            results = []
            for job in jobs:
                desc = job.get("description", "").lower()
                is_fraud = "whatsapp" in desc or "@gmail.com" in job.get("email", "")
                results.append({
                    "job_id": job.get("id"),
                    "is_fraudulent": is_fraud,
                    "confidence": 0.85 if is_fraud else 0.90
                })
            return results


detector = JobFraudDetector()