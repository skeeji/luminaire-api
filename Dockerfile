FROM python:3.9-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements et installer les dépendances
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier les scripts de conversion
COPY convert_files.py ./

# Copier le reste des fichiers
COPY . .

# Exécuter le script de conversion si nécessaire
RUN if [ -f embeddings.pkl ] && [ -f luminaires.json ]; then \
        python convert_files.py; \
        # Remplacer les fichiers originaux par les versions compatibles si la conversion a réussi
        if [ -f embeddings_compatible.pkl ]; then mv embeddings_compatible.pkl embeddings.pkl; fi; \
        if [ -f luminaires_compatible.json ]; then mv luminaires_compatible.json luminaires.json; fi; \
    fi

# Exposition du port
EXPOSE 8080

# Commande pour démarrer l'application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120", "app:app"]