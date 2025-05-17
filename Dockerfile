FROM python:3.9-slim

WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Ajouter un script de démarrage
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier tous les fichiers de l'application
COPY . .

# Vérifier les fichiers présents
RUN echo "Contenu du répertoire:" && ls -la

# Exposer le port
EXPOSE 8080

# Script de démarrage
RUN echo '#!/bin/bash\necho "Vérification des fichiers:"\nls -la\necho "Démarrage de Gunicorn:"\nexec gunicorn --bind 0.0.0.0:8080 --workers 1 --threads 2 --timeout 120 wsgi:application' > /app/start.sh
RUN chmod +x /app/start.sh

# Utiliser le script de démarrage
CMD ["/app/start.sh"]
