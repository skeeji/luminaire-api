FROM python:3.10-slim

WORKDIR /app

# Copier les fichiers de dépendances d'abord (meilleur caching Docker)
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copier le reste du projet
COPY . .

# Créer le script de démarrage (start.sh)
RUN echo '#!/bin/bash\n\
export PYTHONUNBUFFERED=1\n\
export TF_FORCE_GPU_ALLOW_GROWTH=false\n\
export CUDA_VISIBLE_DEVICES=-1\n\
\n\
echo "Vérification des fichiers:"\n\
ls -la\n\
\n\
echo "Démarrage de Gunicorn:"\n\
gunicorn --workers=1 --timeout=120 --threads=4 --worker-class=gthread wsgi:app' > /app/start.sh

# Rendre le script exécutable
RUN chmod +x start.sh

EXPOSE 8080

# Lancer l'application
CMD ["./start.sh"]
