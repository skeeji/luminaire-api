FROM python:3.10-slim

WORKDIR /app

# Copier les dépendances et installer
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copier le code source complet
COPY . .

# Créer le script de démarrage
RUN echo '#!/bin/bash\n\
export PYTHONUNBUFFERED=1\n\
export TF_FORCE_GPU_ALLOW_GROWTH=false\n\
export CUDA_VISIBLE_DEVICES=-1\n\
PORT=${PORT:-8080}\n\
echo "Vérification des fichiers:"\n\
ls -la\n\
echo "Démarrage de Gunicorn sur le port $PORT"\n\
gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --threads 1' > /app/start.sh

# Rendre le script exécutable
RUN chmod +x start.sh

EXPOSE 8080

CMD ["./start.sh"]
