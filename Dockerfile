FROM python:3.9-slim

WORKDIR /app

COPY . .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Créer le fichier start.sh
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

CMD ["/app/start.sh"]
