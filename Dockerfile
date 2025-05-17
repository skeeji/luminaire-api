FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x /app/start.sh

EXPOSE 8080

CMD ["/app/start.sh"]
RUN echo '#!/bin/bash\necho "Vérification des fichiers:"\nls -la\necho "Démarrage de Gunicorn:"\nexec gunicorn --bind 0.0.0.0:8080 --workers 1 --threads 2 --timeout 120 wsgi:application' > /app/start.sh
RUN chmod +x /app/start.sh

# Utiliser le script de démarrage
CMD ["/app/start.sh"]
