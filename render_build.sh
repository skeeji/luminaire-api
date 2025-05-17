#!/bin/bash

# Script de démarrage optimisé pour Render.com

# Configurer l'environnement
export PYTHONUNBUFFERED=1
export TF_FORCE_GPU_ALLOW_GROWTH=false
export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL=3  # Réduit les logs de TensorFlow

echo "Démarrage du service..."
echo "Vérification des fichiers:"
ls -la

# Exécution de la vérification d'environnement
if [ -f "check_env.py" ]; then
  echo "Exécution de la vérification d'environnement..."
  python check_env.py
fi

# Créer un fichier de statut pour les vérifications de santé
mkdir -p /tmp/health
echo "starting" > /tmp/health/status

# Démarrage de l'application en arrière-plan
(
  # Précharger les embeddings dans un processus séparé pour éviter le timeout
  python -c "
import pickle
import os
print('Préchargement des embeddings...')
try:
  if os.path.exists('embeddings.pkl'):
    with open('embeddings.pkl', 'rb') as f:
      data = pickle.load(f)
    print(f'Préchargement terminé: {len(data)} embeddings chargés')
    # Créer un fichier pour indiquer que le préchargement est terminé
    with open('/tmp/health/embeddings_loaded', 'w') as f:
      f.write('ok')
except Exception as e:
  print(f'Erreur lors du préchargement: {e}')
  with open('/tmp/health/error', 'w') as f:
    f.write(str(e))
"

  # Mise à jour du statut
  echo "ready" > /tmp/health/status
) &

# Donner du temps pour le préchargement
sleep 5

# Démarrer Gunicorn avec un timeout plus long
echo "Démarrage de Gunicorn:"
gunicorn --workers=1 --timeout=300 --graceful-timeout=60 --threads=4 --worker-class=gthread wsgi:app
