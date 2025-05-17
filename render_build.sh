#!/bin/bash

# Script d'installation et de démarrage pour Render.com

# Afficher les informations du système
echo "Informations système:"
python3 --version
pip --version

# Installer les dépendances
echo "Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

# Vérifier si les fichiers de données existent
echo "Vérification des fichiers de données..."
if [ -f embeddings.pkl ] && [ -f luminaires.json ]; then
    echo "Fichiers de données trouvés."
    
    # Exécuter le script de conversion
    echo "Conversion des fichiers si nécessaire..."
    python convert_files.py
    
    # Remplacer les fichiers originaux par les versions compatibles si la conversion a réussi
    if [ -f embeddings_compatible.pkl ]; then 
        mv embeddings_compatible.pkl embeddings.pkl
        echo "Fichier embeddings.pkl remplacé par la version compatible."
    fi
    
    if [ -f luminaires_compatible.json ]; then 
        mv luminaires_compatible.json luminaires.json
        echo "Fichier luminaires.json remplacé par la version compatible."
    fi
else
    echo "ATTENTION: Fichiers de données manquants!"
    echo "Veuillez ajouter les fichiers embeddings.pkl et luminaires.json avant le déploiement."
fi

# Démarrer l'application
echo "Démarrage de l'application..."
gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 app:app