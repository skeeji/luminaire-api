#!/bin/bash

echo "Informations système:"
python3 --version
pip --version

echo "Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Conversion des fichiers si nécessaire..."
python convert_files.py

echo "Test de compatibilité..."
python test_compatibility.py

# Télécharger les fichiers de données si nécessaires
if [ ! -f "embeddings.pkl" ] && [ -n "$EMBEDDINGS_URL" ]; then
    echo "Téléchargement des embeddings depuis $EMBEDDINGS_URL..."
    curl -L -o embeddings.pkl "$EMBEDDINGS_URL"
fi

if [ ! -f "luminaires.json" ] && [ -n "$LUMINAIRES_URL" ]; then
    echo "Téléchargement des luminaires depuis $LUMINAIRES_URL..."
    curl -L -o luminaires.json "$LUMINAIRES_URL"
fi

echo "Build terminé avec succès!"
