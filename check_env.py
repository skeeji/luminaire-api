import os
import sys
import tensorflow as tf
import numpy as np
import pickle
import json

def check_file(filename):
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        print(f"✅ {filename} existe (taille: {size} octets)")
    else:
        print(f"❌ {filename} n'existe pas")

def main():
    print("=== Vérification de l'environnement Python ===")
    print(f"Python version: {sys.version}")
    print(f"TensorFlow version: {tf.__version__}")
    
    print("\n=== Vérification des fichiers importants ===")
    for file in ['app.py', 'wsgi.py', 'embeddings.pkl', 'luminaires.json']:
        check_file(file)
    
    print("\n=== Informations sur le système ===")
    print(f"Répertoire de travail: {os.getcwd()}")
    print(f"Contenu du répertoire:")
    print('\n'.join(f"- {f}" for f in os.listdir('.')))
    
    # Vérifier GPU
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\n=== Dispositifs GPU: {len(gpus)} ===")
    for gpu in gpus:
        print(f"- {gpu}")
    
    print("\n=== Test terminé ===")

if __name__ == "__main__":
    main()
