import os
import sys
import time

def check_environment():
    """Vérifie l'environnement et imprime des informations de diagnostic"""
    print("=== Vérification de l'environnement ===")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
    
    # Vérifier la taille des fichiers importants
    for file in ['embeddings.pkl', 'luminaires.json']:
        if os.path.exists(file):
            print(f"File {file} size: {os.path.getsize(file) / 1024 / 1024:.2f} MB")
    
    # Vérifier la mémoire disponible (Linux seulement)
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemTotal' in line or 'MemAvailable' in line:
                    print(line.strip())
    except:
        print("Impossible de lire les informations mémoire")
    
    # Vérifier les variables d'environnement TensorFlow
    tf_vars = {k: v for k, v in os.environ.items() if 'TF_' in k or 'CUDA_' in k}
    print(f"TensorFlow/CUDA environment variables: {tf_vars}")
    
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        print(f"TensorFlow devices: {tf.config.list_physical_devices()}")
    except Exception as e:
        print(f"Error importing TensorFlow: {e}")

if __name__ == "__main__":
    check_environment()
    print("\nScript completed successfully. If your application is still failing, check the logs for specific error messages.")
