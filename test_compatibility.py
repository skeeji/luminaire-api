import pickle
import json
import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import io
import sys
from sklearn.metrics.pairwise import cosine_similarity

def test_embeddings_compatibility():
    """Test si les embeddings sont compatibles avec la version de NumPy utilisée"""
    print("Test de compatibilité des embeddings...")
    
    embeddings_file = 'embeddings.pkl'
    
    if not os.path.exists(embeddings_file):
        print(f"Erreur: Le fichier {embeddings_file} n'existe pas.")
        return False
    
    try:
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
        
        print(f"Embeddings chargés avec succès.")
        print(f"Type: {type(embeddings)}")
        print(f"Shape: {embeddings.shape}")
        print(f"NumPy version: {np.__version__}")
        
        # Tester une opération simple
        print("Test d'opération sur les embeddings...")
        mean_value = np.mean(embeddings)
        print(f"Valeur moyenne: {mean_value}")
        
        return True
    except Exception as e:
        print(f"Erreur lors du test des embeddings: {str(e)}")
        return False

def test_luminaires_json():
    """Test si le fichier JSON des luminaires est valide"""
    print("Test du fichier JSON des luminaires...")
    
    luminaires_file = 'luminaires.json'
    
    if not os.path.exists(luminaires_file):
        print(f"Erreur: Le fichier {luminaires_file} n'existe pas.")
        return False
    
    try:
        with open(luminaires_file, 'r', encoding='utf-8') as f:
            luminaires = json.load(f)
        
        print(f"Fichier JSON chargé avec succès.")
        print(f"Nombre de luminaires: {len(luminaires)}")
        
        # Vérifier la structure
        if len(luminaires) > 0:
            first_luminaire = luminaires[0]
            print(f"Premier luminaire: {first_luminaire}")
            
            # Vérifier les clés nécessaires
            required_keys = ['id', 'name']
            missing_keys = [key for key in required_keys if key not in first_luminaire]
            
            if missing_keys:
                print(f"Attention: Clés manquantes dans le premier luminaire: {missing_keys}")
        
        return True
    except Exception as e:
        print(f"Erreur lors du test du fichier JSON: {str(e)}")
        return False

def test_tensorflow_model():
    """Test si le modèle TensorFlow peut être chargé"""
    print("Test du modèle TensorFlow...")
    
    try:
        model_url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/feature_vector/2"
        model = hub.KerasLayer(model_url)
        
        print("Modèle chargé avec succès.")
        
        # Créer une image test
        test_image = np.random.random((1, 224, 224, 3)).astype(np.float32)
        
        # Tester l'inférence
        print("Test d'inférence...")
        features = model(test_image)
        
        print(f"Inférence réussie. Shape des features: {features.shape}")
        
        return True
    except Exception as e:
        print(f"Erreur lors du test du modèle TensorFlow: {str(e)}")
        return False

def run_all_tests():
    """Exécuter tous les tests"""
    print("=== Démarrage des tests de compatibilité ===\n")
    
    tests = [
        ("Test des embeddings", test_embeddings_compatibility),
        ("Test du fichier JSON", test_luminaires_json),
        ("Test du modèle TensorFlow", test_tensorflow_model)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n=== {test_name} ===")
        if test_func():
            print(f"✓ {test_name} réussi")
        else:
            print(f"✗ {test_name} échoué")
            all_passed = False
    
    print("\n=== Résumé des tests ===")
    if all_passed:
        print("✓ Tous les tests ont réussi. Le système est compatible.")
    else:
        print("✗ Certains tests ont échoué. Veuillez résoudre les problèmes avant le déploiement.")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)