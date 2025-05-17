import pickle
import numpy as np
import sys
import os

def convert_embeddings(input_file, output_file=None):
    """
    Convertit un fichier d'embeddings généré avec NumPy 2.0.2 en format compatible avec NumPy 1.23.5
    """
    if output_file is None:
        output_file = input_file.replace('.pkl', '_compatible.pkl')

    print(f"Conversion du fichier {input_file} vers {output_file}...")
    
    try:
        # Ouvrir le fichier pickle qui contient les embeddings
        with open(input_file, 'rb') as f:
            try:
                embeddings = pickle.load(f)
                print(f"Embeddings chargés avec succès. Shape: {embeddings.shape}")
            except Exception as e:
                print(f"Erreur lors du chargement des embeddings: {str(e)}")
                print("Tentative de chargement avec protocole alternatif...")
                # Essayer avec différents protocoles pickle
                for protocol in range(2, 6):
                    try:
                        f.seek(0)
                        embeddings = pickle.load(f, fix_imports=True, encoding='bytes')
                        print(f"Embeddings chargés avec protocole alternatif. Shape: {embeddings.shape}")
                        break
                    except:
                        continue
                else:
                    raise Exception("Impossible de charger les embeddings avec les protocoles disponibles")
        
        # Convertir les embeddings en format compatible
        # Nous allons simplement les réencoder avec pickle en utilisant un protocole plus ancien
        with open(output_file, 'wb') as f:
            pickle.dump(np.array(embeddings), f, protocol=4)  # Protocole 4 est compatible avec Python 3.8+
        
        print(f"Conversion réussie. Fichier sauvegardé: {output_file}")
        return True
    
    except Exception as e:
        print(f"Erreur lors de la conversion: {str(e)}")
        return False

def convert_luminaires_json(input_file, output_file=None):
    """
    Vérifie et corrige si nécessaire le fichier JSON de luminaires
    """
    import json
    
    if output_file is None:
        output_file = input_file.replace('.json', '_compatible.json')
    
    print(f"Vérification du fichier {input_file}...")
    
    try:
        # Charger le fichier JSON
        with open(input_file, 'r', encoding='utf-8') as f:
            luminaires = json.load(f)
        
        print(f"Fichier JSON chargé avec succès. {len(luminaires)} luminaires trouvés.")
        
        # Vérifier la structure et corriger si nécessaire
        for i, luminaire in enumerate(luminaires):
            # S'assurer que toutes les clés nécessaires sont présentes
            if 'id' not in luminaire:
                luminaire['id'] = f"unknown_{i}"
            if 'name' not in luminaire:
                luminaire['name'] = f"Luminaire {i}"
            if 'price' not in luminaire:
                luminaire['price'] = 0
            if 'image_url' not in luminaire:
                luminaire['image_url'] = ""
        
        # Sauvegarder le fichier JSON corrigé
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(luminaires, f, ensure_ascii=False, indent=4)
        
        print(f"Vérification terminée. Fichier sauvegardé: {output_file}")
        return True
    
    except Exception as e:
        print(f"Erreur lors de la vérification du fichier JSON: {str(e)}")
        return False

if __name__ == "__main__":
    # Vérifier si les fichiers existent
    embeddings_file = 'embeddings.pkl'
    luminaires_file = 'luminaires.json'
    
    if not os.path.exists(embeddings_file):
        print(f"Erreur: Le fichier {embeddings_file} n'existe pas.")
        sys.exit(1)
    
    if not os.path.exists(luminaires_file):
        print(f"Erreur: Le fichier {luminaires_file} n'existe pas.")
        sys.exit(1)
    
    # Convertir les embeddings
    if not convert_embeddings(embeddings_file):
        print("Erreur lors de la conversion des embeddings.")
        sys.exit(1)
    
    # Vérifier et corriger le fichier JSON
    if not convert_luminaires_json(luminaires_file):
        print("Erreur lors de la vérification du fichier JSON.")
        sys.exit(1)
    
    print("Conversion terminée avec succès.")
    sys.exit(0)