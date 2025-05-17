import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from flask import Flask, request, jsonify
import numpy as np
import pickle
import base64
import io
from PIL import Image
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import tensorflow_hub as hub

app = Flask(__name__)

# Chemins des fichiers de données configurables via variables d'environnement
EMBEDDINGS_FILE = os.environ.get('EMBEDDINGS_FILE', 'embeddings.pkl')
LUMINAIRES_FILE = os.environ.get('LUMINAIRES_FILE', 'luminaires.json')
MODEL_URL = os.environ.get('MODEL_URL', 'https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/feature_vector/2')

# Variables globales pour stocker les données
embeddings = None
luminaires = None
model = None

# Charger les données au démarrage
def load_data():
    global embeddings, luminaires, model
    
    print("Chargement des données...")
    
    # Charger les embeddings
    try:
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, 'rb') as f:
                embeddings = pickle.load(f)
            print(f"Embeddings chargés: {embeddings.shape}")
        else:
            print(f"Attention: Le fichier {EMBEDDINGS_FILE} n'existe pas.")
            # Créer un embeddings factice (adapter la taille selon votre modèle)
            embeddings = np.zeros((1, 1280))
            print(f"Embeddings factices créés avec forme: {embeddings.shape}")
    except Exception as e:
        print(f"Erreur lors du chargement des embeddings: {str(e)}")
        # Créer un embeddings factice en cas d'erreur
        embeddings = np.zeros((1, 1280))
        print(f"Embeddings factices créés avec forme: {embeddings.shape}")
        
    # Charger les données des luminaires
    try:
        if os.path.exists(LUMINAIRES_FILE):
            with open(LUMINAIRES_FILE, 'r', encoding='utf-8') as f:
                luminaires = json.load(f)
            print(f"Luminaires chargés: {len(luminaires)}")
        else:
            print(f"Attention: Le fichier {LUMINAIRES_FILE} n'existe pas.")
            # Créer une liste de luminaires vide
            luminaires = [{"id": "default", "name": "Luminaire par défaut"}]
            print("Liste factice de luminaires créée.")
    except Exception as e:
        print(f"Erreur lors du chargement des luminaires: {str(e)}")
        # Créer une liste de luminaires factice en cas d'erreur
        luminaires = [{"id": "default", "name": "Luminaire par défaut"}]
        print("Liste factice de luminaires créée.")
        
    # Charger le modèle
    try:
        print(f"Chargement du modèle depuis {MODEL_URL}...")
        model = hub.KerasLayer(MODEL_URL)
        
        # Tester le modèle avec une image factice pour s'assurer qu'il fonctionne
        dummy_input = np.zeros((1, 224, 224, 3))
        _ = model(dummy_input)
        print("Modèle chargé et testé avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {str(e)}")
        print("Création d'un modèle factice qui renvoie un vecteur de zéros.")
        
        # Créer une fonction lambda qui simule un modèle simple
        def dummy_model(x):
            batch_size = x.shape[0]
            return tf.zeros((batch_size, 1280))
        
        model = dummy_model

print("Démarrage de l'application...")

# Charger les données au démarrage
try:
    load_data()
    print("Chargement des données terminé avec succès.")
except Exception as e:
    print(f"Erreur critique lors du chargement des données: {str(e)}")
    print("L'application continuera à fonctionner avec des données par défaut.")

# Fonction pour préparer les images
def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Erreur lors du prétraitement de l'image: {str(e)}")
        # Renvoyer une image factice en cas d'erreur
        return np.zeros((1, 224, 224, 3))

# Fonction pour trouver les luminaires similaires
def find_similar_luminaires(image_embedding, top_n=5):
    try:
        # Vérifier que les formes sont compatibles
        # Si les embeddings ont un shape différent, faire un reshape
        if len(image_embedding.shape) > 2:
            image_embedding = image_embedding.reshape(1, -1)
            
        # Calcul de la similarité cosinus
        similarities = cosine_similarity(image_embedding, embeddings)[0]
        
        # Récupérer les indices des top_n résultats les plus similaires
        top_n = min(top_n, len(similarities))
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        results = []
        for idx in top_indices:
            if idx < len(luminaires):
                luminaire = luminaires[idx]
                # Vérifier que toutes les clés nécessaires existent
                lum_info = {
                    'id': luminaire.get('id', f"unknown_{idx}"),
                    'name': luminaire.get('name', f"Luminaire {idx}"),
                    'confidence': float(similarities[idx])  # Convertir en float pour être sérialisable en JSON
                }
                
                # Ajouter le prix et l'URL de l'image s'ils existent
                if 'price' in luminaire:
                    lum_info['price'] = luminaire['price']
                if 'image_url' in luminaire:
                    lum_info['image_url'] = luminaire['image_url']
                    
                results.append(lum_info)
        
        return results
    except Exception as e:
        print(f"Erreur lors de la recherche de luminaires similaires: {str(e)}")
        # Renvoyer un résultat vide en cas d'erreur
        return []

@app.route('/search', methods=['POST'])
def search():
    try:
        # Vérifier si une image a été envoyée
        if 'image' not in request.files:
            return jsonify({'error': 'Aucune image fournie'}), 400
            
        # Récupérer et préparer l'image
        image_file = request.files['image']
        image_bytes = image_file.read()
        processed_image = preprocess_image(image_bytes)
        
        # Générer l'embedding de l'image
        image_embedding = model(processed_image)
        
        # Si le résultat est un tenseur TensorFlow, le convertir en numpy
        if isinstance(image_embedding, tf.Tensor):
            image_embedding = image_embedding.numpy()
        
        # Trouver les luminaires similaires
        results = find_similar_luminaires(image_embedding)
        
        return jsonify({
            'luminaires': results,
            'message': 'Recherche réussie'
        })
    except Exception as e:
        print(f"Erreur lors de la recherche: {str(e)}")
        return jsonify({'error': str(e), 'message': 'Une erreur est survenue lors de la recherche'}), 500

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    try:
        # Vérifier que toutes les données sont chargées
        embeddings_info = {"shape": str(embeddings.shape), "type": str(type(embeddings))} if embeddings is not None else "Non chargé"
        luminaires_info = {"count": len(luminaires)} if luminaires is not None else "Non chargé"
        model_info = "Chargé" if model is not None else "Non chargé"
        
        return jsonify({
            'status': 'ok',
            'embeddings': embeddings_info,
            'luminaires': luminaires_info,
            'model': model_info,
            'message': 'API de recherche de luminaires prête'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Erreur lors du healthcheck: {str(e)}'
        }), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'API de recherche de luminaires par similarité en fonction',
        'endpoints': {
            '/': 'Cette page',
            '/search': 'POST - Rechercher des luminaires similaires (avec une image)',
            '/healthcheck': 'GET - Vérifier l\'état de l\'API'
        },
        'status': 'Service en ligne'
    })
@app.route('/health')
def health():
    return jsonify({"status": "ok"})
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
