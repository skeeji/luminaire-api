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

# Chemins des fichiers de données
EMBEDDINGS_FILE = 'embeddings.pkl'
LUMINAIRES_FILE = 'luminaires.json'
MODEL_URL = 'https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/feature_vector/2'

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
        with open(EMBEDDINGS_FILE, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Embeddings chargés: {embeddings.shape}")
    except Exception as e:
        print(f"Erreur lors du chargement des embeddings: {str(e)}")
        raise
        
    # Charger les données des luminaires
    try:
        with open(LUMINAIRES_FILE, 'r', encoding='utf-8') as f:
            luminaires = json.load(f)
        print(f"Luminaires chargés: {len(luminaires)}")
    except Exception as e:
        print(f"Erreur lors du chargement des luminaires: {str(e)}")
        raise
        
    # Charger le modèle
    try:
        model = hub.KerasLayer(MODEL_URL)
        print("Modèle chargé avec succès")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {str(e)}")
        raise

# Charger les données au démarrage (remplace before_first_request)
load_data()

# Fonction pour préparer les images
def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Erreur lors du prétraitement de l'image: {str(e)}")
        raise

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
        raise

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
        image_embedding = model(processed_image).numpy()
        
        # Trouver les luminaires similaires
        results = find_similar_luminaires(image_embedding)
        
        return jsonify({
            'luminaires': results,
            'message': 'Recherche réussie'
        })
    except Exception as e:
        print(f"Erreur lors de la recherche: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    # Vérifier que toutes les données sont chargées correctement
    if embeddings is None or luminaires is None or model is None:
        return jsonify({
            'status': 'error',
            'message': 'Les données ne sont pas correctement chargées'
        }), 500
    
    return jsonify({
        'status': 'ok',
        'embeddings_shape': embeddings.shape,
        'luminaires_count': len(luminaires),
        'message': 'API de recherche de luminaires prête'
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'API de recherche de luminaires par similarité en fonction',
        'endpoints': {
            '/': 'Cette page',
            '/search': 'POST - Rechercher des luminaires similaires (avec une image)',
            '/healthcheck': 'GET - Vérifier l\'état de l\'API'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)