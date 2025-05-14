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

# Chemin des fichiers de données
EMBEDDINGS_FILE = 'embeddings.pkl'  # Modifié de 'embeddings.pickle' à 'embeddings.pkl'
LUMINAIRES_FILE = 'luminaires.json'
MODEL_URL = 'https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2'

# Charger les données au démarrage
@app.before_first_request
def load_data():
    global embeddings, luminaires, model
    
    # Charger les embeddings
    with open(EMBEDDINGS_FILE, 'rb') as f:
        embeddings = pickle.load(f)
    
    # Charger les données des luminaires
    with open(LUMINAIRES_FILE, 'r', encoding='utf-8') as f:
        luminaires = json.load(f)
    
    # Charger le modèle
    model = hub.KerasLayer(MODEL_URL)
    
    print(f"Données chargées: {len(embeddings)} embeddings et {len(luminaires)} luminaires")

# Fonction pour préparer les images
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Fonction pour trouver les luminaires similaires
def find_similar_luminaires(image_embedding, top_n=5):
    # Calcul de la similarité cosinus
    similarities = cosine_similarity(image_embedding, embeddings)[0]
    
    # Récupérer les indices des top_n résultats les plus similaires
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        luminaire = luminaires[idx]
        results.append({
            'id': luminaire['id'],
            'name': luminaire['name'],
            'price': luminaire['price'],
            'image_url': luminaire['image_url'],
            'confidence': float(similarities[idx])  # Convertir en float pour être sérialisable en JSON
        })
    
    return results

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
        print(f"Erreur: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'API de recherche de luminaires par similarité en fonction'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
