import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

# Limiter l'utilisation de la mémoire
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print(f"GPU configuration error (can be ignored on CPU-only machines): {e}")

# Réduire les logs de TensorFlow
tf.get_logger().setLevel('ERROR')

import pickle
import numpy as np
from flask import Flask, request, jsonify
import base64
import io
from PIL import Image
import json
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub

app = Flask(__name__)

# Chemins des fichiers
EMBEDDINGS_FILE = os.environ.get('EMBEDDINGS_FILE', 'embeddings.pkl')
LUMINAIRES_FILE = os.environ.get('LUMINAIRES_FILE', 'luminaires.json')
MODEL_URL = os.environ.get('MODEL_URL', 'https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/feature_vector/2')

# Variables globales
embeddings = None
luminaires = None
model = None

def load_data():
    global embeddings, luminaires, model
    print("Chargement des données...")

    try:
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, 'rb') as f:
                embeddings = pickle.load(f)
            print(f"Embeddings chargés: {embeddings.shape}")
            with open('/tmp/health/embeddings_loaded', 'w') as f:
                f.write('1')
        else:
            embeddings = np.zeros((1, 1280))
            print("Fichier d'embeddings introuvable, vecteur factice utilisé.")
    except Exception as e:
        print(f"Erreur chargement embeddings: {e}")
        embeddings = np.zeros((1, 1280))

    try:
        if os.path.exists(LUMINAIRES_FILE):
            with open(LUMINAIRES_FILE, 'r', encoding='utf-8') as f:
                luminaires = json.load(f)
            print(f"Luminaires chargés: {len(luminaires)}")
        else:
            luminaires = [{"id": "default", "name": "Luminaire par défaut"}]
    except Exception as e:
        print(f"Erreur chargement luminaires: {e}")
        luminaires = [{"id": "default", "name": "Luminaire par défaut"}]

    try:
        print(f"Chargement du modèle depuis {MODEL_URL}...")
        model = hub.KerasLayer(MODEL_URL)
        dummy_input = np.zeros((1, 224, 224, 3))
        _ = model(dummy_input)
        print("Modèle chargé avec succès.")
    except Exception as e:
        print(f"Erreur chargement modèle: {e}")
        def dummy_model(x):
            return tf.zeros((x.shape[0], 1280))
        model = dummy_model

    try:
        os.makedirs('/tmp/health', exist_ok=True)
        with open('/tmp/health/status', 'w') as f:
            f.write('ready')
    except Exception as e:
        with open('/tmp/health/error', 'w') as f:
            f.write(str(e))

print("Initialisation de l'application...")
load_data()

def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Erreur prétraitement image: {e}")
        return np.zeros((1, 224, 224, 3))

def find_similar_luminaires(image_embedding, top_n=5):
    try:
        if len(image_embedding.shape) > 2:
            image_embedding = image_embedding.reshape(1, -1)
        similarities = cosine_similarity(image_embedding, embeddings)[0]
        top_n = min(top_n, len(similarities))
        top_indices = similarities.argsort()[-top_n:][::-1]
        results = []
        for idx in top_indices:
            if idx < len(luminaires):
                lum = luminaires[idx]
                results.append({
                    'id': lum.get('id', f'unknown_{idx}'),
                    'name': lum.get('name', f"Luminaire {idx}"),
                    'confidence': float(similarities[idx]),
                    'price': lum.get('price', 'N/A'),
                    'image_url': lum.get('image_url', None)
                })
        return results
    except Exception as e:
        print(f"Erreur recherche similarité: {e}")
        return []

@app.route('/search', methods=['POST'])
def search():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Aucune image fournie'}), 400
        image_file = request.files['image']
        image_bytes = image_file.read()
        processed_image = preprocess_image(image_bytes)
        image_embedding = model(processed_image)
        if isinstance(image_embedding, tf.Tensor):
            image_embedding = image_embedding.numpy()
        results = find_similar_luminaires(image_embedding)
        return jsonify({'luminaires': results, 'message': 'Recherche réussie'})
    except Exception as e:
        print(f"Erreur /search: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    try:
        embeddings_info = {"shape": str(embeddings.shape)} if embeddings is not None else "Non chargé"
        luminaires_info = {"count": len(luminaires)} if luminaires is not None else "Non chargé"
        model_info = "Chargé" if model is not None else "Non chargé"
        return jsonify({
            'status': 'ok',
            'embeddings': embeddings_info,
            'luminaires': luminaires_info,
            'model': model_info
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/')
def home():
    return jsonify({
        'message': 'API de recherche de luminaires par similarité',
        'endpoints': {
            '/': 'Cette page',
            '/search': 'POST - Recherche par image',
            '/healthcheck': 'GET - Vérification de l\'état',
            '/health': 'GET - Healthcheck Render'
        }
    })

@app.route('/health')
def health():
    status = "unknown"
    if os.path.exists('/tmp/health/status'):
        with open('/tmp/health/status', 'r') as f:
            status = f.read().strip()
    embeddings_loaded = os.path.exists('/tmp/health/embeddings_loaded')
    error = None
    if os.path.exists('/tmp/health/error'):
        with open('/tmp/health/error', 'r') as f:
            error = f.read().strip()

    mem_info = {}
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemTotal' in line or 'MemAvailable' in line:
                    k, v = line.strip().split(':')
                    mem_info[k.strip()] = v.strip()
    except:
        mem_info = {"error": "Lecture mémoire échouée"}

    response = {
        "status": "ok" if status == "ready" and embeddings_loaded else "initializing",
        "details": {
            "app_status": status,
            "embeddings_loaded": embeddings_loaded,
            "memory": mem_info
        }
    }
    if error:
        response["error"] = error
    return jsonify(response)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
