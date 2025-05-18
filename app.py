import os
# Désactiver l'utilisation du GPU pour éviter les problèmes sur Render
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

# Limiter les logs de TensorFlow pour éviter de surcharger les logs
tf.get_logger().setLevel('ERROR')

import pickle
import numpy as np
from flask import Flask, request, jsonify
import io
from PIL import Image
import json
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub
import time
import logging
import threading

# Configuration du logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    logger.info("Initialisation de l'application...")

    # Créer le répertoire pour les fichiers d'état
    os.makedirs('/tmp/health', exist_ok=True)
    
    # Initialisation avec une valeur par défaut pour les embeddings
    # pour permettre un démarrage rapide de l'application
    embeddings = np.zeros((1, 1280))
    
    # Chargement des embeddings avec un délai pour éviter de surcharger la RAM au démarrage
    def load_embeddings():
        global embeddings
        try:
            if os.path.exists(EMBEDDINGS_FILE):
                logger.info(f"Chargement des embeddings depuis {EMBEDDINGS_FILE} (fichier volumineux)")
                with open(EMBEDDINGS_FILE, 'rb') as f:
                    embeddings = pickle.load(f)
                logger.info(f"Embeddings chargés avec succès: {embeddings.shape}")
                with open('/tmp/health/embeddings_loaded', 'w') as f:
                    f.write('1')
            else:
                logger.warning(f"Fichier d'embeddings {EMBEDDINGS_FILE} introuvable, vecteur factice utilisé.")
                embeddings = np.zeros((1, 1280))
        except Exception as e:
            logger.error(f"Erreur chargement embeddings: {e}")
            embeddings = np.zeros((1, 1280))
    
    # Marquer que le chargement des embeddings a commencé
    with open('/tmp/health/embeddings_loading', 'w') as f:
        f.write('1')
    
    # Démarrer le chargement dans un thread séparé pour ne pas bloquer le démarrage de l'application
    embedding_thread = threading.Thread(target=load_embeddings, daemon=True)
    embedding_thread.start()

    # Chargement des données de luminaires
    try:
        if os.path.exists(LUMINAIRES_FILE):
            logger.info(f"Chargement des luminaires depuis {LUMINAIRES_FILE}")
            with open(LUMINAIRES_FILE, 'r', encoding='utf-8') as f:
                luminaires = json.load(f)
            logger.info(f"Luminaires chargés: {len(luminaires)}")
        else:
            logger.warning(f"Fichier de luminaires {LUMINAIRES_FILE} introuvable, données factices utilisées.")
            luminaires = [{"id": "default", "name": "Luminaire par défaut"}]
    except Exception as e:
        logger.error(f"Erreur chargement luminaires: {e}")
        luminaires = [{"id": "default", "name": "Luminaire par défaut"}]

    # Chargement du modèle
    try:
        logger.info(f"Chargement du modèle depuis {MODEL_URL}...")
        model = hub.KerasLayer(MODEL_URL)
        # Test du modèle avec une entrée factice
        dummy_input = np.zeros((1, 224, 224, 3))
        _ = model(dummy_input)
        logger.info("Modèle chargé avec succès.")
    except Exception as e:
        logger.error(f"Erreur chargement modèle: {e}")
        # Modèle factice en cas d'erreur
        def dummy_model(x):
            return tf.zeros((x.shape[0], 1280))
        model = dummy_model

    # Marquer l'application comme prête
    try:
        with open('/tmp/health/status', 'w') as f:
            f.write('ready')
        logger.info("Application initialisée et prête (embeddings en cours de chargement en arrière-plan)")
    except Exception as e:
        logger.error(f"Erreur lors de la création du fichier d'état: {e}")
        with open('/tmp/health/error', 'w') as f:
            f.write(str(e))

# Initialiser l'application et charger les données
load_data()

def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        logger.error(f"Erreur prétraitement image: {e}")
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
        logger.error(f"Erreur recherche similarité: {e}")
        return []

@app.route('/search', methods=['POST'])
def search():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Aucune image fournie'}), 400
        
        # Vérifier si les embeddings sont chargés
        embeddings_loaded = os.path.exists('/tmp/health/embeddings_loaded')
        if not embeddings_loaded:
            return jsonify({
                'error': 'Le modèle est en cours de chargement, veuillez réessayer dans quelques instants',
                'status': 'loading'
            }), 503  # Service Unavailable
        
        image_file = request.files['image']
        image_bytes = image_file.read()
        processed_image = preprocess_image(image_bytes)
        image_embedding = model(processed_image)
        if isinstance(image_embedding, tf.Tensor):
            image_embedding = image_embedding.numpy()
        results = find_similar_luminaires(image_embedding)
        return jsonify({'luminaires': results, 'message': 'Recherche réussie'})
    except Exception as e:
        logger.error(f"Erreur /search: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    try:
        embeddings_loaded = os.path.exists('/tmp/health/embeddings_loaded')
        embeddings_loading = os.path.exists('/tmp/health/embeddings_loading')
        
        embeddings_info = {
            "shape": str(embeddings.shape) if embeddings is not None else "Non défini",
            "loaded": embeddings_loaded,
            "loading": embeddings_loading
        }
        
        luminaires_info = {"count": len(luminaires)} if luminaires is not None else "Non chargé"
        model_info = "Chargé" if model is not None else "Non chargé"
        
        # Informations sur la mémoire
        mem_info = {}
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line or 'MemAvailable' in line:
                        k, v = line.strip().split(':')
                        mem_info[k.strip()] = v.strip()
        except Exception as e:
            mem_info = {"error": f"Lecture mémoire échouée: {str(e)}"}
            
        return jsonify({
            'status': 'ok',
            'embeddings': embeddings_info,
            'luminaires': luminaires_info,
            'model': model_info,
            'memory': mem_info
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/')
def home():
    return jsonify({
        'message': 'API de recherche de luminaires par similarité',
        'endpoints': {
            '/': 'Cette page d\'accueil',
            '/search': 'POST - Recherche par image',
            '/healthcheck': 'GET - Vérification détaillée de l\'état',
            '/health': 'GET - Healthcheck Render'
        },
        'status': {
            'embeddings_loaded': os.path.exists('/tmp/health/embeddings_loaded'),
            'embeddings_loading': os.path.exists('/tmp/health/embeddings_loading'),
            'app_ready': os.path.exists('/tmp/health/status')
        }
    })

@app.route('/health')
def health():
    status = "unknown"
    if os.path.exists('/tmp/health/status'):
        with open('/tmp/health/status', 'r') as f:
            status = f.read().strip()
    
    embeddings_loading = os.path.exists('/tmp/health/embeddings_loading')
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
    except Exception as e:
        mem_info = {"error": f"Lecture mémoire échouée: {str(e)}"}

    # Toujours retourner un statut OK pour que Render considère l'application comme démarrée
    # même si les embeddings sont encore en train de charger
    response = {
        "status": "ok",
        "details": {
            "app_status": status,
            "embeddings_loading": embeddings_loading,
            "embeddings_loaded": embeddings_loaded,
            "memory": mem_info,
            "timestamp": time.time()
        }
    }
    if error:
        response["error"] = error
    
    return jsonify(response)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
