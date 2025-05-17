import os
import sys

# Ajouter le répertoire courant au chemin de recherche Python
sys.path.insert(0, os.path.dirname(__file__))

try:
    print("Démarrage de l'application via wsgi.py...")
    from app import app as application
    print("Application importée avec succès.")
except Exception as e:
    print(f"Erreur lors de l'import de l'application: {str(e)}")
    
    # Créer une application Flask minimale en cas d'échec
    from flask import Flask, jsonify
    application = Flask(__name__)
    
    @application.route('/', methods=['GET'])
    def error_home():
        return jsonify({
            'status': 'error',
            'message': 'Application en mode dégradé en raison d\'une erreur de démarrage',
            'error': str(e)
        }), 500

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
