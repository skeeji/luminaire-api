# API de Recherche de Luminaires Similaires

Cette API permet de rechercher des luminaires similaires à partir d'une image. Elle utilise des embeddings générés à partir d'un modèle EfficientNet pour comparer les similitudes visuelles entre les produits.

## Fonctionnalités

- Recherche de luminaires similaires à partir d'une image
- Utilisation de la similarité cosinus pour trouver les correspondances
- API RESTful avec Flask

## Prérequis

- Python 3.9+
- Fichiers de données:
  - `embeddings.pkl`: Fichier contenant les embeddings des images de luminaires
  - `luminaires.json`: Fichier contenant les informations des luminaires

## Installation locale

1. Cloner le dépôt:
```bash
git clone https://github.com/skeeji/luminaire-api.git
cd luminaire-api
```

2. Installer les dépendances:
```bash
pip install -r requirements.txt
```

3. Placer les fichiers de données (`embeddings.pkl` et `luminaires.json`) à la racine du projet

4. Lancer le script de test pour vérifier la compatibilité:
```bash
python test_compatibility.py
```

5. Exécuter l'application:
```bash
python app.py
```

## Déploiement sur Render.com

1. Connectez-vous à [Render.com](https://render.com)
2. Créez un nouveau service Web
3. Connectez votre dépôt GitHub
4. Configurez les paramètres suivants:
   - **Build Command**: `chmod +x render_build.sh && ./render_build.sh`
   - **Start Command**: (laissez vide, car défini dans render_build.sh)
   - **Environment**: Docker
5. Ajoutez vos fichiers de données (`embeddings.pkl` et `luminaires.json`) au dépôt ou configurez un stockage persistant

## Utilisation de l'API

### Recherche de luminaires similaires

**Endpoint**: `/search`  
**Méthode**: POST  
**Contenu**: Formulaire multipart avec un champ 'image' contenant l'image à rechercher  

**Exemple de requête avec curl**:
```bash
curl -X POST -F "image=@/chemin/vers/votre/image.jpg" https://votre-api.onrender.com/search
```

**Exemple de réponse**:
```json
{
  "luminaires": [
    {
      "id": "lum_123",
      "name": "Luminaire Moderne",
      "confidence": 0.95,
      "price": 299.99,
      "image_url": "https://exemple.com/image.jpg"
    },
    ...
  ],
  "message": "Recherche réussie"
}
```

### Vérification de l'état de l'API

**Endpoint**: `/healthcheck`  
**Méthode**: GET  

**Exemple de requête**:
```bash
curl https://votre-api.onrender.com/healthcheck
```

## Notes sur la compatibilité

Si vous avez généré vos embeddings avec NumPy 2.0.2 dans Google Colab, utilisez le script `convert_files.py` pour les convertir en un format compatible avant le déploiement:

```bash
python convert_files.py
```

## Contribution

Les contributions sont les bienvenues! N'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## Licence

[MIT](https://choosealicense.com/licenses/mit/)