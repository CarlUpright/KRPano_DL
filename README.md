# KRPANO Downloader

## Description

KRPANO Downloader est une application GUI Python qui permet de télécharger et convertir des panoramas 360° au format KRPano. L'application peut extraire des panoramas depuis des fichiers XML KRPano, télécharger automatiquement toutes les tuiles d'images, les assembler en faces de cube, et les convertir en projection équirectangulaire pour une utilisation dans d'autres applications.

## Fonctionnalités

- **Téléchargement automatique** : Parse les fichiers XML KRPano et télécharge toutes les tuiles d'images
- **Support multi-format** : Compatible avec les formats cube et face individuelles
- **Gestion multi-résolution** : Détecte et utilise automatiquement la plus haute résolution disponible
- **Interface utilisateur intuitive** : GUI PyQt5 avec onglets séparés pour les différentes opérations
- **Traitement parallèle** : Téléchargement multi-thread pour une vitesse optimale
- **Conversion équirectangulaire** : Convertit les cubemaps en projection équirectangulaire (format panorama standard)
- **Mode local** : Peut traiter des panoramas déjà téléchargés localement
- **Barre de progression** : Suivi en temps réel du téléchargement et de la conversion

## Prérequis

- Python 3.7 ou supérieur
- Système d'exploitation : Windows, macOS, ou Linux

## Installation

### 1. Cloner ou télécharger le script

Téléchargez le fichier `KRPANO_downloader[CARL].py` sur votre ordinateur.

### 2. Installer les dépendances

Ouvrez un terminal/invite de commande et installez les packages Python requis :

```bash
pip install requests beautifulsoup4 numpy pillow opencv-python tqdm PyQt5 lxml
```

Ou créez un fichier `requirements.txt` avec le contenu suivant :

```
requests>=2.25.0
beautifulsoup4>=4.9.0
numpy>=1.19.0
Pillow>=8.0.0
opencv-python>=4.5.0
tqdm>=4.60.0
PyQt5>=5.15.0
lxml>=4.6.0
```

Puis installez avec :
```bash
pip install -r requirements.txt
```

### 3. Lancer l'application

```bash
python "KRPANO_downloader[CARL].py"
```

## Utilisation

### Mode Téléchargement depuis URL

1. **Lancez l'application**
2. **Dans l'onglet "Download"** :
   - Sélectionnez "Download from URL"
   - Entrez l'URL du fichier XML KRPano dans le champ "XML URL"
   - Choisissez un dossier de destination pour les fichiers téléchargés
   - Cliquez sur "Download"

3. **Suivi du progrès** :
   - La barre de progression affiche l'avancement du téléchargement
   - Les messages de statut indiquent l'étape en cours

### Mode Dossier Local

1. **Dans l'onglet "Download"** :
   - Sélectionnez "Use Local Folder"
   - Choisissez le dossier contenant les panoramas déjà téléchargés
   - Cliquez sur "Load Scenes" pour scanner les panoramas disponibles

### Conversion Équirectangulaire

1. **Après le téléchargement ou le chargement local** :
   - Passez à l'onglet "Convert to Equirectangular"
   - Sélectionnez le panorama à convertir dans la liste déroulante
   - Ajustez la résolution de sortie (par défaut 8192x4096)
   - Cochez "Maintain aspect ratio (2:1)" pour conserver les proportions correctes
   - Cliquez sur "Convert to Equirectangular"

2. **Paramètres de résolution** :
   - La résolution maximale recommandée est calculée automatiquement
   - Largeur recommandée : 4x la taille des faces
   - Hauteur recommandée : 2x la taille des faces (ratio 2:1)

## Exemples d'URLs supportées

- Sites utilisant KRPano avec structure XML standard
- Panoramas du Louvre et sites similaires
- URLs avec patterns `%SWFPATH%`, `%s`, `%v`, `%h`, `%u` pour les tuiles

Exemple d'URL typique :
```
https://example.com/panorama/tour.xml
```

## Structure des fichiers de sortie

```
Dossier_de_sortie/
├── Panorama_1/
│   ├── 0/          # Face front (tuiles individuelles)
│   ├── 1/          # Face right
│   ├── 2/          # Face back
│   ├── 3/          # Face left
│   ├── 4/          # Face up
│   ├── 5/          # Face down
│   ├── face_front.jpg    # Face assemblée
│   ├── face_right.jpg
│   ├── face_back.jpg
│   ├── face_left.jpg
│   ├── face_up.jpg
│   └── face_down.jpg
└── Panorama_1_equirectangular.jpg  # Image équirectangulaire finale
```

## Résolution des problèmes

### Erreur "No panoramas found"
- Vérifiez que l'URL XML est correcte et accessible
- Assurez-vous que le fichier XML contient bien des éléments `<scene>` avec des images

### Échec de téléchargement de tuiles
- Vérifiez votre connexion internet
- Certains sites peuvent avoir des restrictions d'accès
- L'application essaie automatiquement plusieurs URL de base alternatives

### Erreur de conversion équirectangulaire
- Assurez-vous que toutes les 6 faces du cube sont présentes
- Vérifiez que les images des faces ne sont pas corrompues

### Problèmes de mémoire
- Réduisez la résolution de sortie pour les gros panoramas
- Fermez les autres applications gourmandes en mémoire

## Limitations

- Fonctionne uniquement avec les panoramas au format KRPano
- Nécessite que les fichiers XML soient accessibles publiquement
- La conversion équirectangulaire peut être lente pour de très grandes résolutions
- Certains sites peuvent avoir des protections anti-bot

## Licence

Ce script est fourni tel quel, utilisez-le à vos propres risques. Respectez les conditions d'utilisation des sites web dont vous téléchargez le contenu.
