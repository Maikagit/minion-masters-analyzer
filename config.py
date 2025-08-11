"""
Configuration centrale pour l'analyseur Minion Masters.
Tous les paramètres de l'application sont définis ici.
"""

# ==============================================================================
# --- AFFICHAGE ET SUPERPOSITION GRAPHIQUE ---
# ==============================================================================

# Couleur qui sera rendue transparente dans la superposition Tkinter
TRANSPARENT_COLOR = 'magenta'

# ==============================================================================
# --- CAPTURE D'ÉCRAN ET PARAMÈTRES DE JEU ---
# ==============================================================================

# Configuration de la capture d'écran (plein écran 1920x1080)
MONITOR = {
    "top": 0, 
    "left": 0, 
    "width": 1920, 
    "height": 1080
}

# Côté du joueur - détermine la logique de classification des ennemis
# Valeurs possibles: "left" ou "right"
PLAYER_SIDE = "left"

# ==============================================================================
# --- STRATÉGIE DE PLACEMENT ---
# ==============================================================================

# Seuil pour activer la stratégie défensive (nombre d'ennemis proches)
SAFE_ZONE_THRESHOLD = 2

# Positions stratégiques définies en ratios (0.0 à 1.0)
# Ces ratios seront convertis en pixels selon la résolution d'écran

# Positions défensives - pour protéger sa base
DEFENSIVE_POSITIONS_RATIOS = [
    (0.1021, 0.412),   # Position défensive 1
    (0.1432, 0.1731)   # Position défensive 2
]

# Positions offensives - pour attaquer l'ennemi
OFFENSIVE_POSITIONS_RATIOS = [
    (0.2, 0.163),      # Position offensive 1
    (0.2057, 0.2787)   # Position offensive 2
]

# Positions centrales - pour contrôler le milieu
CENTRAL_POSITIONS_RATIOS = [
    (0.1729, 0.4176),  # Position centrale 1
    (0.2927, 0.1481),  # Position centrale 2
    (0.2807, 0.4491)   # Position centrale 3
]

# Toutes les positions principales combinées
POSITIONS_PRINCIPALES_RATIOS = (
    DEFENSIVE_POSITIONS_RATIOS + 
    OFFENSIVE_POSITIONS_RATIOS + 
    CENTRAL_POSITIONS_RATIOS
)

# ==============================================================================
# --- PARAMÈTRES DE DÉTECTION ET SUIVI ---
# ==============================================================================

# Taille des minions détectés (en pixels²)
MINION_MIN_AREA = 30
MINION_MAX_AREA = 500

# Paramètres de mémoire et suivi
MOVEMENT_MEMORY = 5.0        # Temps en secondes avant de considérer un minion inactif
REID_TOLERANCE = 80          # Distance maximale pour ré-associer un minion (pixels)
REID_MAX_TIME = 10.0         # Temps maximum pour tenter de ré-identifier un minion perdu (secondes)
HIST_SIMILARITY_THRESHOLD = 0.3  # Seuil de similarité d'histogramme pour la ré-association

# ==============================================================================
# --- ZONES D'EXCLUSION ---
# ==============================================================================

# Pour ignorer les détections près des tours de maître et autres structures fixes
EXCLUSION_RADIUS_RATIO = 0.10    # Rayon d'exclusion en ratio de la largeur d'écran
EXCLUSION_CENTER_Y_RATIO = 0.36  # Position Y du centre des zones d'exclusion

# ==============================================================================
# --- FILTRES DE VALIDATION ---
# ==============================================================================

# Critères pour valider qu'un objet détecté est un "vrai" minion
MIN_LIFETIME = 0.8              # Doit exister pendant au moins 0.8 seconde
MIN_DISTANCE_TRAVELED = 40      # Doit avoir parcouru au moins 40 pixels
MASS_SPAWN_THRESHOLD = 20       # Seuil pour détecter des effets de sort de masse
STATIONARY_THRESHOLD = 10       # Distance de mouvement sous laquelle un minion est considéré immobile
GAME_START_BUFFER = 5           # Temps avant que les filtres stricts ne s'activent (secondes)

# ==============================================================================
# --- PRÉDICTION ET STRATÉGIE AVANCÉE ---
# ==============================================================================

# Paramètres de prédiction de trajectoire
PREDICTION_TIME = 5.0              # Prédiction de trajectoire sur 5 secondes
PREDICTION_TOLERANCE = 120         # Tolérance pour la ré-identification basée sur la prédiction (pixels)
MIN_POINTS_FOR_PREDICTION = 5      # Nombre minimum de points de données avant de prédire
PLACEMENT_PREDICTION_TIME = 1.5    # Anticiper de 1.5 seconde pour placer un contre
PLACEMENT_OFFSET_X = 50            # Décalage en pixels pour placer le minion devant l'ennemi

# ==============================================================================
# --- PARAMÈTRES DE PERFORMANCE ---
# ==============================================================================

# Délai entre les frames pour éviter de surcharger le CPU (secondes)
FRAME_DELAY = 0.01

# Paramètres du détecteur de mouvement (MOG2)
MOG2_HISTORY = 500
MOG2_VAR_THRESHOLD = 30
MOG2_DETECT_SHADOWS = False

# Paramètres de lissage
MEDIAN_BLUR_KERNEL = 7

# ==============================================================================
# --- PARAMÈTRES D'AFFICHAGE ---
# ==============================================================================

# Rayon du cercle de suggestion affiché sur l'overlay (pixels)
SUGGESTION_CIRCLE_RADIUS = 25

# Couleur et épaisseur du cercle de suggestion
SUGGESTION_CIRCLE_COLOR = 'lime'
SUGGESTION_CIRCLE_WIDTH = 4
