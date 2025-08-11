"""
Fonctions utilitaires pour l'analyseur Minion Masters.
Contient les fonctions de conversion, géométrie et manipulation d'images.
"""

import numpy as np
import cv2

from config import EXCLUSION_RADIUS_RATIO, EXCLUSION_CENTER_Y_RATIO


def ratio_to_pixels(ratio_list, frame_width, frame_height):
    """
    Convertit une liste de positions en ratios (0.0-1.0) vers des coordonnées en pixels.
    
    Args:
        ratio_list (list): Liste de tuples (ratio_x, ratio_y)
        frame_width (int): Largeur de l'écran en pixels
        frame_height (int): Hauteur de l'écran en pixels
        
    Returns:
        list: Liste de tuples (x, y) en pixels
        
    Example:
        >>> ratio_to_pixels([(0.5, 0.5), (0.25, 0.75)], 1920, 1080)
        [(960, 540), (480, 810)]
    """
    return [(int(rx * frame_width), int(ry * frame_height)) for rx, ry in ratio_list]


def pixels_to_ratio(pixel_list, frame_width, frame_height):
    """
    Convertit une liste de positions en pixels vers des ratios (0.0-1.0).
    
    Args:
        pixel_list (list): Liste de tuples (x, y) en pixels
        frame_width (int): Largeur de l'écran en pixels
        frame_height (int): Hauteur de l'écran en pixels
        
    Returns:
        list: Liste de tuples (ratio_x, ratio_y)
        
    Example:
        >>> pixels_to_ratio([(960, 540), (480, 810)], 1920, 1080)
        [(0.5, 0.5), (0.25, 0.75)]
    """
    return [(x / frame_width, y / frame_height) for x, y in pixel_list]


def distance_2d(point1, point2):
    """
    Calcule la distance euclidienne entre deux points 2D.
    
    Args:
        point1 (tuple): Premier point (x1, y1)
        point2 (tuple): Deuxième point (x2, y2)
        
    Returns:
        float: Distance euclidienne
        
    Example:
        >>> distance_2d((0, 0), (3, 4))
        5.0
    """
    return np.hypot(point1[0] - point2[0], point1[1] - point2[1])


def is_in_exclusion_zone(x, y, exclusion_zones):
    """
    Vérifie si un point se trouve dans une zone d'exclusion.
    Les zones d'exclusion sont utilisées pour ignorer les détections près des structures fixes.
    
    Args:
        x (float): Coordonnée X du point
        y (float): Coordonnée Y du point
        exclusion_zones (list): Liste de tuples (center_x, center_y, radius)
        
    Returns:
        bool: True si le point est dans une zone d'exclusion
    """
    for center_x, center_y, radius in exclusion_zones:
        if distance_2d((x, y), (center_x, center_y)) <= radius:
            return True
    return False


def create_exclusion_zones(frame_width, frame_height):
    """
    Crée les zones d'exclusion standard basées sur les dimensions de l'écran.
    Les zones d'exclusion sont placées près des tours de maître pour éviter les fausses détections.
    
    Args:
        frame_width (int): Largeur de l'écran en pixels
        frame_height (int): Hauteur de l'écran en pixels
        
    Returns:
        list: Liste de zones d'exclusion (center_x, center_y, radius)
    """
    exclusion_radius = frame_width * EXCLUSION_RADIUS_RATIO
    center_y = frame_height * EXCLUSION_CENTER_Y_RATIO
    
    return [
        (exclusion_radius, center_y, exclusion_radius),  # Zone gauche
        (frame_width - exclusion_radius, center_y, exclusion_radius)  # Zone droite
    ]


def point_in_rect(point, rect):
    """
    Vérifie si un point se trouve dans un rectangle.
    
    Args:
        point (tuple): Point (x, y)
        rect (tuple): Rectangle (x, y, width, height)
        
    Returns:
        bool: True si le point est dans le rectangle
    """
    x, y = point
    rect_x, rect_y, rect_width, rect_height = rect
    
    return (rect_x <= x <= rect_x + rect_width and 
            rect_y <= y <= rect_y + rect_height)


def clamp(value, min_value, max_value):
    """
    Limite une valeur entre min et max.
    
    Args:
        value (float): Valeur à limiter
        min_value (float): Valeur minimum
        max_value (float): Valeur maximum
        
    Returns:
        float: Valeur limitée
        
    Example:
        >>> clamp(15, 0, 10)
        10
        >>> clamp(-5, 0, 10)
        0
        >>> clamp(7, 0, 10)
        7
    """
    return max(min_value, min(value, max_value))


def clamp_position(position, frame_width, frame_height):
    """
    Limite une position aux dimensions de l'écran.
    
    Args:
        position (tuple): Position (x, y)
        frame_width (int): Largeur maximum
        frame_height (int): Hauteur maximum
        
    Returns:
        tuple: Position limitée (x, y)
    """
    x, y = position
    return (
        clamp(x, 0, frame_width - 1),
        clamp(y, 0, frame_height - 1)
    )


def get_roi_safe(image, center, size):
    """
    Extrait une région d'intérêt (ROI) de manière sécurisée.
    Gère automatiquement les cas où la ROI dépasse les limites de l'image.
    
    Args:
        image (np.array): Image source
        center (tuple): Centre de la ROI (x, y)
        size (int): Taille de la ROI (carré de côté size*2)
        
    Returns:
        tuple: (roi, roi_mask, valid) où :
            - roi: Région d'intérêt extraite
            - roi_mask: Masque correspondant (si fourni)
            - valid: True si la ROI est valide
    """
    if image is None or image.size == 0:
        return None, None, False
        
    x, y = int(center[0]), int(center[1])
    height, width = image.shape[:2]
    
    # Calculer les limites de la ROI
    x1, y1 = max(0, x - size), max(0, y - size)
    x2, y2 = min(width, x + size), min(height, y + size)
    
    # Vérifier que la ROI est valide
    if x2 <= x1 or y2 <= y1:
        return None, None, False
    
    roi = image[y1:y2, x1:x2]
    return roi, None, roi.size > 0


def calculate_movement_vector(positions, timestamps, window_size=5):
    """
    Calcule le vecteur de mouvement moyen sur une fenêtre glissante.
    
    Args:
        positions (deque): Historique des positions
        timestamps (deque): Historique des timestamps
        window_size (int): Taille de la fenêtre pour le calcul
        
    Returns:
        tuple: Vecteur de mouvement (dx, dy) en pixels/seconde
    """
    if len(positions) < 2 or len(timestamps) < 2:
        return (0, 0)
    
    # Utiliser les dernières positions dans la fenêtre
    recent_positions = list(positions)[-window_size:]
    recent_timestamps = list(timestamps)[-window_size:]
    
    if len(recent_positions) < 2:
        return (0, 0)
    
    # Calculer le mouvement total sur la période
    total_time = recent_timestamps[-1] - recent_timestamps[0]
    if total_time <= 0:
        return (0, 0)
    
    total_dx = recent_positions[-1][0] - recent_positions[0][0]
    total_dy = recent_positions[-1][1] - recent_positions[0][1]
    
    return (total_dx / total_time, total_dy / total_time)


def smooth_positions(positions, window_size=3):
    """
    Lisse une séquence de positions en utilisant une moyenne mobile.
    
    Args:
        positions (list): Liste de positions (x, y)
        window_size (int): Taille de la fenêtre de lissage
        
    Returns:
        list: Positions lissées
    """
    if len(positions) < window_size:
        return list(positions)
    
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(positions)):
        # Déterminer les limites de la fenêtre
        start = max(0, i - half_window)
        end = min(len(positions), i + half_window + 1)
        
        # Calculer la moyenne sur la fenêtre
        window_positions = positions[start:end]
        avg_x = sum(pos[0] for pos in window_positions) / len(window_positions)
        avg_y = sum(pos[1] for pos in window_positions) / len(window_positions)
        
        smoothed.append((avg_x, avg_y))
    
    return smoothed


def calculate_speed(position1, position2, time1, time2):
    """
    Calcule la vitesse entre deux positions.
    
    Args:
        position1 (tuple): Première position (x, y)
        position2 (tuple): Deuxième position (x, y)
        time1 (float): Timestamp de la première position
        time2 (float): Timestamp de la deuxième position
        
    Returns:
        float: Vitesse en pixels/seconde
    """
    time_diff = time2 - time1
    if time_diff <= 0:
        return 0
    
    distance = distance_2d(position1, position2)
    return distance / time_diff


def filter_detections_by_area(contours, min_area, max_area):
    """
    Filtre une liste de contours par leur aire.
    
    Args:
        contours (list): Liste de contours OpenCV
        min_area (float): Aire minimum
        max_area (float): Aire maximum
        
    Returns:
        list: Positions des centres des contours valides (x, y)
    """
    valid_positions = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Calculer le centre du contour
            (x, y), _ = cv2.minEnclosingCircle(contour)
            valid_positions.append((x, y))
    
    return valid_positions


def create_circular_mask(image_shape, center, radius):
    """
    Crée un masque circulaire.
    
    Args:
        image_shape (tuple): Forme de l'image (height, width)
        center (tuple): Centre du cercle (x, y)
        radius (float): Rayon du cercle
        
    Returns:
        np.array: Masque binaire (255 à l'intérieur du cercle, 0 à l'extérieur)
    """
    height, width = image_shape[:2]
    y, x = np.ogrid[:height, :width]
    center_x, center_y = center
    
    mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
    return mask.astype(np.uint8) * 255


def get_screen_quadrant(position, frame_width, frame_height):
    """
    Détermine dans quel quadrant de l'écran se trouve une position.
    
    Args:
        position (tuple): Position (x, y)
        frame_width (int): Largeur de l'écran
        frame_height (int): Hauteur de l'écran
        
    Returns:
        str: Quadrant ('top-left', 'top-right', 'bottom-left', 'bottom-right')
    """
    x, y = position
    mid_x = frame_width / 2
    mid_y = frame_height / 2
    
    if x < mid_x:
        return 'top-left' if y < mid_y else 'bottom-left'
    else:
        return 'top-right' if y < mid_y else 'bottom-right'


def interpolate_position(pos1, pos2, ratio):
    """
    Interpole entre deux positions.
    
    Args:
        pos1 (tuple): Première position (x, y)
        pos2 (tuple): Deuxième position (x, y)
        ratio (float): Ratio d'interpolation (0.0 = pos1, 1.0 = pos2)
        
    Returns:
        tuple: Position interpolée (x, y)
    """
    ratio = clamp(ratio, 0.0, 1.0)
    x = pos1[0] + (pos2[0] - pos1[0]) * ratio
    y = pos1[1] + (pos2[1] - pos1[1]) * ratio
    return (x, y)


def format_time_duration(seconds):
    """
    Formate une durée en secondes en format lisible.
    
    Args:
        seconds (float): Durée en secondes
        
    Returns:
        str: Durée formatée
        
    Example:
        >>> format_time_duration(125.5)
        '2m 5.5s'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:.1f}s"
    
    hours = int(minutes // 60)
    remaining_minutes = minutes % 60
    return f"{hours}h {remaining_minutes}m {remaining_seconds:.1f}s"


def debug_print_minion_info(minion):
    """
    Affiche des informations de debug sur un minion.
    Utile pour le développement et le débogage.
    
    Args:
        minion (Minion): Instance de minion à analyser
    """
    if not minion:
        print("❌ Minion None")
        return
    
    current_pos = minion.positions[-1] if minion.positions else "Unknown"
    
    print(f"🔍 Minion #{minion.id}")
    print(f"   Position: {current_pos}")
    print(f"   Direction: {minion.general_direction}")
    print(f"   Côté: {minion.current_side}")
    print(f"   Valide: {'✅' if minion.is_valid_minion else '❌'}")
    print(f"   Distance parcourue: {minion.total_distance_traveled:.1f}px")
    print(f"   Vitesse max: {minion.max_speed:.1f}px/s")
    
    strategy_info = minion.get_strategy_info()
    if strategy_info:
        print(f"   Ennemi probable: {'⚔️' if strategy_info['likely_enemy'] else '🛡️'}")
