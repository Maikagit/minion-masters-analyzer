"""
Classe Minion pour le suivi et l'analyse des minions dans Minion Masters.
Gère la détection, le suivi, la prédiction de trajectoire et la classification.
"""

import cv2
import numpy as np
from collections import deque
import time

from config import (
    MIN_LIFETIME, MIN_DISTANCE_TRAVELED, STATIONARY_THRESHOLD,
    MIN_POINTS_FOR_PREDICTION, PREDICTION_TIME, PLAYER_SIDE
)


class Minion:
    """
    Représente un minion détecté et suivi dans le jeu.
    
    Gère :
    - Le suivi de position dans le temps
    - La prédiction de trajectoire
    - La classification ennemi/allié
    - La validation comme vrai minion
    """
    
    def __init__(self, id, position, frame, mask, frame_width, frame_height, timestamp):
        """
        Initialise un nouveau minion.
        
        Args:
            id (int): Identifiant unique du minion
            position (tuple): Position initiale (x, y)
            frame (np.array): Image de la frame actuelle
            mask (np.array): Masque de détection
            frame_width (int): Largeur de l'écran
            frame_height (int): Hauteur de l'écran
            timestamp (float): Timestamp de création
        """
        # Identité et suivi
        self.id = id
        self.positions = deque(maxlen=80)  # Historique des positions
        self.timestamps = deque(maxlen=80)  # Historique des timestamps
        self.positions.append(position)
        self.timestamps.append(timestamp)
        self.creation_time = timestamp
        self.last_seen = timestamp
        self.active = True
        
        # Signature visuelle pour le ré-identification
        self.hist = self.compute_hist(frame, mask, position)
        
        # Données spatiales et stratégiques
        self.spawn_side = "left" if position[0] < frame_width / 2 else "right"
        self.current_side = self.spawn_side
        self.general_direction = "unknown"
        self.optimal_bridge = "unknown"
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.total_distance_traveled = 0
        self.original_classification = None
        
        # Validation et métriques de mouvement
        self.is_valid_minion = False
        self.consecutive_stationary_frames = 0
        self.max_speed = 0
        
        # Prédiction de trajectoire
        self.predicted_positions = []
        self.velocity = (0, 0)
        self.acceleration = (0, 0)

    def compute_hist(self, frame, mask, position):
        """
        Calcule l'histogramme de couleur autour de la position du minion.
        Utilisé pour la ré-identification.
        
        Args:
            frame (np.array): Image de la frame
            mask (np.array): Masque de détection
            position (tuple): Position (x, y) du minion
            
        Returns:
            np.array: Histogramme HSV normalisé
        """
        x, y = int(position[0]), int(position[1])
        size = 25  # Taille de la région d'intérêt autour du minion
        
        # Définir la ROI en s'assurant qu'elle reste dans les limites de l'image
        x1, y1 = max(0, x - size), max(0, y - size)
        x2, y2 = min(frame.shape[1], x + size), min(frame.shape[0], y + size)
        
        # Vérifications de sécurité
        if x2 <= x1 or y2 <= y1:
            return np.zeros((30, 32))
            
        roi = frame[y1:y2, x1:x2]
        roi_mask = mask[y1:y2, x1:x2]
        
        if roi.size == 0 or roi_mask.size == 0:
            return np.zeros((30, 32))
            
        # Conversion en HSV et calcul de l'histogramme
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], roi_mask, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def calculate_prediction(self, current_time):
        """
        Calcule la prédiction de trajectoire basée sur l'historique de mouvement.
        Utilise la vitesse et l'accélération pour prédire les positions futures.
        
        Args:
            current_time (float): Timestamp actuel
        """
        if len(self.positions) < MIN_POINTS_FOR_PREDICTION:
            return

        # Utiliser les 10 dernières positions pour plus de stabilité
        recent_positions = list(self.positions)[-10:]
        recent_timestamps = list(self.timestamps)[-10:]
        
        # Calcul de la vitesse moyenne
        if len(recent_positions) >= 3:
            time_span = recent_timestamps[-1] - recent_timestamps[0]
            if time_span > 0:
                dx = recent_positions[-1][0] - recent_positions[0][0]
                dy = recent_positions[-1][1] - recent_positions[0][1]
                self.velocity = (dx / time_span, dy / time_span)
                
                # Calcul de l'accélération si on a assez de données
                if len(recent_positions) >= 6:
                    mid_idx = len(recent_positions) // 2
                    time_span_1 = recent_timestamps[mid_idx] - recent_timestamps[0]
                    time_span_2 = recent_timestamps[-1] - recent_timestamps[mid_idx]
                    
                    if time_span_1 > 0 and time_span_2 > 0:
                        # Vitesse sur la première moitié
                        vel1_x = (recent_positions[mid_idx][0] - recent_positions[0][0]) / time_span_1
                        vel1_y = (recent_positions[mid_idx][1] - recent_positions[0][1]) / time_span_1
                        
                        # Vitesse sur la seconde moitié
                        vel2_x = (recent_positions[-1][0] - recent_positions[mid_idx][0]) / time_span_2
                        vel2_y = (recent_positions[-1][1] - recent_positions[mid_idx][1]) / time_span_2
                        
                        # Calcul de l'accélération
                        accel_x = (vel2_x - vel1_x) / time_span_2
                        accel_y = (vel2_y - vel1_y) / time_span_2
                        self.acceleration = (accel_x, accel_y)
        
        # Génération des positions prédites
        self.predicted_positions = []
        if abs(self.velocity[0]) > 1 or abs(self.velocity[1]) > 1:  # Seulement si le minion bouge
            last_pos = self.positions[-1]
            for dt in np.arange(0.1, PREDICTION_TIME, 0.1):
                # Équation cinématique : position = position_initiale + vitesse*temps + 0.5*accélération*temps²
                pred_x = last_pos[0] + self.velocity[0] * dt + 0.5 * self.acceleration[0] * dt * dt
                pred_y = last_pos[1] + self.velocity[1] * dt + 0.5 * self.acceleration[1] * dt * dt
                
                # S'assurer que la prédiction reste dans les limites de l'écran
                pred_x = max(0, min(self.frame_width, pred_x))
                pred_y = max(0, min(self.frame_height, pred_y))
                
                self.predicted_positions.append((pred_x, pred_y, current_time + dt))

    def get_predicted_position_at_time(self, target_time):
        """
        Obtient la position prédite à un moment donné.
        
        Args:
            target_time (float): Timestamp cible
            
        Returns:
            tuple: Position prédite (x, y) ou None si impossible
        """
        # Fallback si pas assez de données
        if not self.predicted_positions or len(self.timestamps) < 2:
            if len(self.positions) > 0:
                return self.positions[-1]
            return None

        # Extrapolation si le temps est dans le futur
        last_pos = self.positions[-1]
        dt = target_time - self.timestamps[-1]
        pred_x = last_pos[0] + self.velocity[0] * dt + 0.5 * self.acceleration[0] * dt * dt
        pred_y = last_pos[1] + self.velocity[1] * dt + 0.5 * self.acceleration[1] * dt * dt
        return (pred_x, pred_y)

    def calculate_direction_and_bridge(self):
        """
        Analyse la direction générale du minion et détermine le pont optimal.
        Met à jour les attributs general_direction et optimal_bridge.
        """
        if len(self.positions) < 5:
            return
        
        # Déterminer la direction générale (gauche vers droite ou vice versa)
        start_pos_x = self.positions[0][0]
        current_pos_x = self.positions[-1][0]
        horizontal_movement = current_pos_x - start_pos_x

        # Seuil de mouvement significatif
        if abs(horizontal_movement) > 50:
            self.general_direction = "right" if horizontal_movement > 0 else "left"
        
        # Classification initiale comme ennemi (fait une seule fois)
        if self.original_classification is None and self.general_direction != "unknown":
            self.original_classification = self.classify_as_enemy()

        # Mise à jour du côté actuel
        self.current_side = "left" if self.positions[-1][0] < self.frame_width / 2 else "right"
        
        # Déterminer le pont optimal basé sur la position verticale
        current_y = self.positions[-1][1]
        vertical_ratio = current_y / self.frame_height
        vertical_trend = self.positions[-1][1] - self.positions[0][1]
        
        if vertical_ratio < 0.35:
            self.optimal_bridge = "top"
        elif vertical_ratio > 0.65:
            self.optimal_bridge = "bottom"
        else:
            # Position centrale, utiliser la tendance de mouvement
            self.optimal_bridge = "top" if vertical_trend <= 0 else "bottom"

    def classify_as_enemy(self):
        """
        Détermine si ce minion est un ennemi basé sur sa direction de mouvement.
        
        Returns:
            bool: True si c'est probablement un ennemi
        """
        if self.general_direction == "unknown":
            return False
            
        # Logique basée sur le côté du joueur
        if PLAYER_SIDE == "left":
            # Si le joueur est à gauche, les ennemis vont vers la droite
            return self.general_direction == "right"
        else:  # PLAYER_SIDE == "right"
            # Si le joueur est à droite, les ennemis vont vers la gauche
            return self.general_direction == "left"

    def validate_as_minion(self, current_time):
        """
        Valide que cet objet est effectivement un minion basé sur plusieurs critères.
        
        Args:
            current_time (float): Timestamp actuel
            
        Returns:
            bool: True si validé comme minion
        """
        lifetime = current_time - self.creation_time
        
        # Critères de validation
        self.is_valid_minion = (
            lifetime > MIN_LIFETIME and  # Doit exister assez longtemps
            self.total_distance_traveled > MIN_DISTANCE_TRAVELED and  # Doit avoir bougé
            self.consecutive_stationary_frames < 30 and  # Ne doit pas être immobile trop longtemps
            self.max_speed < 200  # Vitesse réaliste (px/s)
        )
        
        return self.is_valid_minion

    def update_position(self, position, frame, mask, timestamp):
        """
        Met à jour la position du minion et recalcule toutes les métriques.
        
        Args:
            position (tuple): Nouvelle position (x, y)
            frame (np.array): Image de la frame
            mask (np.array): Masque de détection
            timestamp (float): Timestamp de la mise à jour
        """
        # Calcul des métriques de mouvement
        last_pos = self.positions[-1]
        distance = np.hypot(position[0] - last_pos[0], position[1] - last_pos[1])
        time_diff = timestamp - self.timestamps[-1]
        
        if time_diff > 0:
            speed = distance / time_diff
            self.max_speed = max(self.max_speed, speed)
            
            # Suivi des frames stationnaires
            if distance < STATIONARY_THRESHOLD:
                self.consecutive_stationary_frames += 1
            else:
                self.consecutive_stationary_frames = 0
        
        # Mise à jour des données
        self.total_distance_traveled += distance
        self.positions.append(position)
        self.timestamps.append(timestamp)
        self.last_seen = timestamp
        self.active = True
        
        # Mise à jour de la signature visuelle
        self.hist = self.compute_hist(frame, mask, position)
        
        # Recalcul des analyses stratégiques
        self.calculate_direction_and_bridge()
        self.calculate_prediction(timestamp)

    def similarity(self, frame, mask, position):
        """
        Calcule la similarité entre ce minion et une nouvelle détection.
        Utilisé pour la ré-identification.
        
        Args:
            frame (np.array): Image de la frame
            mask (np.array): Masque de détection
            position (tuple): Position de la nouvelle détection
            
        Returns:
            float: Score de similarité (0-1, plus haut = plus similaire)
        """
        hist = self.compute_hist(frame, mask, position)
        if hist is None or self.hist is None:
            return 0
        return cv2.compareHist(self.hist, hist, cv2.HISTCMP_CORREL)

    def get_strategy_info(self):
        """
        Retourne les informations stratégiques sur ce minion.
        
        Returns:
            dict: Informations stratégiques ou None si le minion n'est pas validé
        """
        if not self.is_valid_minion:
            return None
            
        # Utiliser la classification originale si disponible, sinon recalculer
        is_enemy = (self.original_classification 
                   if self.original_classification is not None 
                   else self.classify_as_enemy())
        
        return {
            'likely_enemy': is_enemy,
            'direction': self.general_direction,
            'optimal_bridge': self.optimal_bridge,
            'current_side': self.current_side,
            'velocity': self.velocity,
            'predicted_positions': self.predicted_positions
        }
