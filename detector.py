"""
Système de détection et suivi des minions pour Minion Masters.
Ce module contient toute la logique OpenCV pour détecter, suivre et analyser les minions.
"""

import cv2
import numpy as np
import time
import mss
from typing import Dict, List, Tuple, Optional, Set

from minion import Minion
import config


class MinionDetector:
    """
    Détecteur principal pour les minions dans Minion Masters.
    Utilise OpenCV pour la détection de mouvement et le suivi d'objets.
    """
    
    def __init__(self):
        """Initialise le détecteur avec tous les paramètres nécessaires."""
        # Configuration de base
        self.frame_width = config.MONITOR["width"]
        self.frame_height = config.MONITOR["height"]
        
        # Système de détection de mouvement
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.MOG2_HISTORY,
            varThreshold=config.MOG2_VAR_THRESHOLD,
            detectShadows=config.MOG2_DETECT_SHADOWS
        )
        
        # Suivi des minions
        self.minions: Dict[int, Minion] = {}
        self.next_minion_id = 1
        
        # Zones d'exclusion (tours de maître, etc.)
        self.exclusion_zones = self._calculate_exclusion_zones()
        
        # Capture d'écran
        self.screen_capture = mss.mss()
        
        print(f"✅ Détecteur initialisé pour {self.frame_width}x{self.frame_height}")
        print(f"   Zones d'exclusion: {len(self.exclusion_zones)} zones configurées")
    
    def _calculate_exclusion_zones(self) -> List[Tuple[float, float, float]]:
        """
        Calcule les zones d'exclusion pour ignorer les détections près des structures fixes.
        
        Returns:
            List[Tuple[float, float, float]]: Liste de (center_x, center_y, radius) pour chaque zone
        """
        exclusion_radius = self.frame_width * config.EXCLUSION_RADIUS_RATIO
        center_y = self.frame_height * config.EXCLUSION_CENTER_Y_RATIO
        
        zones = [
            (exclusion_radius, center_y, exclusion_radius),                    # Zone gauche
            (self.frame_width - exclusion_radius, center_y, exclusion_radius) # Zone droite
        ]
        
        return zones
    
    def capture_screen(self) -> np.ndarray:
        """
        Capture l'écran selon la configuration définie.
        
        Returns:
            np.ndarray: Image capturée au format BGR
        """
        sct_img = self.screen_capture.grab(config.MONITOR)
        frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
        return frame
    
    def detect_movement(self, frame: np.ndarray) -> np.ndarray:
        """
        Détecte les zones de mouvement dans l'image.
        
        Args:
            frame: Image à analyser
            
        Returns:
            np.ndarray: Masque binaire des zones en mouvement
        """
        # Application du détecteur de fond
        mask = self.background_subtractor.apply(frame)
        
        # Lissage pour réduire le bruit
        mask = cv2.medianBlur(mask, config.MEDIAN_BLUR_KERNEL)
        
        return mask
    
    def find_contours(self, mask: np.ndarray) -> List[Tuple[float, float]]:
        """
        Trouve les contours dans le masque et extrait les positions des minions potentiels.
        
        Args:
            mask: Masque binaire des zones en mouvement
            
        Returns:
            List[Tuple[float, float]]: Liste des positions (x, y) des détections valides
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_positions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filtrage par taille
            if config.MINION_MIN_AREA < area < config.MINION_MAX_AREA:
                (x, y), _ = cv2.minEnclosingCircle(contour)
                
                # Vérification des zones d'exclusion
                if not self._is_in_exclusion_zone(x, y):
                    detected_positions.append((x, y))
        
        return detected_positions
    
    def _is_in_exclusion_zone(self, x: float, y: float) -> bool:
        """
        Vérifie si une position se trouve dans une zone d'exclusion.
        
        Args:
            x, y: Coordonnées à vérifier
            
        Returns:
            bool: True si la position est dans une zone d'exclusion
        """
        for center_x, center_y, radius in self.exclusion_zones:
            distance = np.hypot(x - center_x, y - center_y)
            if distance <= radius:
                return True
        return False
    
    def update_minion_tracking(self, detected_positions: List[Tuple[float, float]], 
                             frame: np.ndarray, mask: np.ndarray, current_time: float):
        """
        Met à jour le suivi des minions avec les nouvelles détections.
        
        Args:
            detected_positions: Positions détectées dans cette frame
            frame: Image actuelle
            mask: Masque de mouvement
            current_time: Timestamp actuel
        """
        used_detections: Set[int] = set()
        
        # 1. Association des détections aux minions existants
        self._associate_detections_to_existing_minions(
            detected_positions, used_detections, frame, mask, current_time
        )
        
        # 2. Création de nouveaux minions pour les détections non associées
        self._create_new_minions(
            detected_positions, used_detections, frame, mask, current_time
        )
        
        # 3. Mise à jour du statut des minions
        self._update_minion_status(current_time)
    
    def _associate_detections_to_existing_minions(self, detected_positions: List[Tuple[float, float]],
                                                used_detections: Set[int], frame: np.ndarray,
                                                mask: np.ndarray, current_time: float):
        """
        Associe les nouvelles détections aux minions existants.
        """
        for minion in self.minions.values():
            if not minion.active:
                continue
                
            best_match_idx = -1
            min_distance = config.REID_TOLERANCE
            best_similarity = 0
            
            # Recherche de la meilleure correspondance
            for i, (x, y) in enumerate(detected_positions):
                if i in used_detections:
                    continue
                
                # Distance euclidienne
                distance = np.hypot(minion.positions[-1][0] - x, minion.positions[-1][1] - y)
                
                # Vérification de la prédiction si disponible
                predicted_pos = minion.get_predicted_position_at_time(current_time)
                if predicted_pos:
                    pred_distance = np.hypot(predicted_pos[0] - x, predicted_pos[1] - y)
                    if pred_distance < config.PREDICTION_TOLERANCE:
                        distance *= 0.5  # Bonus pour les prédictions correctes
                
                # Similarité d'histogramme
                similarity = minion.similarity(frame, mask, (x, y))
                
                # Critères de sélection combinés
                if (distance < min_distance and 
                    similarity > config.HIST_SIMILARITY_THRESHOLD and
                    similarity > best_similarity):
                    min_distance = distance
                    best_match_idx = i
                    best_similarity = similarity
            
            # Mise à jour du minion si une correspondance est trouvée
            if best_match_idx != -1:
                x, y = detected_positions[best_match_idx]
                minion.update_position((x, y), frame, mask, current_time)
                used_detections.add(best_match_idx)
    
    def _create_new_minions(self, detected_positions: List[Tuple[float, float]],
                          used_detections: Set[int], frame: np.ndarray,
                          mask: np.ndarray, current_time: float):
        """
        Crée de nouveaux minions pour les détections non associées.
        """
        for i, (x, y) in enumerate(detected_positions):
            if i not in used_detections:
                # Vérification anti-spam (détection de sorts de masse)
                if self._is_mass_spawn_event(detected_positions, current_time):
                    continue
                
                # Création du nouveau minion
                new_minion = Minion(
                    id=self.next_minion_id,
                    position=(x, y),
                    frame=frame,
                    mask=mask,
                    frame_width=self.frame_width,
                    frame_height=self.frame_height,
                    timestamp=current_time
                )
                
                self.minions[self.next_minion_id] = new_minion
                self.next_minion_id += 1
    
    def _is_mass_spawn_event(self, detected_positions: List[Tuple[float, float]], 
                           current_time: float) -> bool:
        """
        Détecte si nous avons un événement de spawn massif (sort de zone).
        
        Args:
            detected_positions: Positions détectées
            current_time: Timestamp actuel
            
        Returns:
            bool: True s'il s'agit probablement d'un événement de spawn massif
        """
        return len(detected_positions) > config.MASS_SPAWN_THRESHOLD
    
    def _update_minion_status(self, current_time: float):
        """
        Met à jour le statut de tous les minions et nettoie les anciens.
        """
        minions_to_remove = []
        
        for minion_id, minion in self.minions.items():
            # Marquer comme inactif si pas vu récemment
            if current_time - minion.last_seen > config.MOVEMENT_MEMORY:
                minion.active = False
            
            # Validation du minion
            minion.validate_as_minion(current_time)
            
            # Suppression des très anciens minions
            if current_time - minion.last_seen > config.REID_MAX_TIME:
                minions_to_remove.append(minion_id)
        
        # Nettoyage
        for minion_id in minions_to_remove:
            del self.minions[minion_id]
    
    def get_active_minions(self) -> List[Minion]:
        """
        Retourne la liste des minions actifs et validés.
        
        Returns:
            List[Minion]: Liste des minions actifs
        """
        return [minion for minion in self.minions.values() 
                if minion.is_valid_minion and minion.active]
    
    def get_enemy_minions(self) -> List[Minion]:
        """
        Retourne la liste des minions ennemis identifiés.
        
        Returns:
            List[Minion]: Liste des minions ennemis
        """
        enemy_minions = []
        
        for minion in self.get_active_minions():
            strategy_info = minion.get_strategy_info()
            if strategy_info and strategy_info.get('likely_enemy', False):
                enemy_minions.append(minion)
        
        return enemy_minions
    
    def process_frame(self) -> Tuple[List[Minion], List[Minion]]:
        """
        Traite une frame complète: capture, détection, suivi.
        
        Returns:
            Tuple[List[Minion], List[Minion]]: (minions actifs, minions ennemis)
        """
        current_time = time.time()
        
        # 1. Capture d'écran
        frame = self.capture_screen()
        
        # 2. Détection de mouvement
        mask = self.detect_movement(frame)
        
        # 3. Extraction des contours
        detected_positions = self.find_contours(mask)
        
        # 4. Mise à jour du suivi
        self.update_minion_tracking(detected_positions, frame, mask, current_time)
        
        # 5. Retour des résultats
        active_minions = self.get_active_minions()
        enemy_minions = self.get_enemy_minions()
        
        return active_minions, enemy_minions
    
    def get_detection_stats(self) -> Dict[str, any]:
        """
        Retourne des statistiques sur l'état actuel de la détection.
        
        Returns:
            Dict: Statistiques de détection
        """
        active_count = len(self.get_active_minions())
        enemy_count = len(self.get_enemy_minions())
        total_count = len(self.minions)
        
        return {
            'total_minions': total_count,
            'active_minions': active_count,
            'enemy_minions': enemy_count,
            'inactive_minions': total_count - active_count,
            'next_id': self.next_minion_id
        }
    
    def cleanup(self):
        """Nettoie les ressources utilisées par le détecteur."""
        if hasattr(self, 'screen_capture'):
            self.screen_capture.close()
        print("🧹 Détecteur nettoyé.")


def create_detector() -> MinionDetector:
    """
    Factory function pour créer un détecteur configuré.
    
    Returns:
        MinionDetector: Instance configurée du détecteur
    """
    return MinionDetector()


# ==============================================================================
# --- FONCTIONS UTILITAIRES POUR LA DÉTECTION ---
# ==============================================================================

def analyze_detection_quality(minions: List[Minion]) -> Dict[str, float]:
    """
    Analyse la qualité des détections actuelles.
    
    Args:
        minions: Liste des minions à analyser
        
    Returns:
        Dict[str, float]: Métriques de qualité
    """
    if not minions:
        return {
            'average_lifetime': 0,
            'average_distance_traveled': 0,
            'valid_ratio': 0,
            'enemy_ratio': 0
        }
    
    total_lifetime = sum(m.timestamps[-1] - m.creation_time for m in minions)
    total_distance = sum(m.total_distance_traveled for m in minions)
    valid_count = sum(1 for m in minions if m.is_valid_minion)
    enemy_count = sum(1 for m in minions if m.get_strategy_info() and 
                     m.get_strategy_info().get('likely_enemy', False))
    
    return {
        'average_lifetime': total_lifetime / len(minions),
        'average_distance_traveled': total_distance / len(minions),
        'valid_ratio': valid_count / len(minions),
        'enemy_ratio': enemy_count / len(minions) if len(minions) > 0 else 0
    }


def filter_minions_by_position(minions: List[Minion], 
                             x_range: Optional[Tuple[float, float]] = None,
                             y_range: Optional[Tuple[float, float]] = None) -> List[Minion]:
    """
    Filtre les minions par position.
    
    Args:
        minions: Liste des minions à filtrer
        x_range: Range X (min, max) ou None
        y_range: Range Y (min, max) ou None
        
    Returns:
        List[Minion]: Minions dans la zone spécifiée
    """
    filtered = []
    
    for minion in minions:
        if not minion.positions:
            continue
            
        x, y = minion.positions[-1]
        
        if x_range and not (x_range[0] <= x <= x_range[1]):
            continue
        
        if y_range and not (y_range[0] <= y <= y_range[1]):
            continue
        
        filtered.append(minion)
    
    return filtered


def get_minions_in_danger_zone(minions: List[Minion], 
                              player_side: str,
                              frame_width: int, 
                              danger_threshold: float = 0.3) -> List[Minion]:
    """
    Identifie les minions ennemis dans la zone de danger.
    
    Args:
        minions: Liste des minions à analyser
        player_side: Côté du joueur ("left" ou "right")
        frame_width: Largeur de l'écran
        danger_threshold: Seuil de la zone de danger (ratio de l'écran)
        
    Returns:
        List[Minion]: Minions ennemis dans la zone de danger
    """
    danger_zone_x = frame_width * danger_threshold
    
    dangerous_minions = []
    
    for minion in minions:
        if not minion.positions:
            continue
            
        strategy_info = minion.get_strategy_info()
        if not strategy_info or not strategy_info.get('likely_enemy', False):
            continue
        
        x, y = minion.positions[-1]
        
        # Vérification selon le côté du joueur
        if player_side == "left" and x <= danger_zone_x:
            dangerous_minions.append(minion)
        elif player_side == "right" and x >= (frame_width - danger_zone_x):
            dangerous_minions.append(minion)
    
    return dangerous_minions
