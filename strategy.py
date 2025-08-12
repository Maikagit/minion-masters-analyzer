"""
Module de stratégie pour l'analyseur Minion Masters.
Gère la logique de placement optimal basée sur l'analyse des menaces ennemies.
"""

import random
import time
from typing import List, Optional, Tuple

from config import (
    PLAYER_SIDE, SAFE_ZONE_THRESHOLD, PLACEMENT_PREDICTION_TIME, PLACEMENT_OFFSET_X,
    DEFENSIVE_POSITIONS_RATIOS, OFFENSIVE_POSITIONS_RATIOS, CENTRAL_POSITIONS_RATIOS
)
from utils import ratio_to_pixels, distance_2d


class StrategyAnalyzer:
    """
    Analyseur stratégique qui détermine le meilleur placement de minions
    basé sur l'analyse des menaces ennemies et de la situation tactique.
    """
    
    def __init__(self, frame_width: int, frame_height: int):
        """
        Initialise l'analyseur stratégique.
        
        Args:
            frame_width (int): Largeur de l'écran en pixels
            frame_height (int): Hauteur de l'écran en pixels
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Conversion des positions de ratios vers pixels
        self.defensive_positions = ratio_to_pixels(
            DEFENSIVE_POSITIONS_RATIOS, frame_width, frame_height
        )
        self.offensive_positions = ratio_to_pixels(
            OFFENSIVE_POSITIONS_RATIOS, frame_width, frame_height
        )
        self.central_positions = ratio_to_pixels(
            CENTRAL_POSITIONS_RATIOS, frame_width, frame_height
        )
        
        # Historique pour l'adaptation stratégique
        self.strategy_history = []
        self.last_placement_time = 0
        self.consecutive_same_strategy = 0
        self.current_strategy_type = None

    def analyze_threat_distribution(self, enemy_minions: List) -> dict:
        """
        Analyse la distribution des menaces ennemies sur l'écran.
        
        Args:
            enemy_minions (List): Liste des minions ennemis validés
            
        Returns:
            dict: Analyse des menaces avec zones et intensités
        """
        threat_analysis = {
            'left_threat': 0,
            'right_threat': 0,
            'top_threat': 0,
            'bottom_threat': 0,
            'center_threat': 0,
            'total_enemies': len(enemy_minions),
            'average_position': None,
            'threat_clusters': []
        }
        
        if not enemy_minions:
            return threat_analysis
        
        # Seuils pour les zones
        left_threshold = self.frame_width * 0.4
        right_threshold = self.frame_width * 0.6
        top_threshold = self.frame_height * 0.35
        bottom_threshold = self.frame_height * 0.65
        
        total_x, total_y = 0, 0
        
        for minion in enemy_minions:
            if not minion.positions:
                continue
                
            x, y = minion.positions[-1]
            total_x += x
            total_y += y
            
            # Analyse horizontale
            if x < left_threshold:
                threat_analysis['left_threat'] += 1
            elif x > right_threshold:
                threat_analysis['right_threat'] += 1
            else:
                threat_analysis['center_threat'] += 1
            
            # Analyse verticale
            if y < top_threshold:
                threat_analysis['top_threat'] += 1
            elif y > bottom_threshold:
                threat_analysis['bottom_threat'] += 1
        
        # Position moyenne des ennemis
        if enemy_minions:
            threat_analysis['average_position'] = (
                total_x / len(enemy_minions),
                total_y / len(enemy_minions)
            )
        
        return threat_analysis

    def calculate_optimal_placement(self, enemy_minions: List) -> Optional[Tuple[int, int]]:
        """
        Calcule le placement optimal basé sur l'analyse des minions ennemis.
        
        Args:
            enemy_minions (List): Liste des minions ennemis validés
            
        Returns:
            Optional[Tuple[int, int]]: Position optimale (x, y) ou None si aucune
        """
        if not enemy_minions:
            return self._get_default_placement()
        
        # Analyse de la situation tactique
        threat_analysis = self.analyze_threat_distribution(enemy_minions)
        strategy_type = self._determine_strategy_type(threat_analysis)
        
        # Sélection de la position basée sur la stratégie
        position = self._select_position_by_strategy(strategy_type, threat_analysis)
        
        # Mise à jour de l'historique stratégique
        self._update_strategy_history(strategy_type)
        
        return position

    def _determine_strategy_type(self, threat_analysis: dict) -> str:
        """
        Détermine le type de stratégie à adopter basé sur l'analyse des menaces.
        
        Args:
            threat_analysis (dict): Résultats de l'analyse des menaces
            
        Returns:
            str: Type de stratégie ('defensive', 'offensive', 'central')
        """
        left_threat = threat_analysis['left_threat']
        right_threat = threat_analysis['right_threat']
        total_enemies = threat_analysis['total_enemies']
        
        # Logique stratégique basée sur le côté du joueur
        if PLAYER_SIDE == "left":
            # Joueur à gauche
            if left_threat >= SAFE_ZONE_THRESHOLD:
                # Menace proche - stratégie défensive
                return 'defensive'
            elif right_threat == 0 and total_enemies > 0:
                # Pas d'ennemis à droite - stratégie offensive
                return 'offensive'
            elif total_enemies >= 4:
                # Beaucoup d'ennemis - position centrale pour contrôler
                return 'central'
            else:
                # Situation équilibrée - stratégie adaptative
                return self._adaptive_strategy(threat_analysis)
        
        else:  # PLAYER_SIDE == "right"
            # Joueur à droite - logique inversée
            if right_threat >= SAFE_ZONE_THRESHOLD:
                return 'defensive'
            elif left_threat == 0 and total_enemies > 0:
                return 'offensive'
            elif total_enemies >= 4:
                return 'central'
            else:
                return self._adaptive_strategy(threat_analysis)

    def _adaptive_strategy(self, threat_analysis: dict) -> str:
        """
        Détermine une stratégie adaptative basée sur l'historique et la situation.
        
        Args:
            threat_analysis (dict): Analyse des menaces
            
        Returns:
            str: Type de stratégie adaptée
        """
        # Éviter de répéter la même stratégie trop souvent
        if (self.consecutive_same_strategy >= 3 and 
            self.current_strategy_type in ['defensive', 'offensive']):
            return 'central'
        
        # Stratégie basée sur la position moyenne des ennemis
        if threat_analysis['average_position']:
            avg_x = threat_analysis['average_position'][0]
            center_x = self.frame_width / 2
            
            if PLAYER_SIDE == "left":
                if avg_x < center_x:
                    return 'defensive'  # Ennemis proches
                else:
                    return 'offensive'  # Ennemis loin
            else:
                if avg_x > center_x:
                    return 'defensive'  # Ennemis proches
                else:
                    return 'offensive'  # Ennemis loin
        
        return 'central'  # Fallback

    def _select_position_by_strategy(self, strategy_type: str, threat_analysis: dict) -> Tuple[int, int]:
        """
        Sélectionne une position spécifique basée sur le type de stratégie.
        
        Args:
            strategy_type (str): Type de stratégie
            threat_analysis (dict): Analyse des menaces
            
        Returns:
            Tuple[int, int]: Position sélectionnée (x, y)
        """
        if strategy_type == 'defensive':
            return self._select_defensive_position(threat_analysis)
        elif strategy_type == 'offensive':
            return self._select_offensive_position(threat_analysis)
        elif strategy_type == 'central':
            return self._select_central_position(threat_analysis)
        else:
            return random.choice(self.central_positions)

    def _select_defensive_position(self, threat_analysis: dict) -> Tuple[int, int]:
        """
        Sélectionne une position défensive optimale.
        
        Args:
            threat_analysis (dict): Analyse des menaces
            
        Returns:
            Tuple[int, int]: Position défensive (x, y)
        """
        # Prioriser les positions défensives basées sur la menace verticale
        if threat_analysis['top_threat'] > threat_analysis['bottom_threat']:
            # Plus de menaces en haut, privilégier les défenses hautes
            defensive_positions_sorted = sorted(
                self.defensive_positions, 
                key=lambda pos: pos[1]
            )
        else:
            # Plus de menaces en bas, privilégier les défenses basses
            defensive_positions_sorted = sorted(
                self.defensive_positions, 
                key=lambda pos: pos[1], 
                reverse=True
            )
        
        # Sélection avec un peu de randomness pour éviter la prédictibilité
        if len(defensive_positions_sorted) > 1 and random.random() > 0.7:
            return random.choice(defensive_positions_sorted[:2])
        
        return defensive_positions_sorted[0]

    def _select_offensive_position(self, threat_analysis: dict) -> Tuple[int, int]:
        """
        Sélectionne une position offensive optimale.
        
        Args:
            threat_analysis (dict): Analyse des menaces
            
        Returns:
            Tuple[int, int]: Position offensive (x, y)
        """
        # Stratégie offensive basée sur les zones faibles
        if threat_analysis['average_position']:
            avg_y = threat_analysis['average_position'][1]
            mid_y = self.frame_height / 2
            
            if avg_y < mid_y:
                # Ennemis plutôt en haut, attaquer en bas
                return min(self.offensive_positions, key=lambda pos: -pos[1])
            else:
                # Ennemis plutôt en bas, attaquer en haut
                return min(self.offensive_positions, key=lambda pos: pos[1])
        
        return random.choice(self.offensive_positions)

    def _select_central_position(self, threat_analysis: dict) -> Tuple[int, int]:
        """
        Sélectionne une position centrale optimale.
        
        Args:
            threat_analysis (dict): Analyse des menaces
            
        Returns:
            Tuple[int, int]: Position centrale (x, y)
        """
        # Position centrale adaptée à la distribution des ennemis
        if threat_analysis['average_position']:
            avg_x, avg_y = threat_analysis['average_position']
            
            # Sélectionner la position centrale la plus équilibrée
            best_position = min(
                self.central_positions,
                key=lambda pos: abs(pos[0] - avg_x/2) + abs(pos[1] - avg_y)
            )
            return best_position
        
        return random.choice(self.central_positions)

    def _get_default_placement(self) -> Tuple[int, int]:
        """
        Retourne une position par défaut quand aucun ennemi n'est détecté.
        
        Returns:
            Tuple[int, int]: Position par défaut (x, y)
        """
        # Alterner entre positions centrales et offensives
        if time.time() - self.last_placement_time > 5:  # 5 secondes
            return random.choice(self.offensive_positions)
        else:
            return random.choice(self.central_positions)

    def _update_strategy_history(self, strategy_type: str):
        """
        Met à jour l'historique stratégique pour l'analyse adaptative.
        
        Args:
            strategy_type (str): Type de stratégie utilisée
        """
        current_time = time.time()
        
        # Ajouter à l'historique
        self.strategy_history.append({
            'strategy': strategy_type,
            'timestamp': current_time
        })
        
        # Limiter la taille de l'historique (garder les 20 dernières)
        if len(self.strategy_history) > 20:
            self.strategy_history = self.strategy_history[-20:]
        
        # Compter les stratégies consécutives
        if strategy_type == self.current_strategy_type:
            self.consecutive_same_strategy += 1
        else:
            self.consecutive_same_strategy = 1
            self.current_strategy_type = strategy_type
        
        self.last_placement_time = current_time

    def get_predictive_placement(self, enemy_minions: List, prediction_time: float = None) -> Optional[Tuple[int, int]]:
        """
        Calcule un placement prédictif basé sur les trajectoires ennemies.
        
        Args:
            enemy_minions (List): Liste des minions ennemis
            prediction_time (float): Temps de prédiction en secondes
            
        Returns:
            Optional[Tuple[int, int]]: Position prédictive ou None
        """
        if prediction_time is None:
            prediction_time = PLACEMENT_PREDICTION_TIME
        
        if not enemy_minions:
            return None
        
        # Analyser les positions futures des ennemis
        future_positions = []
        current_time = time.time()
        target_time = current_time + prediction_time
        
        for minion in enemy_minions:
            pred_pos = minion.get_predicted_position_at_time(target_time)
            if pred_pos:
                future_positions.append(pred_pos)
        
        if not future_positions:
            return self.calculate_optimal_placement(enemy_minions)
        
        # Calculer une position d'interception basée sur les prédictions
        avg_future_x = sum(pos[0] for pos in future_positions) / len(future_positions)
        avg_future_y = sum(pos[1] for pos in future_positions) / len(future_positions)
        
        # Ajuster la position d'interception
        if PLAYER_SIDE == "left":
            intercept_x = max(0, avg_future_x - PLACEMENT_OFFSET_X)
        else:
            intercept_x = min(self.frame_width, avg_future_x + PLACEMENT_OFFSET_X)
        
        intercept_y = avg_future_y
        
        # Trouver la position stratégique la plus proche de l'interception
        all_positions = self.defensive_positions + self.offensive_positions + self.central_positions
        
        best_position = min(
            all_positions,
            key=lambda pos: distance_2d(pos, (intercept_x, intercept_y))
        )
        
        return best_position

    def get_strategy_stats(self) -> dict:
        """
        Retourne des statistiques sur l'utilisation des stratégies.
        
        Returns:
            dict: Statistiques stratégiques
        """
        if not self.strategy_history:
            return {'total_placements': 0}
        
        strategy_counts = {}
        recent_strategies = self.strategy_history[-10:]  # 10 dernières
        
        for entry in recent_strategies:
            strategy = entry['strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            'total_placements': len(self.strategy_history),
            'recent_strategy_distribution': strategy_counts,
            'current_strategy': self.current_strategy_type,
            'consecutive_same': self.consecutive_same_strategy
        }


def create_strategy_analyzer(frame_width: int, frame_height: int) -> StrategyAnalyzer:
    """
    Factory function pour créer un analyseur stratégique.
    
    Args:
        frame_width (int): Largeur de l'écran
        frame_height (int): Hauteur de l'écran
        
    Returns:
        StrategyAnalyzer: Instance configurée de l'analyseur
    """
    return StrategyAnalyzer(frame_width, frame_height)
