"""
Interface graphique overlay pour Minion Masters.
Ce module gère la superposition transparente Tkinter qui affiche les suggestions de placement.
"""

import tkinter as tk
from typing import Optional, Tuple, List, Callable
import threading
import queue
import time

import config


class GameOverlay:
    """
    Superposition transparente pour afficher les suggestions de placement en jeu.
    Utilise Tkinter avec transparence pour se superposer au jeu sans l'interférer.
    """
    
    def __init__(self):
        """Initialise l'overlay avec tous les paramètres nécessaires."""
        self.frame_width = config.MONITOR["width"]
        self.frame_height = config.MONITOR["height"]
        
        # Interface Tkinter
        self.root = None
        self.canvas = None
        self.suggestion_circle = None
        self.debug_elements = []
        
        # État de l'overlay
        self.is_running = False
        self.show_debug = False
        
        # Communication thread-safe
        self.update_queue = queue.Queue()
        
        # Callbacks
        self.on_close_callback: Optional[Callable] = None
        
        print("🎨 Overlay initialisé")
    
    def setup_window(self):
        """Configure la fenêtre principale de l'overlay."""
        self.root = tk.Tk()
        self.root.title("Minion Masters Assistant")
        
        # Configuration de la fenêtre
        self.root.geometry(f"{self.frame_width}x{self.frame_height}+0+0")
        self.root.attributes('-topmost', True)  # Toujours au premier plan
        self.root.attributes('-transparentcolor', config.TRANSPARENT_COLOR)  # Transparence
        self.root.overrideredirect(True)  # Supprime les bordures de fenêtre
        
        # Gestion de la fermeture
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)
        
        # Création du canvas
        self.canvas = tk.Canvas(
            self.root,
            width=self.frame_width,
            height=self.frame_height,
            bg=config.TRANSPARENT_COLOR,
            highlightthickness=0
        )
        self.canvas.pack()
        
        # Création des éléments graphiques
        self._create_ui_elements()
        
        print(f"✅ Fenêtre overlay configurée ({self.frame_width}x{self.frame_height})")
    
    def _create_ui_elements(self):
        """Crée tous les éléments graphiques de l'overlay."""
        # Cercle de suggestion principal
        self.suggestion_circle = self.canvas.create_oval(
            0, 0, 0, 0,
            outline=config.SUGGESTION_CIRCLE_COLOR,
            width=config.SUGGESTION_CIRCLE_WIDTH,
            state='hidden'
        )
        
        # Éléments de debug (masqués par défaut)
        self._create_debug_elements()
    
    def _create_debug_elements(self):
        """Crée les éléments de debug (zones d'exclusion, positions stratégiques, etc.)."""
        # Zones d'exclusion
        exclusion_radius = self.frame_width * config.EXCLUSION_RADIUS_RATIO
        center_y = self.frame_height * config.EXCLUSION_CENTER_Y_RATIO
        
        # Zone d'exclusion gauche
        left_exclusion = self.canvas.create_oval(
            0, center_y - exclusion_radius,
            exclusion_radius * 2, center_y + exclusion_radius,
            outline='red', width=2, state='hidden'
        )
        
        # Zone d'exclusion droite
        right_exclusion = self.canvas.create_oval(
            self.frame_width - exclusion_radius * 2, center_y - exclusion_radius,
            self.frame_width, center_y + exclusion_radius,
            outline='red', width=2, state='hidden'
        )
        
        self.debug_elements.extend([left_exclusion, right_exclusion])
        
        # Positions stratégiques
        self._create_strategic_positions_debug()
    
    def _create_strategic_positions_debug(self):
        """Crée les marqueurs de debug pour les positions stratégiques."""
        from utils import ratio_to_pixels
        
        # Positions défensives (bleu)
        defensive_positions = ratio_to_pixels(
            config.DEFENSIVE_POSITIONS_RATIOS, 
            self.frame_width, 
            self.frame_height
        )
        for x, y in defensive_positions:
            marker = self.canvas.create_oval(
                x - 5, y - 5, x + 5, y + 5,
                outline='blue', width=2, fill='blue',
                state='hidden'
            )
            self.debug_elements.append(marker)
        
        # Positions offensives (rouge)
        offensive_positions = ratio_to_pixels(
            config.OFFENSIVE_POSITIONS_RATIOS, 
            self.frame_width, 
            self.frame_height
        )
        for x, y in offensive_positions:
            marker = self.canvas.create_oval(
                x - 5, y - 5, x + 5, y + 5,
                outline='red', width=2, fill='red',
                state='hidden'
            )
            self.debug_elements.append(marker)
        
        # Positions centrales (vert)
        central_positions = ratio_to_pixels(
            config.CENTRAL_POSITIONS_RATIOS, 
            self.frame_width, 
            self.frame_height
        )
        for x, y in central_positions:
            marker = self.canvas.create_oval(
                x - 5, y - 5, x + 5, y + 5,
                outline='green', width=2, fill='green',
                state='hidden'
            )
            self.debug_elements.append(marker)
    
    def update_suggestion(self, position: Optional[Tuple[float, float]]):
        """
        Met à jour la position de suggestion de placement.
        
        Args:
            position: Coordonnées (x, y) de la suggestion, ou None pour masquer
        """
        try:
            self.update_queue.put(('suggestion', position), block=False)
        except queue.Full:
            pass  # Ignore si la queue est pleine
    
    def toggle_debug_mode(self):
        """Active/désactive l'affichage des éléments de debug."""
        self.show_debug = not self.show_debug
        try:
            self.update_queue.put(('debug_toggle', self.show_debug), block=False)
        except queue.Full:
            pass
        print(f"🔧 Mode debug: {'ACTIVÉ' if self.show_debug else 'DÉSACTIVÉ'}")
    
    def add_minion_marker(self, minion_id: int, position: Tuple[float, float], 
                         is_enemy: bool = False, is_predicted: bool = False):
        """
        Ajoute un marqueur pour visualiser un minion détecté.
        
        Args:
            minion_id: ID unique du minion
            position: Position (x, y) du minion
            is_enemy: True si c'est un ennemi
            is_predicted: True si c'est une position prédite
        """
        if not self.show_debug:
            return
        
        try:
            self.update_queue.put(('minion_marker', {
                'id': minion_id,
                'position': position,
                'is_enemy': is_enemy,
                'is_predicted': is_predicted
            }), block=False)
        except queue.Full:
            pass
    
    def clear_minion_markers(self):
        """Efface tous les marqueurs de minions."""
        try:
            self.update_queue.put(('clear_markers', None), block=False)
        except queue.Full:
            pass
    
    def _process_updates(self):
        """Traite toutes les mises à jour en attente dans la queue."""
        try:
            while True:
                try:
                    update_type, data = self.update_queue.get_nowait()
                    
                    if update_type == 'suggestion':
                        self._update_suggestion_display(data)
                    elif update_type == 'debug_toggle':
                        self._toggle_debug_display(data)
                    elif update_type == 'minion_marker':
                        self._add_minion_marker_display(data)
                    elif update_type == 'clear_markers':
                        self._clear_minion_markers_display()
                    
                except queue.Empty:
                    break
                    
        except Exception as e:
            print(f"⚠️  Erreur lors du traitement des mises à jour: {e}")
    
    def _update_suggestion_display(self, position: Optional[Tuple[float, float]]):
        """Met à jour l'affichage du cercle de suggestion."""
        if position is None:
            self.canvas.itemconfig(self.suggestion_circle, state='hidden')
        else:
            x, y = position
            radius = config.SUGGESTION_CIRCLE_RADIUS
            
            # Mise à jour des coordonnées
            self.canvas.coords(
                self.suggestion_circle,
                x - radius, y - radius,
                x + radius, y + radius
            )
            self.canvas.itemconfig(self.suggestion_circle, state='normal')
    
    def _toggle_debug_display(self, show_debug: bool):
        """Active/désactive l'affichage des éléments de debug."""
        state = 'normal' if show_debug else 'hidden'
        for element in self.debug_elements:
            self.canvas.itemconfig(element, state=state)
    
    def _add_minion_marker_display(self, marker_data: dict):
        """Ajoute un marqueur de minion à l'affichage."""
        if not hasattr(self, '_minion_markers'):
            self._minion_markers = {}
        
        minion_id = marker_data['id']
        x, y = marker_data['position']
        is_enemy = marker_data['is_enemy']
        is_predicted = marker_data['is_predicted']
        
        # Couleur selon le type
        if is_predicted:
            color = 'yellow'
            size = 3
        elif is_enemy:
            color = 'red'
            size = 4
        else:
            color = 'cyan'
            size = 4
        
        # Supprime l'ancien marqueur s'il existe
        if minion_id in self._minion_markers:
            self.canvas.delete(self._minion_markers[minion_id])
        
        # Crée le nouveau marqueur
        marker = self.canvas.create_oval(
            x - size, y - size, x + size, y + size,
            outline=color, width=2, fill=color,
            state='normal' if self.show_debug else 'hidden'
        )
        
        self._minion_markers[minion_id] = marker
    
    def _clear_minion_markers_display(self):
        """Efface tous les marqueurs de minions de l'affichage."""
        if hasattr(self, '_minion_markers'):
            for marker in self._minion_markers.values():
                self.canvas.delete(marker)
            self._minion_markers.clear()
    
    def _on_window_close(self):
        """Gestionnaire de fermeture de fenêtre."""
        print("🔴 Fermeture de l'overlay demandée")
        self.stop()
        if self.on_close_callback:
            self.on_close_callback()
    
    def set_close_callback(self, callback: Callable):
        """
        Définit le callback à appeler lors de la fermeture.
        
        Args:
            callback: Fonction à appeler lors de la fermeture
        """
        self.on_close_callback = callback
    
    def run(self):
        """
        Lance l'overlay en mode bloquant.
        Cette méthode doit être appelée dans le thread principal.
        """
        if self.is_running:
            print("⚠️  L'overlay est déjà en cours d'exécution")
            return
        
        self.is_running = True
        self.setup_window()
        
        print("🚀 Overlay démarré")
        print("   - Superposition transparente active")
        print("   - Appuyez sur Alt+Tab pour voir les autres fenêtres")
        print("   - Fermez cette fenêtre pour arrêter l'assistant")
        
        # Boucle principale de l'interface
        self._main_loop()
    
    def _main_loop(self):
        """Boucle principale de l'interface utilisateur."""
        try:
            while self.is_running and self.root:
                # Traitement des mises à jour
                self._process_updates()
                
                # Mise à jour de l'interface
                self.root.update_idletasks()
                self.root.update()
                
                # Pause pour éviter de surcharger le CPU
                time.sleep(config.FRAME_DELAY)
                
        except tk.TclError:
            # La fenêtre a été fermée
            self.is_running = False
        except Exception as e:
            print(f"⚠️  Erreur dans la boucle principale de l'overlay: {e}")
            self.is_running = False
    
    def stop(self):
        """Arrête l'overlay et nettoie les ressources."""
        print("⏹️  Arrêt de l'overlay...")
        self.is_running = False
        
        if self.root:
            try:
                self.root.quit()
                self.root.destroy()
            except:
                pass
            self.root = None
        
        print("✅ Overlay arrêté")
    
    def is_active(self) -> bool:
        """
        Vérifie si l'overlay est actif.
        
        Returns:
            bool: True si l'overlay est en cours d'exécution
        """
        return self.is_running and self.root is not None


class OverlayController:
    """
    Contrôleur pour gérer l'overlay dans un thread séparé.
    Permet l'utilisation non-bloquante de l'overlay.
    """
    
    def __init__(self):
        """Initialise le contrôleur d'overlay."""
        self.overlay = GameOverlay()
        self.overlay_thread = None
        self.is_running = False
    
    def start(self, close_callback: Optional[Callable] = None):
        """
        Démarre l'overlay dans un thread séparé.
        
        Args:
            close_callback: Callback appelé lors de la fermeture
        """
        if self.is_running:
            print("⚠️  L'overlay est déjà en cours d'exécution")
            return
        
        if close_callback:
            self.overlay.set_close_callback(close_callback)
        
        # Démarrage dans un thread séparé
        self.overlay_thread = threading.Thread(
            target=self.overlay.run,
            daemon=True,
            name="OverlayThread"
        )
        
        self.is_running = True
        self.overlay_thread.start()
        
        # Attendre que l'overlay soit initialisé
        timeout = 5.0
        start_time = time.time()
        while not self.overlay.is_active() and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if self.overlay.is_active():
            print("✅ Overlay démarré avec succès dans un thread séparé")
        else:
            print("❌ Échec du démarrage de l'overlay")
            self.is_running = False
    
    def stop(self):
        """Arrête l'overlay et attend la fin du thread."""
        if not self.is_running:
            return
        
        self.overlay.stop()
        self.is_running = False
        
        if self.overlay_thread and self.overlay_thread.is_alive():
            self.overlay_thread.join(timeout=2.0)
        
        print("✅ Contrôleur d'overlay arrêté")
    
    def update_suggestion(self, position: Optional[Tuple[float, float]]):
        """Met à jour la suggestion de placement."""
        if self.is_running and self.overlay.is_active():
            self.overlay.update_suggestion(position)
    
    def toggle_debug(self):
        """Active/désactive le mode debug."""
        if self.is_running and self.overlay.is_active():
            self.overlay.toggle_debug_mode()
    
    def add_minion_marker(self, minion_id: int, position: Tuple[float, float],
                         is_enemy: bool = False, is_predicted: bool = False):
        """Ajoute un marqueur de minion."""
        if self.is_running and self.overlay.is_active():
            self.overlay.add_minion_marker(minion_id, position, is_enemy, is_predicted)
    
    def clear_minion_markers(self):
        """Efface tous les marqueurs de minions."""
        if self.is_running and self.overlay.is_active():
            self.overlay.clear_minion_markers()
    
    def is_active(self) -> bool:
        """Vérifie si l'overlay est actif."""
        return self.is_running and self.overlay.is_active()


# ==============================================================================
# --- FONCTIONS UTILITAIRES POUR L'OVERLAY ---
# ==============================================================================

def create_overlay_controller() -> OverlayController:
    """
    Factory function pour créer un contrôleur d'overlay.
    
    Returns:
        OverlayController: Instance configurée du contrôleur
    """
    return OverlayController()


def test_overlay():
    """Fonction de test pour l'overlay."""
    print("🧪 Test de l'overlay...")
    
    overlay = GameOverlay()
    
    def test_positions():
        """Test des différentes positions."""
        from utils import ratio_to_pixels
        
        positions = ratio_to_pixels(
            config.POSITIONS_PRINCIPALES_RATIOS,
            config.MONITOR["width"],
            config.MONITOR["height"]
        )
        
        for i, pos in enumerate(positions):
            print(f"   Position {i+1}: {pos}")
            overlay.update_suggestion(pos)
            time.sleep(2)
        
        overlay.update_suggestion(None)
        print("   Test terminé")
    
    # Démarrer le test dans un thread
    test_thread = threading.Thread(target=test_positions, daemon=True)
    test_thread.start()
    
    # Lancer l'overlay
    overlay.run()


if __name__ == "__main__":
    # Test de l'overlay si exécuté directement
    test_overlay()
