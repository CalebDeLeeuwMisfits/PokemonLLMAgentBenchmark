import logging
from typing import List, Dict, Any, Optional, Union
from smolagents import tool
from game_interface import PokemonRedMemoryMap, Emulator, Controller, ScreenCapture

logger = logging.getLogger(__name__)

class PokemonTools:
    """Collection of tools for the Pokémon LLM Agent to interact with PyBoy"""
    
    def __init__(self, emulator: Emulator, controller: Controller, screen_capture: ScreenCapture):
        self.emulator = emulator
        self.controller = controller
        self.screen_capture = screen_capture
    
    @tool
    def press_buttons(self, buttons: List[str]) -> str:
        """Send button presses to the Pokémon game.
        
        Args:
            buttons: List of buttons to press in sequence. Valid options: ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT", "wait"]
        """
        if not buttons:
            return "Error: No buttons specified"
            
        success = self.controller.press_sequence(buttons)
        return f"Button sequence {buttons} sent to game. Success: {success}"
    
    @tool
    def get_player_position(self) -> Dict[str, int]:
        """Get the current position of the player character on the map.
        """
        x, y = self.emulator.get_player_position()
        direction_code = self.emulator.read_memory(PokemonRedMemoryMap.PLAYER_DIRECTION)
        
        direction_map = {
            0: "down",
            4: "up",
            8: "left", 
            12: "right"  # 0C in hex
        }
        
        direction = direction_map.get(direction_code, "unknown")
        
        return {
            "x": x,
            "y": y,
            "direction": direction,
            "map_id": self.emulator.read_memory(PokemonRedMemoryMap.CURRENT_MAP)
        }
    
    @tool
    def get_pokemon_party(self) -> List[Dict[str, Any]]:
        """Get information about the current Pokémon party.
        """
        party = self.emulator.get_pokemon_party()
        return party
    
    @tool
    def has_badge(self, badge_index: int) -> bool:
        """Check if the player has a specific gym badge.
        
        Args:
            badge_index: Badge index (0-7). 0=Boulder, 1=Cascade, 2=Thunder, 3=Rainbow, 4=Soul, 5=Marsh, 6=Volcano, 7=Earth
        """
        return self.emulator.has_badge(badge_index)
    
    @tool
    def extract_text_from_screen(self, region: Optional[Dict[str, int]] = None) -> str:
        """Extract text from the current screen using OCR.
        
        Args:
            region: Optional dictionary with x, y, w, h keys defining a region to extract text from.
                   If not provided, extracts from the entire screen.
        """
        region_tuple = None
        if region:
            region_tuple = (region.get("x", 0), region.get("y", 0), 
                          region.get("w", 160), region.get("h", 144))
        
        text = self.screen_capture.extract_text(region=region_tuple)
        return text
    
    @tool
    def detect_dialog_box(self) -> bool:
        """Detect if a dialog box is currently present on screen.
        """
        return self.screen_capture.detect_dialog_box()
    
    @tool
    def advance_frames(self, count: int = 1, render: bool = True) -> str:
        """Advance the game by a specific number of frames.
        
        Args:
            count: Number of frames to advance
            render: Whether to render frames (set to False for faster emulation)
        """
        self.emulator.advance_frames(count, render)
        return f"Advanced game by {count} frames"
    
    @tool
    def set_emulation_speed(self, speed: Union[int, float]) -> str:
        """Set the emulation speed.
        
        Args:
            speed: Speed multiplier (0 for unlimited, 1 for normal speed, 2 for double speed, etc.)
        """
        if self.emulator.pyboy:
            self.emulator.pyboy.set_emulation_speed(speed)
            return f"Set emulation speed to {speed}x"
        return "Error: Emulator not initialized"