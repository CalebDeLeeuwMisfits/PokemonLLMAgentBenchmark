from __future__ import annotations
import os
import re
import json
import logging
import time
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Tuple, Union

# Import smolagents
from smolagents import CodeAgent, tool
from game_interface import PokemonRedMemoryMap, Emulator, Controller, ScreenCapture

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store component references
_emulator = None
_controller = None
_screen_capture = None

def setup_tool_dependencies(emulator: Emulator, controller: Controller, screen_capture: ScreenCapture):
    """Setup global dependencies for tool functions"""
    global _emulator, _controller, _screen_capture
    _emulator = emulator
    _controller = controller
    _screen_capture = screen_capture

# Define standalone tool functions

@tool
def press_buttons(buttons: List[str]) -> str:
    """Send button presses to the Pokémon game.
    
    Args:
        buttons: List of buttons to press in sequence. Valid options: ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT", "wait"]
    """
    if not buttons:
        return "Error: No buttons specified"
        
    success = _controller.press_sequence(buttons)
    return f"Button sequence {buttons} sent to game. Success: {success}"

@tool
def get_player_position() -> Dict[str, Any]:
    """Get the current position of the player character on the map."""
    x, y = _emulator.get_player_position()
    direction_code = _emulator.read_memory(PokemonRedMemoryMap.PLAYER_DIRECTION)
    
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
        "map_id": _emulator.read_memory(PokemonRedMemoryMap.CURRENT_MAP)
    }

@tool
def get_pokemon_party() -> List[Dict[str, Any]]:
    """Get information about the current Pokémon party."""
    party = _emulator.get_pokemon_party()
    return party

@tool
def has_badge(badge_index: int) -> bool:
    """Check if the player has a specific gym badge.
    
    Args:
        badge_index: Badge index (0-7). 0=Boulder, 1=Cascade, 2=Thunder, 3=Rainbow, 4=Soul, 5=Marsh, 6=Volcano, 7=Earth
    """
    return _emulator.has_badge(badge_index)

@tool
def extract_text_from_screen(region: Optional[Dict[str, int]] = None) -> str:
    """Extract text from the current screen using OCR.
    
    Args:
        region: Optional dictionary with x, y, w, h keys defining a region to extract text from.
               If not provided, extracts from the entire screen.
    """
    region_tuple = None
    if region:
        region_tuple = (region.get("x", 0), region.get("y", 0), 
                      region.get("w", 160), region.get("h", 144))
    
    text = _screen_capture.extract_text(region=region_tuple)
    return text

@tool
def detect_dialog_box() -> bool:
    """Detect if a dialog box is currently present on screen."""
    return _screen_capture.detect_dialog_box()

@tool
def advance_frames(count: int = 1, render: bool = True) -> str:
    """Advance the game by a specific number of frames.
    
    Args:
        count: Number of frames to advance
        render: Whether to render frames (set to False for faster emulation)
    """
    _emulator.advance_frames(count, render)
    return f"Advanced game by {count} frames"

@tool
def set_emulation_speed(speed: Union[int, float]) -> str:
    """Set the emulation speed.
    
    Args:
        speed: Speed multiplier (0 for unlimited, 1 for normal speed, 2 for double speed, etc.)
    """
    if _emulator.pyboy:
        _emulator.pyboy.set_emulation_speed(speed)
        return f"Set emulation speed to {speed}x"
    return "Error: Emulator not initialized"

class KnowledgeBase:
    def __init__(self):
        self.sections = {}
        # Initialize with basic Pokémon knowledge
        self.sections["game_controls"] = """
        Game controls:
        - A: Confirm/Interact
        - B: Cancel/Back
        - UP/DOWN/LEFT/RIGHT: Movement
        - START: Open menu
        - SELECT: Use registered item
        """
        self.sections["locations"] = "Currently in Pallet Town."
        self.sections["pokemon_team"] = "No Pokémon in team yet."
        self.sections["current_objective"] = "Start the game and choose a starter Pokémon."
        self.sections["map_knowledge"] = "Pallet Town is a small town with a few houses and Professor Oak's lab."
        
    def to_xml(self):
        """Convert knowledge base to XML format"""
        root = ET.Element("knowledge_base")
        for key, value in self.sections.items():
            section = ET.SubElement(root, "section", id=key)
            section.text = value
        
        return ET.tostring(root, encoding="unicode")
    
    def update_section(self, section_id, content):
        """Update a section of the knowledge base"""
        self.sections[section_id] = content
        
    def get_section(self, section_id):
        """Get a section from the knowledge base"""
        return self.sections.get(section_id, "")
    
    def get_all_sections(self):
        """Return all knowledge base sections as a formatted string"""
        result = "KNOWLEDGE BASE:\n\n"
        for key, value in self.sections.items():
            result += f"[{key.upper()}]\n{value}\n\n"
        return result

class PokemonAgent:
    def __init__(self, api_key=None, pokemon_tools=None, knowledge_base=None, 
                 llm_provider="anthropic", model_name=None):
        """
        Initialize the Pokémon agent with configurable LLM provider
        
        Args:
            api_key: API key for cloud providers (Anthropic)
            pokemon_tools: Pokemon tools interface
            knowledge_base: Knowledge base instance
            llm_provider: "anthropic" or "ollama" 
            model_name: Model name (default: "claude-3-sonnet-20240229" for Anthropic or "deepseek-r1" for Ollama)
        """
        self.llm_provider = llm_provider.lower()
        
        # Set default model name based on provider if not specified
        if model_name is None:
            if self.llm_provider == "anthropic":
                self.model_name = "claude-3-sonnet-20240229"
            elif self.llm_provider == "ollama":
                self.model_name = "deepseek-coder:16b-instruct"
            else:
                raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
        else:
            self.model_name = model_name
        
        # Initialize model client
        if self.llm_provider == "anthropic":
            try:
                from smolagents import AnthropicModel
                if not api_key:
                    api_key = os.environ.get("ANTHROPIC_API_KEY")
                    if not api_key:
                        raise ValueError("API key is required for Anthropic. Set it directly or via ANTHROPIC_API_KEY environment variable.")
                self.model = AnthropicModel(
                    model=self.model_name,
                    api_key=api_key,
                    max_tokens=2000,
                    temperature=0.7
                )
                logger.info(f"Initialized Anthropic client with model {self.model_name}")
            except ImportError:
                raise ImportError("The 'smolagents' package is required. Install it with 'pip install smolagents'.")
                
        elif self.llm_provider == "ollama":
            try:
                from smolagents import OllamaModel
                self.model = OllamaModel(
                    model=self.model_name,
                    max_tokens=2000,
                    temperature=0.7
                )
                logger.info(f"Initialized Ollama client with model {self.model_name}")
            except ImportError:
                raise ImportError("The 'smolagents' package is required. Install it with 'pip install smolagents'.")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
            
        self.pokemon_tools = pokemon_tools
        self.knowledge_base = knowledge_base
        self.last_screenshot = None
        self.step_count = 0
        self.callback = None
        
        # Create tools for knowledge base
        @tool
        def update_knowledge(section: str, content: str) -> str:
            """Update the agent's knowledge base with new information
            
            Args:
                section: The section of knowledge to update (e.g., "locations", "pokemon_team")
                content: The new content to store in this section
            """
            self.knowledge_base.update_section(section, content)
            return f"Updated knowledge base section '{section}'"
        
        # Create the list of tools
        tools = []
        
        # Add knowledge base tool
        self.update_knowledge_tool = update_knowledge
        tools.append(update_knowledge)
        
        # Add all Pokémon tools
        tools.extend([
            press_buttons,
            get_player_position,
            get_pokemon_party,
            has_badge,
            extract_text_from_screen,
            detect_dialog_box,
            advance_frames,
            set_emulation_speed
        ])
        
        # Initialize the CodeAgent
        self.agent = CodeAgent(
            model=self.model,
            tools=tools,
            max_steps=10, # Unlimit or give massive limit when running through entire game
            verbosity_level=1,
            planning_interval=3  # Plan every 3 steps
        )
    
    def register_callback(self, callback):
        """Register a callback function for agent thoughts"""
        self.callback = callback
        
    def process_game_state(self, screenshot):
        """Process the current game state and take action"""
        # Default task for the agent
        task = "Analyze the current game state and take the next logical action to make progress."
        
        # Run the agent
        result = self.run(task, screenshot)
        
        # Extract information from the result for the callback
        if self.callback:
            game_state = "Current game state based on screenshot"
            action = "Action taken by agent"
            response = result.get('reasoning', '')
            result_text = result.get('result', '')
            
            # Call the callback with the information
            self.callback(self.step_count, game_state, response, action, result_text)
        
        return result
        
    def run(self, task, screenshot=None):
        """Run the agent with a specific task and screenshot"""
        self.step_count += 1
        self.last_screenshot = screenshot
        
        # Prepare additional context from knowledge base
        kb_content = self.knowledge_base.get_all_sections()
        
        # Run the agent
        task_with_context = f"""
        {task}
        
        Current knowledge base:
        {kb_content}
        
        Use the tools available to you to interact with the Pokémon game and make progress.
        """
        
        # Execute the agent
        result = self.agent.run(task_with_context, additional_args={"screenshot": screenshot})
        
        return result