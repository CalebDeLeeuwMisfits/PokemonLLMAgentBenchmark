import os
import re
import json
import logging
import time
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Tuple

# Import smolagents
from smolagents import CodeAgent, HfApiModel, tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        # Create the CodeAgent
        tools = []
        
        # Add knowledge base tool
        self.update_knowledge_tool = update_knowledge
        tools.append(update_knowledge)
        
        # Add all Pokémon tools if provided
        if pokemon_tools:
            tools.extend([
                pokemon_tools.press_buttons,
                pokemon_tools.get_player_position,
                pokemon_tools.get_pokemon_party,
                pokemon_tools.has_badge,
                pokemon_tools.extract_text_from_screen,
                pokemon_tools.detect_dialog_box,
                pokemon_tools.advance_frames,
                pokemon_tools.set_emulation_speed
            ])
        
        # Initialize the CodeAgent
        self.agent = CodeAgent(
            model=self.model,
            tools=tools,
            max_steps=10, # Unlimit or give massive limit when running through entire game
            verbosity_level=1,
            planning_interval=3  # Plan every 3 steps
        )
        
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