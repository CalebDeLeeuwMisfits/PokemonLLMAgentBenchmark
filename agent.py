import os
import re
import json
import logging
import time
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Tuple

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
    def __init__(self, api_key=None, controller=None, screen_capture=None, knowledge_base=None, 
                 llm_provider="anthropic", model_name=None):
        """
        Initialize the Pokémon agent with configurable LLM provider
        
        Args:
            api_key: API key for cloud providers (Anthropic)
            controller: Game controller interface
            screen_capture: Screen capture interface
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
        
        # Initialize appropriate client
        if self.llm_provider == "anthropic":
            try:
                import anthropic
                if not api_key:
                    api_key = os.environ.get("ANTHROPIC_API_KEY")
                    if not api_key:
                        raise ValueError("API key is required for Anthropic. Set it directly or via ANTHROPIC_API_KEY environment variable.")
                self.client = anthropic.Anthropic(api_key=api_key)
                logger.info(f"Initialized Anthropic client with model {self.model_name}")
            except ImportError:
                raise ImportError("The 'anthropic' package is required for using Claude. Install it with 'pip install anthropic'.")
                
        elif self.llm_provider == "ollama":
            try:
                import ollama
                self.client = ollama  # The ollama module itself is the client
                logger.info(f"Initialized Ollama client with model {self.model_name}")
                # Check if the model is available
                try:
                    self.client.list()
                    logger.info("Ollama is running and available")
                except Exception as e:
                    logger.warning(f"Ollama may not be properly installed or running: {e}")
                    logger.warning("Make sure Ollama is installed and the Ollama service is running")
            except ImportError:
                raise ImportError("The 'ollama' package is required for local inference. Install it with 'pip install ollama'.")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
            
        self.controller = controller
        self.screen_capture = screen_capture
        self.knowledge_base = knowledge_base
        self.conversation_history = []
        self.last_screenshot = None
        self.current_location = None
        self.step_count = 0
        self.callbacks = []
        
    def register_callback(self, callback_function):
        """Register a callback function for agent decisions"""
        self.callbacks.append(callback_function)
        
    def _create_prompt(self, game_state_description, screenshot_path=None):
        """Create a prompt for the LLM with current game state and knowledge base"""
        
        # Include recent conversation history (last 3 exchanges)
        history_text = ""
        if self.conversation_history:
            history = self.conversation_history[-3:]
            for i, entry in enumerate(history):
                history_text += f"PREVIOUS TURN {self.step_count - len(history) + i}:\n"
                history_text += f"Game state: {entry.get('game_state', '')[:100]}...\n"
                history_text += f"Action: {entry.get('action', '')}\n"
                history_text += f"Result: {entry.get('result', '')}\n\n"
        
        # Combine all components
        system_prompt = f"""
        You are an AI agent playing Pokémon Red. You can see the game through screenshots and control it using button commands.
        
        Your goal is to progress through the game by making strategic decisions, battling trainers, catching and training Pokémon, and ultimately becoming the champion.
        
        CURRENT GAME STATE:
        {game_state_description}
        
        {self.knowledge_base.get_all_sections()}
        
        {history_text}
        
        You can perform actions by using these tools:
        
        1. use_emulator - Send button presses to the game. Valid buttons: ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT"]
          Example: {{ "tool": "use_emulator", "parameters": {{ "buttons": ["UP", "A", "wait"] }} }}
        
        2. update_knowledge - Update your knowledge base with new information.
          Example: {{ "tool": "update_knowledge", "parameters": {{ "section": "locations", "content": "Currently in Viridian City Pokémon Center." }} }}
        
        3. navigate - Move to a specific location using pathfinding.
          Example: {{ "tool": "navigate", "parameters": {{ "destination": "Pokémon Center", "current_location": "Viridian City" }} }}
          
        4. analyze_screen - Request detailed analysis of the current screen to identify specific elements.
          Example: {{ "tool": "analyze_screen", "parameters": {{ "focus_area": "dialog_text" }} }}
        
        Respond with your reasoning about the current game state, then call ONE tool per turn to interact with the game.
        Format your tool calls as JSON objects exactly as shown in the examples.
        """
        
        return system_prompt
        
    def process_game_state(self, screenshot):
        """Process the current game state and decide on the next action"""
        self.step_count += 1
        self.last_screenshot = screenshot
        
        # 1. Analyze the screenshot
        game_state_description = self._analyze_screenshot(screenshot)
        
        # 2. Create the prompt
        prompt = self._create_prompt(game_state_description)
        
        # 3. Call LLM API
        response = self._call_llm(prompt)
        
        # 4. Parse response and execute actions
        action_result = self._execute_action(response)
        
        # 5. Update conversation history
        self._update_conversation_history(
            game_state_description, 
            response, 
            self._extract_tool_call(response),
            action_result
        )
        
        # 6. Log the agent's thinking and actions
        self._log_agent_action(game_state_description, response, action_result)
        
        return action_result
        
    def _analyze_screenshot(self, screenshot):
        """Analyze the screenshot to understand the game state"""
        # In a real implementation, this would use OCR and image recognition
        # For now, we'll return a placeholder description
        
        # Convert numpy array to PIL Image for processing
        if isinstance(screenshot, np.ndarray):
            image = Image.fromarray(screenshot)
            # Here you would do actual image analysis
            
            # For placeholder, get basic image info
            width, height = image.size
            return f"Game screen shows a Pokémon game view. Resolution: {width}x{height}. The game appears to be running. Please analyze what's visible and make a strategic decision."
        else:
            return "No screenshot available. Please take actions to explore the game."
    
    def _call_llm(self, prompt):
        """Call the configured LLM with the given prompt"""
        try:
            start_time = time.time()
            
            if self.llm_provider == "anthropic":
                # Call Claude API
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=2000,
                    temperature=0.7,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                result = response.content[0].text
                
            elif self.llm_provider == "ollama":
                # Call Ollama API
                response = self.client.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an AI agent playing Pokémon Red."},
                        {"role": "user", "content": prompt}
                    ]
                )
                result = response["message"]["content"]
            
            elapsed = time.time() - start_time
            logger.info(f"LLM response received in {elapsed:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error calling {self.llm_provider} LLM: {e}")
            return f"Error: Unable to get response from {self.llm_provider} LLM. Exception: {str(e)}"
    
    def _extract_tool_call(self, response_text):
        """Extract tool call JSON from the response text"""
        try:
            # Find JSON-like tool call in the response
            json_match = re.search(r'({[\s\S]*"tool"[\s\S]*})', response_text)
            if json_match:
                json_str = json_match.group(1)
                # Parse the JSON
                tool_call = json.loads(json_str)
                return tool_call
            else:
                logger.warning("No valid tool call found in response")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting tool call: {e}")
            return None
            
    def _execute_action(self, response):
        """Execute the action specified in the response"""
        tool_call = self._extract_tool_call(response)
        
        if not tool_call:
            return "No valid action found in response."
            
        try:
            tool = tool_call.get("tool")
            parameters = tool_call.get("parameters", {})
            
            if tool == "use_emulator":
                buttons = parameters.get("buttons", [])
                if not buttons:
                    return "No buttons specified for emulator."
                    
                # Send button presses to the controller
                success = self.controller.press_sequence(buttons)
                return f"Button sequence {buttons} sent to emulator. Success: {success}"
                
            elif tool == "update_knowledge":
                section = parameters.get("section")
                content = parameters.get("content")
                
                if not section or not content:
                    return "Section and content required for knowledge update."
                    
                self.knowledge_base.update_section(section, content)
                return f"Knowledge base updated. Section '{section}' now contains: {content[:50]}..."
                
            elif tool == "navigate":
                destination = parameters.get("destination")
                current_location = parameters.get("current_location") or self.current_location
                
                if not destination:
                    return "Destination required for navigation."
                    
                # In a real implementation, this would use pathfinding logic
                return f"Navigation from {current_location} to {destination} would be implemented here."
                
            elif tool == "analyze_screen":
                focus_area = parameters.get("focus_area", "full_screen")
                
                # In a real implementation, this would perform detailed image analysis
                return f"Detailed analysis of {focus_area} would be performed here."
                
            else:
                return f"Unknown tool: {tool}"
                
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            return f"Error executing action: {str(e)}"
            
    def _update_conversation_history(self, game_state, response, tool_call, result):
        """Update the conversation history with the current interaction"""
        # Limit history to last 5 exchanges
        if len(self.conversation_history) >= 5:
            self.conversation_history.pop(0)
            
        # Add current exchange
        self.conversation_history.append({
            'game_state': game_state,
            'response': response,
            'action': tool_call,
            'result': result
        })
            
    def _log_agent_action(self, game_state, response, result):
        """Log the agent's thinking and actions for visualization"""
        # Extract the reasoning part from the response (everything before the JSON tool call)
        reasoning = response.split('{')[0].strip() if '{' in response else response
        
        print("\n" + "="*50)
        print(f"STEP {self.step_count} | GAME STATE:")
        print("-"*50)
        print(game_state[:100] + "..." if len(game_state) > 100 else game_state)
        print("\nAGENT REASONING:")
        print("-"*50)
        print(reasoning[:150] + "..." if len(reasoning) > 150 else reasoning)
        print("\nACTION TAKEN:")
        print("-"*50)
        tool_call = self._extract_tool_call(response)
        print(json.dumps(tool_call, indent=2) if tool_call else "No valid tool call found")
        print("\nRESULT:")
        print("-"*50)
        print(result)
        print("="*50 + "\n")
        
        # Call all registered callbacks
        for callback in self.callbacks:
            try:
                callback(self.step_count, game_state, reasoning, tool_call, result)
            except Exception as e:
                logger.error(f"Error in callback: {e}")