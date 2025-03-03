import time
import json
import anthropic
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

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
        root = ET.Element("knowledge_base")
        for key, value in self.sections.items():
            section = ET.SubElement(root, "section", id=key)
            section.text = value
        
        return ET.tostring(root, encoding="unicode")
    
    def update_section(self, section_id, content):
        self.sections[section_id] = content
        
    def get_section(self, section_id):
        return self.sections.get(section_id, "")
    
    def get_all_sections(self):
        """Return all knowledge base sections as a formatted string"""
        result = "KNOWLEDGE BASE:\n\n"
        for key, value in self.sections.items():
            result += f"--- {key} ---\n{value}\n\n"
        return result

class PokemonAgent:
    def __init__(self, api_key, controller, screen_capture, knowledge_base):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.controller = controller
        self.screen_capture = screen_capture
        self.knowledge_base = knowledge_base
        self.conversation_history = []
        self.last_screenshot = None
        self.current_location = None
        self.step_count = 0
        self.callbacks = []  # Add this line
        
    def register_callback(self, callback_function):
        """Register a callback function for agent decisions"""
        self.callbacks.append(callback_function)
        
    def _create_prompt(self, game_state_description, screenshot_path=None):
        """Create a prompt for Claude with current game state and knowledge base"""
        
        # Include recent conversation history (last 3 exchanges)
        history_text = ""
        if self.conversation_history:
            history = self.conversation_history[-3:]
            for i, entry in enumerate(history):
                history_text += f"\nPrevious Turn {i+1}:\n"
                history_text += f"Game state: {entry['game_state']}\n"
                history_text += f"Your reasoning: {entry['response']}\n"
                history_text += f"Actions taken: {entry['actions_taken']}\n"
                history_text += f"Result: {entry['result']}\n"
        
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
        
        # 3. Call Claude API
        response = self._call_claude_api(prompt)
        
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
            return f"The game screen shows a {width}x{height} pixel view. The player character appears to be in what looks like Pallet Town. There are buildings visible and some NPCs. There's text at the bottom of the screen that appears to be dialog."
        else:
            return "The game screen shows the player character standing in what appears to be Pallet Town. There are two houses visible, and Professor Oak's lab is at the bottom of the screen. No wild Pokémon or trainers are visible."
    
    def _call_claude_api(self, prompt):
        """Call Claude using the Anthropic API"""
        try:
            message = self.client.messages.create(
                model="claude-3-7-sonnet-20240229",
                max_tokens=1024,
                temperature=0.7,
                system=prompt,
                messages=[{"role": "user", "content": "What should I do next in the game?"}]
            )
            return message.content[0].text
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            return "Failed to get response from Claude."
    
    def _extract_tool_call(self, response_text):
        """Extract tool call from Claude's response"""
        try:
            # Look for JSON-formatted tool call
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}")
            
            if start_idx >= 0 and end_idx >= 0:
                json_str = response_text[start_idx:end_idx+1]
                
                # Try to parse the extracted JSON
                try:
                    tool_call = json.loads(json_str)
                    if "tool" in tool_call and "parameters" in tool_call:
                        return tool_call
                except json.JSONDecodeError:
                    # If that fails, try to find more specific patterns
                    pass
            
            # Try to match specific tool patterns if JSON parsing failed
            patterns = [
                {"tool": "use_emulator", "pattern": r'{\s*"tool"\s*:\s*"use_emulator"\s*,\s*"parameters"\s*:\s*{\s*"buttons"\s*:\s*\[(.*?)\]\s*}\s*}'},
                {"tool": "update_knowledge", "pattern": r'{\s*"tool"\s*:\s*"update_knowledge"\s*,\s*"parameters"\s*:\s*{\s*"section"\s*:\s*"(.*?)"\s*,\s*"content"\s*:\s*"(.*?)"\s*}\s*}'},
                {"tool": "navigate", "pattern": r'{\s*"tool"\s*:\s*"navigate"\s*,\s*"parameters"\s*:\s*{\s*"destination"\s*:\s*"(.*?)"\s*}\s*}'},
                {"tool": "analyze_screen", "pattern": r'{\s*"tool"\s*:\s*"analyze_screen"\s*,\s*"parameters"\s*:\s*{\s*"focus_area"\s*:\s*"(.*?)"\s*}\s*}'}
            ]
            
            import re
            for pattern_info in patterns:
                matches = re.search(pattern_info["pattern"], response_text, re.DOTALL)
                if matches:
                    if pattern_info["tool"] == "use_emulator":
                        buttons_str = matches.group(1)
                        buttons = [b.strip(' "\'') for b in buttons_str.split(",")]
                        return {"tool": "use_emulator", "parameters": {"buttons": buttons}}
                    elif pattern_info["tool"] == "update_knowledge":
                        return {"tool": "update_knowledge", "parameters": {"section": matches.group(1), "content": matches.group(2)}}
                    elif pattern_info["tool"] == "navigate":
                        return {"tool": "navigate", "parameters": {"destination": matches.group(1)}}
                    elif pattern_info["tool"] == "analyze_screen":
                        return {"tool": "analyze_screen", "parameters": {"focus_area": matches.group(1)}}
            
            # If no patterns match, return None
            return None
            
        except Exception as e:
            print(f"Error extracting tool call: {e}")
            return None
            
    def _execute_action(self, response):
        """Parse the response to extract and execute the action"""
        # Parse the response to extract the action
        tool_call = self._extract_tool_call(response)
        
        if not tool_call or "tool" not in tool_call:
            print("No valid tool call found in response")
            return "No action taken - couldn't identify a valid tool call"
        
        tool = tool_call.get("tool")
        parameters = tool_call.get("parameters", {})
        
        result = "Action complete"
        
        if tool == "use_emulator":
            buttons = parameters.get("buttons", [])
            print(f"Executing button presses: {buttons}")
            for button in buttons:
                if button.lower() == "wait":
                    time.sleep(0.5)  # Wait command for pauses
                else:
                    self.controller.press_button(button)
                    time.sleep(0.2)  # Give the game time to process each button
            result = f"Buttons pressed: {buttons}"
                
        elif tool == "update_knowledge":
            section = parameters.get("section", "")
            content = parameters.get("content", "")
            print(f"Updating knowledge base section '{section}' with: {content}")
            self.knowledge_base.update_section(section, content)
            result = f"Knowledge base updated: {section}"
            
        elif tool == "navigate":
            destination = parameters.get("destination", "")
            print(f"Navigating to: {destination}")
            # In a complete implementation, this would use pathfinding
            # For now, just simulate movement
            self.controller.press_button("UP")
            time.sleep(0.2)
            self.controller.press_button("RIGHT")
            time.sleep(0.2)
            result = f"Attempted navigation to {destination}"
            
        elif tool == "analyze_screen":
            focus_area = parameters.get("focus_area", "")
            print(f"Analyzing screen area: {focus_area}")
            # In a complete implementation, this would do targeted image analysis
            # For now, just return basic analysis
            if focus_area == "dialog_text":
                result = "Dialog text appears to say: 'Welcome to the world of POKEMON!'"
            else:
                result = f"Screen analysis of {focus_area}: No specific elements detected"
            
        else:
            print(f"Unknown tool: {tool}")
            result = f"Unknown tool: {tool}"
            
        return result
            
    def _update_conversation_history(self, game_state, response, tool_call, result):
        """Add the latest exchange to history"""
        self.conversation_history.append({
            "game_state": game_state,
            "response": response,
            "actions_taken": str(tool_call) if tool_call else "No action",
            "result": result
        })
        
        # Keep only the last 5 exchanges to avoid token limits
        if len(self.conversation_history) > 5:
            self.conversation_history.pop(0)
            
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
                print(f"Error in callback: {e}")