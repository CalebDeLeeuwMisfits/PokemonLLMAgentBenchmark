# Pokémon LLM Agent

This project is an autonomous agent designed to play Pokémon Red using a large language model (LLM). The agent interacts with the game through an emulator, analyzes screenshots for game state, manages a knowledge base, and navigates the game world. Originally created as part of a Wandering project at Misfits and Machines https://misfitsandmachines.com/

## Simplified Pokémon LLM Agent

A simplified version with just four core files that can use the Anthropic API to play Pokémon Red and collect gameplay data.

## Overview

This system will use:
1. An external Pokémon Red emulator
2. Claude via Anthropic's API for decision making
3. Screenshot capture and analysis to understand game state
4. Optional dataset collection to Hugging Face for gameplay recordings

## The Core Files

### 1. main.py
The entry point and orchestrator for the application. It loads environment configurations, initializes system components, and runs the main game loop. This file:
- Starts the emulator process
- Creates the controller and screen capture interfaces
- Establishes the knowledge base
- Runs the continuous loop that captures screenshots and feeds them to the agent
- Handles graceful shutdown on interruption

### 2. agent.py
Contains the core intelligence of the system with two main classes:
- **KnowledgeBase**: Manages the agent's memory of game state, locations, Pokémon team, objectives, and game controls in a structured format
- **PokemonAgent**: The decision-making engine that:
  - Processes screenshots to understand the current state
  - Calls Claude's API with contextual prompts
  - Executes actions based on Claude's decisions using available tools
  - Maintains conversation history for context
  - Logs the agent's thinking process and actions

### 3. game_interface.py
Handles all interaction with the Pokémon emulator through three classes:
- **Emulator**: Manages starting and stopping the emulator process
- **Controller**: Sends button inputs (UP, DOWN, A, B, etc.) to the game
- **ScreenCapture**: Captures screenshots from the emulator window and processes them with OpenCV and pytesseract for image analysis and text recognition

### 4. dataset_manager.py
Manages the collection and uploading of gameplay data:
- **DatasetManager**: Records gameplay sessions as structured datasets:
  - Saves screenshots with corresponding agent reasoning and actions
  - Maintains a local dataset with all gameplay samples
  - Periodically pushes data to Hugging Face Hub for sharing and analysis
  - Organizes data with images, game states, agent reasoning, and actions

## How to Run

1. **Install Dependencies:**
   ```bash
   pip install anthropic opencv-python pytesseract numpy pillow huggingface_hub datasets
   ```

2. **Set up Environment:**
   ```bash
   # Set your Anthropic API key
   export ANTHROPIC_API_KEY="your-api-key-here"
   
   # Set paths to emulator and ROM (optional)
   export EMULATOR_PATH="path/to/your/emulator"
   export ROM_PATH="path/to/pokemon/rom.gb"
   ```

3. **Run the Agent:**
   ```bash
   python main.py
   ```

## Additional Information

- This implementation uses Claude 3.7 Sonnet via the Anthropic API
- You need to provide your own Pokémon Red ROM and compatible emulator
- The agent interfaces with an existing emulator rather than including the game
- For a full implementation, you would need to:
  - Implement proper screenshot capture from the emulator window
  - Add emulator-specific input handling
  - Enhance the OCR and image recognition capabilities
  - Improve the pathfinding and navigation logic

  ## License

Mozilla Public License (MPL)
