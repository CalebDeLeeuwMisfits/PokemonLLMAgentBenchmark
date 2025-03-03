# Pokémon LLM Agent

This project is an autonomous agent (using Hugging Face smolsgents) designed to play Pokémon Red using a large language model (LLM). The agent interacts with the game through an emulator, analyzes screenshots for game state, manages a knowledge base, and navigates the game world. Currently uses Claude API or Ollama locally. Originally created as part of a Wandering project by team members at Misfits and Machines https://misfitsandmachines.com/ inspired by the work of Anthropic in their now-famous ClaudePlaysPokemon report and livestream https://www.twitch.tv/claudeplayspokemon https://excalidraw.com/#json=WrM9ViixPu2je5cVJZGCe,no_UoONhF6UxyMpTqltYkg  

## Simplified Pokémon LLM Agent

A simplified version with just four core files that can use either the Anthropic API or local LLM inference via Ollama to play Pokémon Red and collect gameplay data.

## Overview

This system will use:
1. An external Pokémon Red emulator. Reddit discussion may offer insights into sourcing, ie: https://www.reddit.com/r/Roms/comments/16bfto5/where_can_i_download_a_pok%C3%A9mon_fire_red_for_an/
2. Either Claude via Anthropic's API or local LLMs via Ollama for decision making
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
  - Supports multiple LLM backends (Claude API or Ollama local inference)
  - Executes actions based on LLM decisions using available tools
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

## LLM Options

The agent supports two LLM providers:

1. **Anthropic Claude (Cloud-based)**
   - Requires an API key
   - Default model: `claude-3-sonnet-20240229`

2. **Ollama (Local inference)**
   - Runs models locally on your machine
   - Default model: `deepseek-coder:16b-instruct`
   - Requires [Ollama](https://ollama.com) to be installed and running

## How to Run

1. **Install Dependencies:**
   ```bash
   # For both providers
   pip install opencv-python pytesseract numpy pillow huggingface_hub datasets
   
   # For Anthropic Claude
   pip install anthropic
   
   # For Ollama local inference
   pip install ollama
   ```

2. **Set up Ollama (for local inference):**

* Install Ollama from ollama.com
* Pull the model: ollama pull deepseek-coder:16b-instruct
* Make sure the Ollama service is running

3. **Setup Environment:**
# For Anthropic Claude
export ANTHROPIC_API_KEY="your-api-key-here"

4. **Run the Agent:**
```
# Using Claude (default)
python main.py

# Using Ollama local inference
python main.py --llm-provider ollama

# Using Ollama with a specific model
python main.py --llm-provider ollama --model-name llama2:13b

# Using Claude with a specific model
python main.py --llm-provider anthropic --model-name claude-3-opus-20240229
```

5. **Run with Dataset Collection:**
```
python main.py --enable-dataset --hf-repo-id "your-username/pokemon-gameplay-dataset" --push-interval 50
```

# For dataset collection (optional)
export HF_TOKEN="your-huggingface-token-here"

# Set paths to emulator and ROM (optional)
export EMULATOR_PATH="path/to/your/emulator"
export ROM_PATH="path/to/pokemon/rom.gb"

**Command Line Arguments:**

--llm-provider: LLM provider to use (anthropic or ollama, default: anthropic)
--model-name: Name of the model to use (defaults to provider-specific values)
--enable-dataset: Enables dataset collection
--dataset-name: Name for your dataset (default: "pokemon_gameplay")
--hf-repo-id: Hugging Face repository ID as "username/repo-name"
--hf-token: Hugging Face API token (can also use HF_TOKEN env var)
--push-interval: Number of samples to collect before pushing to Hugging Face (default: 50)

**Additional Information:**

When using Claude, this implementation defaults to Claude 3.7 Sonnet via the Anthropic API
When using Ollama, this implementation defaults to deepseek-coder:16b-instruct
You need to provide your own Pokémon Red ROM and compatible emulator
The agent interfaces with an existing emulator rather than including the game
For a full implementation, you would need to:
Implement proper screenshot capture from the emulator window
Add emulator-specific input handling
Enhance the OCR and image recognition capabilities
Improve the pathfinding and navigation logic
The collected datasets can be used for:
Training new models on gameplay decisions
Analyzing agent performance and behavior
Creating benchmarks for LLM-based game agents

  ## License

Mozilla Public License (MPL)
