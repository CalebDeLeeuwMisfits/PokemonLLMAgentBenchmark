# Pokémon LLM Agent

This project is an autonomous agent (using Hugging Face smolagents) designed to play Pokémon Red using a large language model (LLM). The agent interacts with the game through the PyBoy emulator, analyzes screenshots for game state, manages a knowledge base, and navigates the game world. Currently uses Claude API or Ollama locally. Originally created as part of a Wandering project by team members at Misfits and Machines https://misfitsandmachines.com/ inspired by the work of Anthropic in their now-famous ClaudePlaysPokemon report and livestream https://www.twitch.tv/claudeplayspokemon https://excalidraw.com/#json=WrM9ViixPu2je5cVJZGCe,no_UoONhF6UxyMpTqltYkg  

## Simplified Pokémon LLM Agent

A simplified version with just five core files that can use either the Anthropic API or local LLM inference via Ollama to play Pokémon Red and collect gameplay data.

## Overview

This system will use:
1. PyBoy, a Python-based Game Boy emulator integrated directly into the code
2. Either Claude via Anthropic's API or local LLMs via Ollama for decision making
3. Screenshot capture and analysis to understand game state
4. Optional dataset collection to Hugging Face for gameplay recordings
5. Hugging Face smolagents for structured agent-tool interaction

## The Core Files

### 1. main.py
The entry point and orchestrator for the application. It loads environment configurations, initializes system components, and runs the main game loop. This file:
- Initializes the PyBoy emulator with the Pokémon ROM
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
  - Uses smolagents' CodeAgent for structured decision making
  - Maintains agent state and planning capabilities
  - Logs the agent's thinking process and actions

### 3. game_interface.py
Handles all interaction with the Pokémon game through the PyBoy emulator using three classes:
- **Emulator**: Manages the PyBoy emulator instance and provides memory access methods
- **Controller**: Sends button inputs (UP, DOWN, A, B, etc.) to the game
- **ScreenCapture**: Captures screenshots from PyBoy and processes them with OpenCV and pytesseract for image analysis and text recognition
- **PokemonRedMemoryMap**: Provides memory addresses for game state tracking

### 4. pokemon_tools.py
Defines tools that the agent can use to interact with the Pokémon game:
- Uses smolagents' `@tool` decorator pattern for LLM-friendly function access
- Provides structured, well-documented interfaces for game interaction
- Connects the agent to the PyBoy emulator's capabilities
- Enables memory reading and game state analysis

### 5. dataset_manager.py
Manages the collection and uploading of gameplay data:
- **DatasetManager**: Records gameplay sessions as structured datasets:
  - Saves screenshots with corresponding agent reasoning and actions
  - Maintains a local dataset with all gameplay samples
  - Periodically pushes data to Hugging Face Hub for sharing and analysis
  - Organizes data with images, game states, agent reasoning, and actions

## smolagents Integration

This project leverages Hugging Face's smolagents library to create a more robust agent:

### Key Features of smolagents Integration

1. **Structured Tool Use**: Tools are defined using the `@tool` decorator pattern, providing clear documentation and typing for the LLM
2. **CodeAgent Implementation**: Uses smolagents' CodeAgent class for improved reasoning and decision making
3. **Planning Capabilities**: Implements periodic planning steps to help the agent strategize
4. **Model Flexibility**: Supports both Anthropic and Ollama models through smolagents' model interfaces
5. **Well-Defined Input/Output**: Tools have explicit parameter types and return values for reliable agent interaction

### Benefits of smolagents

- **Cleaner Code Organization**: Separates tools into dedicated files with proper structure
- **Better Error Handling**: Tools can provide informative error messages for the agent
- **Improved Planning**: The agent can periodically reflect and plan next steps
- **Enhanced Reasoning**: The CodeAgent has a thought-action-observation loop for better decision quality

## LLM Options

The agent supports two LLM providers through smolagents:

1. **Anthropic Claude (Cloud-based)**
   - Requires an API key
   - Default model: `claude-3-sonnet-20240229`
   - Integrated through smolagents' AnthropicModel

2. **Ollama (Local inference)**
   - Runs models locally on your machine
   - Default model: `deepseek-coder:16b-instruct`
   - Integrated through smolagents' OllamaModel
   - Requires [Ollama](https://ollama.com) to be installed and running

## How to Run

1. **Install Dependencies:**

### Option 1: Using Python Virtual Environment (venv)

```bash
# Create a virtual environment
python -m venv pokemon_env

# Activate the virtual environment
# On Windows
pokemon_env\Scripts\activate
# On macOS/Linux
# source pokemon_env/bin/activate

# Install dependencies
pip install pyboy opencv-python pytesseract numpy pillow huggingface_hub datasets

# Install LLM and agent-specific dependencies
pip install smolagents  # Required for agent functionality
pip install anthropic   # For Claude API
pip install ollama      # For Ollama local inference
```
OR
Conda:
```
# Create a conda environment
conda create -n pokemon_env python=3.9
conda activate pokemon_env

# Install dependencies
pip install pyboy opencv-python pytesseract numpy pillow huggingface_hub datasets

# Install LLM-specific dependencies
pip install smolagents  # Required for agent functionality
pip install anthropic  # For Claude API
pip install ollama     # For Ollama local inference
```
OR
```
# Install Poetry if you don't have it
# On Windows: (run in PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# On macOS/Linux
# curl -sSL https://install.python-poetry.org | python3 -

# Initialize project with Poetry
poetry init --no-interaction
poetry add pyboy opencv-python pytesseract numpy pillow huggingface_hub datasets
poetry add smolagents anthropic ollama

# Run commands through Poetry
poetry run python main.py
```

2. **Install Tesseract OCR:**
For text recognition, install Tesseract OCR:

Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
macOS: brew install tesseract
Linux: sudo apt-get install tesseract-ocr
Set up Ollama (for local inference):
Install Ollama from ollama.com
Pull the model: ollama pull deepseek-coder:16b-instruct
Make sure the Ollama service is running

3. **Set up Ollama (for local inference):**

* Install Ollama from ollama.com
* Pull the model: ollama pull deepseek-coder:16b-instruct
* Make sure the Ollama service is running

4. **Setup Environment:**
```
# For Anthropic Claude
# On Windows
set ANTHROPIC_API_KEY=your-api-key-here
# On macOS/Linux
# export ANTHROPIC_API_KEY="your-api-key-here"

# For dataset collection (optional)
# On Windows
set HF_TOKEN=your-huggingface-token-here
# On macOS/Linux
# export HF_TOKEN="your-huggingface-token-here"

# Set paths to emulator and ROM (optional - may need path to ROM but emulator is built in using pyboy)
# On Windows
set ROM_PATH=path\to\pokemon\rom.gb
# On macOS/Linux
# export EMULATOR_PATH="path/to/your/emulator"
# export ROM_PATH="path/to/pokemon/rom.gb"
```

You'll need to provide your own Pokémon Red ROM file. Due to copyright reasons, we cannot provide the ROM. The ROM file should have a .gb extension. If you own(ed) pokemon Red you may be able to use a third party ROM legitimately depending on your jurisdiction, at your own risk.

5. **Run the Agent:**
```
# Using Claude (default)
python main.py

# Using Ollama local inference
python main.py --llm-provider ollama

# Using Ollama with a specific model
python main.py --llm-provider ollama --model-name llama2:13b

# Using Claude with a specific model
python main.py --llm-provider anthropic --model-name claude-3-opus-20240229

# Run with Dataset Collection
python main.py --enable-dataset --hf-repo-id "your-username/pokemon-gameplay-dataset" --push-interval 50
```

6. **Run with Dataset Collection:**
```
python main.py --enable-dataset --hf-repo-id "your-username/pokemon-gameplay-dataset" --push-interval 50
```

# For dataset collection (optional)
export HF_TOKEN="your-huggingface-token-here"

# Set paths to emulator and ROM (optional)
export EMULATOR_PATH="path/to/your/emulator"
export ROM_PATH="path/to/pokemon/rom.gb"

**Command Line Arguments:**

* --rom: Path to the Pokémon ROM file
* --llm-provider: LLM provider to use (anthropic or ollama, default: anthropic)
* --model-name: Name of the model to use (defaults to provider-specific values)
* --save-screenshots: Save screenshots during gameplay
* --screenshot-interval: Interval between saved screenshots in seconds (default: 10)
* --debug: Enable debug mode with verbose output
* --load-knowledge: Load knowledge base from file
* --save-knowledge: Save knowledge base to file on exit
* --enable-dataset: Enables dataset collection
* --dataset-name: Name for your dataset (default: "pokemon_gameplay")
* --hf-repo-id: Hugging Face repository ID as "username/repo-name"
* --hf-token: Hugging Face API token (can also use HF_TOKEN env var)
* --push-interval: Number of samples to collect before pushing to Hugging Face (default: 50)

## smolagents Implementation Details
Our implementation leverages smolagents to create a more structured and reliable agent:

1. Tool Organization: Each PyBoy interaction is wrapped in a well-documented tool function
2. Planning Capabilities: The agent performs planning steps every 3 actions to strategize
3. Thought-Action-Observation Loop: The agent follows a reasoning process before taking actions
4. Memory Access Tools: Direct memory reading is available as tools with proper documentation
5. Error Handling: Tools provide descriptive errors to help the agent recover and adapt

**Additional Information:**

* When using Claude, this implementation defaults to Claude 3.7 Sonnet via the Anthropic API
* When using Ollama, this implementation defaults to deepseek-coder:16b-instruct
* You need to provide your own Pokémon Red ROM file
* PyBoy's integration allows direct control of the emulator rather than simulating external inputs
Advanced features available through PyBoy integration:
- Direct memory access for detecting game state
- Control over emulation speed
- Option to skip rendering for faster gameplay (useful for trainig)

The collected datasets can be used for:
Training new models on gameplay decisions
Analyzing agent performance and behavior
Creating benchmarks for LLM-based game agents

## Advanced PyBoy Features for Pokémon Red

This project leverages PyBoy's advanced features for better game state understanding:

### Memory Mapping

Instead of relying solely on OCR from screenshots, the agent can directly access game memory:

- Player position and direction
- Current map ID
- Pokémon party composition
- Battle information
- Game progress flags (badges, events, items)

### Performance Optimization

PyBoy supports frame skipping and render control for better performance:

```python
# Skip rendering for performance (use during training)
emulator.pyboy.set_emulation_speed(0)  # Unlimited speed

# Advance multiple frames at once
emulator.pyboy.tick(30, False)  # Advance 30 frames without rendering
```

## Game State Tracking could be aded
For more reliable gameplay:

Track game progress through memory flags
Detect battles through memory rather than image analysis
Access precise Pokémon stats and moves
Monitor inventory items and map location

## What's Still Needed

The code is quite comprehensive already, but you might consider these additions:

1. **Game-specific memory mapping**: Add more Pokémon Red memory locations for detailed state tracking
2. **Save state management**: Use PyBoy's save/load state functionality for experimentation
3. **Advanced battle strategy**: Create battle-specific logic using memory values instead of OCR
4. **Map navigation**: Build a navigation system using known map connections and in-game coordinates
5. **Event flags**: Track completed events through memory flags to avoid repeating actions

The current implementation already handles the core functionality needed to connect PyBoy to Pokémon Red, so these would be enhancements rather than requirements.

## License

Mozilla Public License (MPL)
````
