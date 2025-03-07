import os
import time
import sys
import logging
import argparse
import datetime
from pathlib import Path
import threading
import queue
from typing import Optional
from game_interface import Emulator, Controller, ScreenCapture, PokemonRedMemoryMap
from agent import PokemonAgent, KnowledgeBase
from dataset_manager import DatasetManager
from agent import PokemonAgent, KnowledgeBase, PokemonTools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pokemon_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_screenshot_directory():
    """Create a directory for saving screenshots with timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshots_dir = Path(f"screenshots_{timestamp}")
    if not screenshots_dir.exists():
        screenshots_dir.mkdir(parents=True)
    return screenshots_dir

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Run the Pokémon LLM Agent with PyBoy")
    
    parser.add_argument("--rom", type=str, help="Path to Pokémon ROM file")
    parser.add_argument("--save-screenshots", action="store_true", help="Save screenshots during gameplay")
    parser.add_argument("--screenshot-interval", type=float, default=10.0, 
                        help="Interval between saved screenshots in seconds (default: 10)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose output")
    parser.add_argument("--load-knowledge", type=str, help="Load knowledge base from file")
    parser.add_argument("--save-knowledge", type=str, help="Save knowledge base to file on exit")
    parser.add_argument("--llm-provider", type=str, default="anthropic", 
                        choices=["anthropic", "ollama"], help="LLM provider to use")
    parser.add_argument("--model-name", type=str, help="Name of the model to use")
    
    # Add dataset arguments
    parser.add_argument("--enable-dataset", action="store_true", help="Enable dataset collection")
    parser.add_argument("--dataset-name", type=str, default="pokemon_gameplay", help="Name for the dataset")
    parser.add_argument("--hf-repo-id", type=str, help="Hugging Face repository ID (username/repo-name)")
    parser.add_argument("--hf-token", type=str, help="Hugging Face API token (or use HF_TOKEN env var)")
    parser.add_argument("--push-interval", type=int, default=50, 
                      help="Number of samples to collect before pushing to Hugging Face")
    
    return parser.parse_args()

class VisualizationManager:
    """Manages the visualization of the agent's actions and game state"""
    def __init__(self, save_screenshots: bool = False, screenshot_dir: Optional[Path] = None):
        self.save_screenshots = save_screenshots
        self.screenshot_dir = screenshot_dir
        self.screenshot_count = 0
        self.last_screenshot_time = 0
        self.agent_thought_queue = queue.Queue()
        self.visualization_thread = None
        self.running = False

    def start(self):
        """Start the visualization thread"""
        self.running = True
        self.visualization_thread = threading.Thread(target=self._visualization_loop)
        self.visualization_thread.daemon = True
        self.visualization_thread.start()
        
    def stop(self):
        """Stop the visualization thread"""
        self.running = False
        if self.visualization_thread:
            self.visualization_thread.join(timeout=1.0)
    
    def _visualization_loop(self):
        """Main loop for visualization updates"""
        while self.running:
            try:
                thought = self.agent_thought_queue.get(timeout=0.1)
                self._display_agent_thought(thought)
                self.agent_thought_queue.task_done()
            except queue.Empty:
                pass
            time.sleep(0.05)
    
    def _display_agent_thought(self, thought_data):
        """Display the agent's thought process"""
        print("\n" + "="*70)
        print(f"STEP {thought_data.get('step', '?')} | {thought_data.get('timestamp', '')}")
        print("-"*70)
        print(f"GAME STATE: {thought_data.get('game_state', '')[:100]}...")
        print("\nAGENT REASONING:")
        print(thought_data.get('reasoning', '')[:200] + "..." if len(thought_data.get('reasoning', '')) > 200 else thought_data.get('reasoning', ''))
        print("\nACTION:")
        print(thought_data.get('action', 'No action'))
        print("="*70)
    
    def save_screenshot(self, screenshot, force=False, interval=10.0):
        """Save a screenshot if enabled and interval has passed"""
        if not self.save_screenshots or not self.screenshot_dir:
            return
            
        current_time = time.time()
        if force or (current_time - self.last_screenshot_time >= interval):
            try:
                filename = self.screenshot_dir / f"screenshot_{self.screenshot_count:05d}.png"
                import cv2
                cv2.imwrite(str(filename), screenshot)
                self.screenshot_count += 1
                self.last_screenshot_time = current_time
                logger.info(f"Saved screenshot to {filename}")
            except Exception as e:
                logger.error(f"Error saving screenshot: {e}")
    
    def enqueue_agent_thought(self, thought_data):
        """Add agent thought data to the visualization queue"""
        thought_data['timestamp'] = datetime.datetime.now().strftime("%H:%M:%S")
        self.agent_thought_queue.put(thought_data)

def load_knowledge_base(file_path):
    """Load knowledge base from a file"""
    try:
        kb = KnowledgeBase()
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r') as f:
                import json
                data = json.load(f)
                for section, content in data.items():
                    kb.update_section(section, content)
            logger.info(f"Loaded knowledge base from {file_path}")
        return kb
    except Exception as e:
        logger.error(f"Error loading knowledge base: {e}")
        return KnowledgeBase()  # Return a fresh knowledge base on error

def save_knowledge_base(knowledge_base, file_path):
    """Save knowledge base to a file"""
    if not file_path:
        return
        
    try:
        import json
        with open(file_path, 'w') as f:
            json.dump(knowledge_base.sections, f, indent=2)
        logger.info(f"Saved knowledge base to {file_path}")
    except Exception as e:
        logger.error(f"Error saving knowledge base: {e}")

def main():
    args = parse_arguments()
    
    # Set debug level if specified
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Load configuration
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if args.llm_provider == "anthropic" and not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required when using Anthropic")
    
    # Set up ROM path, preferring command line argument over environment variable
    rom_path = args.rom or os.environ.get("ROM_PATH")
    if not rom_path:
        raise ValueError("ROM path must be provided (--rom argument or ROM_PATH environment variable)")
    
    # Set up screenshot directory if saving is enabled
    screenshot_dir = None
    if args.save_screenshots:
        screenshot_dir = setup_screenshot_directory()
        logger.info(f"Screenshots will be saved to {screenshot_dir}")
    
    # Initialize visualization manager
    viz_manager = VisualizationManager(
        save_screenshots=args.save_screenshots,
        screenshot_dir=screenshot_dir
    )
    
    # Initialize dataset manager if enabled
    dataset_manager = None
    if args.enable_dataset:
        # Get token from args or environment
        hf_token = args.hf_token or os.environ.get("HF_TOKEN")
        
        if args.hf_repo_id and hf_token:
            dataset_dir = Path(f"datasets/{args.dataset_name}")
            dataset_manager = DatasetManager(
                dataset_name=args.dataset_name,
                dataset_dir=dataset_dir,
                push_interval=args.push_interval,
                hf_token=hf_token,
                hf_repo_id=args.hf_repo_id
            )
            logger.info(f"Dataset collection enabled: {args.hf_repo_id}")
        else:
            logger.warning("Dataset collection requested but missing required parameters (--hf-repo-id and --hf-token or HF_TOKEN env var)")

    # Initialize components
    logger.info("Initializing PyBoy emulator...")
    emulator = Emulator(rom_path)
    controller = Controller(emulator)
    screen_capture = ScreenCapture(emulator)
    
    # Initialize Pokemon tools
    pokemon_tools = PokemonTools(emulator, controller, screen_capture)
    
    # Load knowledge base
    knowledge_base = load_knowledge_base(args.load_knowledge)
    
    # Initialize the agent with a callback to the visualization manager
    # Caused errors with duplicate variables in past smolagents projects
    logger.info("Initializing Pokémon agent...")
    agent = PokemonAgent(
        api_key=api_key, 
        pokemon_tools=pokemon_tools,
        knowledge_base=knowledge_base,
        llm_provider=args.llm_provider,
        model_name=args.model_name
    )

    
    # Add visualization callback to the agent
    def agent_thought_callback(step_count, game_state, response, action, result):
        viz_manager.enqueue_agent_thought({
            'step': step_count,
            'game_state': game_state,
            'reasoning': response,
            'action': action,
            'result': result
        })
        
        # Add to dataset if enabled
        if dataset_manager:
            # Get current screenshot
            current_screenshot = screen_capture.last_image
            if current_screenshot is not None:
                dataset_manager.add_sample(
                    screenshot=current_screenshot,
                    game_state=game_state,
                    reasoning=response,
                    action=action,
                    result=result
                )
        
    agent.register_callback(agent_thought_callback)
    
    print("Starting Pokémon LLM Agent with PyBoy...")
    viz_manager.start()
    emulator.start()
    time.sleep(1)  # Give emulator time to initialize
    
    screenshot_interval = args.screenshot_interval
    
    try:
        # Take an initial screenshot to verify everything is working
        initial_screenshot = screen_capture.capture()
        viz_manager.save_screenshot(initial_screenshot, force=True)
        
        # Main game loop
        logger.info("Entering main game loop")
        while True:
            # Get game state
            screenshot = screen_capture.capture()
            
            # Periodically save screenshots if enabled
            viz_manager.save_screenshot(screenshot, interval=screenshot_interval)
            
            # Let the agent make a decision
            agent.process_game_state(screenshot)
            
            # Brief pause to not overwhelm the system
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        logger.info("\nReceived interrupt signal, stopping agent...")
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
    finally:
        # Save knowledge base if requested
        if args.save_knowledge:
            save_knowledge_base(knowledge_base, args.save_knowledge)
        
        # Push final dataset updates if enabled
        if dataset_manager:
            dataset_manager.push_to_hub()
            logger.info(f"Dataset stats: {dataset_manager.get_stats()}")
        
        # Clean shutdown
        viz_manager.stop()
        emulator.stop()
        logger.info("Agent stopped. Game progress has been saved.")

if __name__ == "__main__":
    main()