import time
import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
import logging
from pyboy import PyBoy

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Emulator:
    def __init__(self, rom_path):
        """Initialize the PyBoy emulator
        
        Args:
            rom_path: Path to the Pokémon ROM file
        """
        self.rom_path = rom_path
        self.pyboy = None
        
    def start(self):
        """Start the PyBoy emulator"""
        try:
            # Initialize PyBoy with the ROM
            self.pyboy = PyBoy(self.rom_path, window_type="SDL2", game_wrapper=True)
            
            # Disable speed limit for faster emulation
            self.pyboy.set_emulation_speed(1)  # Normal speed; use 0 for unlimited
            
            # Advance a few frames to ensure the game is loaded
            for _ in range(10):
                self.pyboy.tick()
                
            logger.info(f"Started PyBoy emulator with ROM: {self.rom_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start PyBoy emulator: {e}")
            return False
            
    def stop(self):
        """Stop the PyBoy emulator"""
        if self.pyboy:
            try:
                self.pyboy.stop()
                logger.info("PyBoy emulator stopped")
            except Exception as e:
                logger.error(f"Error stopping PyBoy emulator: {e}")
                
    def send_input(self, input_command):
        """Send button input to PyBoy"""
        if not self.pyboy:
            logger.error("PyBoy emulator not initialized")
            return False
            
        try:
            # Map button names to PyBoy button names (PyBoy uses lowercase)
            button_map = {
                "UP": "up",
                "DOWN": "down",
                "LEFT": "left",
                "RIGHT": "right",
                "A": "a",
                "B": "b",
                "START": "start", 
                "SELECT": "select"
            }
            
            if input_command in button_map:
                # Press and release the button
                button_name = button_map[input_command]
                self.pyboy.button(button_name)
                self.pyboy.tick()  # Process at least one frame
                
                logger.info(f"Sent key: {input_command}")
                return True
            else:
                logger.warning(f"Unknown button: {input_command}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending input: {e}")
            return False

    def read_memory(self, address):
        """Read a single byte from memory"""
        if not self.pyboy:
            logger.error("PyBoy emulator not initialized")
            return None
        
        try:
            return self.pyboy.memory[address]
        except Exception as e:
            logger.error(f"Error reading memory at {hex(address)}: {e}")
            return None

    def read_memory_range(self, start_address, length):
        """Read a range of bytes from memory"""
        if not self.pyboy:
            logger.error("PyBoy emulator not initialized")
            return None
        
        try:
            return [self.pyboy.memory[start_address + i] for i in range(length)]
        except Exception as e:
            logger.error(f"Error reading memory range from {hex(start_address)}: {e}")
            return None

    def get_player_position(self):
        """Get the player's current position"""
        x = self.read_memory(PokemonRedMemoryMap.PLAYER_X)
        y = self.read_memory(PokemonRedMemoryMap.PLAYER_Y)
        return x, y

    def get_pokemon_party(self):
        """Get information about the Pokémon party"""
        count = self.read_memory(PokemonRedMemoryMap.PARTY_COUNT)
        if count is None or count == 0:
            return []
        
        species_list = self.read_memory_range(PokemonRedMemoryMap.PARTY_SPECIES, 6)
        if not species_list:
            return []
            
        party = []
        for i in range(min(count, 6)):
            species_id = species_list[i]
            name = PokemonRedMemoryMap.get_pokemon_name(species_id, self.pyboy.memory)
            party.append({"species_id": species_id, "name": name})
        
        return party

    def has_badge(self, badge_index):
        """Check if player has a specific gym badge (0-7)"""
        if badge_index < 0 or badge_index > 7:
            return False
        
        badges = self.read_memory(PokemonRedMemoryMap.BADGE_FLAGS)
        if badges is None:
            return False
            
        return bool(badges & (1 << badge_index))

    def advance_frames(self, count=1, render=True):
        """Advance a specific number of frames"""
        if not self.pyboy:
            return
            
        self.pyboy.tick(count, render)

class Controller:
    def __init__(self, emulator):
        self.emulator = emulator
        self.valid_buttons = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"]
        
    def press_button(self, button):
        """Press a button on the controller"""
        if not isinstance(button, str):
            logger.warning(f"Invalid button type: {type(button)}")
            return False
            
        button = button.upper()
        if button.lower() == "wait":
            # Special case for wait command - just advance frames
            if self.emulator.pyboy:
                self.emulator.pyboy.tick()
                return True
            return False
            
        if button not in self.valid_buttons:
            logger.warning(f"Invalid button: {button}")
            return False
            
        # Send the button press to the emulator
        success = self.emulator.send_input(button)
        return success
        
    def press_sequence(self, button_sequence):
        """Press a sequence of buttons with delays between them"""
        if not isinstance(button_sequence, list):
            logger.warning(f"Button sequence must be a list, got {type(button_sequence)}")
            return False
            
        for button in button_sequence:
            success = self.press_button(button)
            if not success:
                logger.warning(f"Failed to press button: {button}")
                return False
            time.sleep(0.1)  # Small delay between button presses
        return True
        
    def navigate_to(self, x, y, current_x, current_y):
        """Navigate to a specific point using simple pathfinding"""
        # Calculate direction to move
        dx = x - current_x
        dy = y - current_y
        
        # Create sequence of buttons to press
        sequence = []
        
        # Handle horizontal movement
        for _ in range(abs(dx)):
            if dx > 0:
                sequence.append("RIGHT")
            else:
                sequence.append("LEFT")
        
        # Handle vertical movement
        for _ in range(abs(dy)):
            if dy > 0:
                sequence.append("DOWN")
            else:
                sequence.append("UP")
        
        # Execute the sequence
        return self.press_sequence(sequence)


class ScreenCapture:
    def __init__(self, emulator):
        self.emulator = emulator
        self.last_image = None
        self.last_processed_image = None
        
        # Initialize Tesseract OCR path if necessary
        if os.name == 'nt':  # Windows
            try:
                pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            except:
                logger.warning("Couldn't set Tesseract path. Make sure it's installed.")
        
    def capture(self):
        """Capture a screenshot from PyBoy"""
        if not self.emulator.pyboy:
            logger.error("PyBoy emulator not initialized")
            dummy_image = np.zeros((144, 160, 3), dtype=np.uint8)  # GB resolution
            self.last_image = dummy_image
            return dummy_image
            
        try:
            # Get screen as PIL Image from PyBoy
            pil_image = self.emulator.pyboy.screen.image
            
            # Convert to numpy array for OpenCV processing
            self.last_image = np.array(pil_image)
            logger.debug(f"Captured screenshot: {pil_image.width}x{pil_image.height}")
            return self.last_image
            
        except Exception as e:
            logger.error(f"Error capturing screenshot: {e}")
            dummy_image = np.zeros((144, 160, 3), dtype=np.uint8)
            self.last_image = dummy_image
            return dummy_image
    
    # The rest of the class remains the same as in the original
    def process_image(self, image=None):
        """Process the screenshot for better analysis"""
        if image is None:
            if self.last_image is None:
                return None
            image = self.last_image
            
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding for better text detection
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Save processed image
            self.last_processed_image = thresh
            return thresh
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None
        
    def extract_text(self, image=None, region=None):
        """Extract text from the image using OCR"""
        try:
            # Process the image if not already provided
            if image is None:
                processed = self.process_image()
                if processed is None:
                    return ""
            else:
                processed = self.process_image(image)
            
            # Crop to region if specified
            if region is not None:
                x, y, w, h = region
                processed = processed[y:y+h, x:x+w]
            
            # Convert to PIL Image
            pil_image = Image.fromarray(processed)
            
            # Extract text with Tesseract
            text = pytesseract.image_to_string(pil_image, config='--psm 6')
            return text.strip()
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""
    
    def save_screenshot(self, filename):
        """Save the current screenshot to a file"""
        if self.last_image is not None:
            try:
                cv2.imwrite(filename, self.last_image)
                logger.info(f"Screenshot saved to {filename}")
                return True
            except Exception as e:
                logger.error(f"Error saving screenshot: {e}")
        return False
    
    def detect_dialog_box(self, image=None):
        """Detect if a dialog box is present in the image"""
        if image is None:
            if self.last_image is None:
                return False
            image = self.last_image
            
        try:
            # In Pokemon Red/Blue, dialog boxes are typically at the bottom
            # and have a rectangular shape with a distinct border
            height, width = image.shape[:2]
            
            # Check the bottom third of the screen
            bottom_third = image[int(height*2/3):height, :]
            
            # Convert to grayscale
            gray = cv2.cvtColor(bottom_third, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Look for horizontal lines that could be the dialog box
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                    minLineLength=width*0.7, maxLineGap=10)
            
            return lines is not None and len(lines) > 0
            
        except Exception as e:
            logger.error(f"Error detecting dialog box: {e}")
            return False

class PokemonRedMemoryMap:
    """Memory map locations for Pokémon Red"""
    
    # Player information
    PLAYER_X = 0xD362  # X position on map
    PLAYER_Y = 0xD361  # Y position on map
    PLAYER_DIRECTION = 0xD368  # 0 = Down, 4 = Up, 8 = Left, 0C = Right
    CURRENT_MAP = 0xD35E  # Current map ID
    
    # Pokémon party
    PARTY_COUNT = 0xD163  # Number of Pokémon in party
    PARTY_SPECIES = 0xD164  # List of species IDs in party (6 bytes)
    PARTY_DATA = 0xD16B  # Start of party Pokémon data
    
    # Battle information
    BATTLE_TYPE = 0xD057  # Type of battle
    ENEMY_POKEMON_SPECIES = 0xD0B5  # Current enemy species
    ENEMY_POKEMON_LEVEL = 0xD127  # Current enemy level
    
    # Game state flags
    BADGE_FLAGS = 0xD356  # Badges obtained
    EVENT_FLAGS = 0xD747  # Start of event flags (800 bytes)
    ITEM_FLAGS = 0xD31D  # Owned items
    
    @classmethod
    def get_pokemon_name(cls, species_id, memory):
        """Get Pokémon name from species ID using memory map"""
        # Simplified - would need actual implementation
        pokemon_names = {
            1: "BULBASAUR", 4: "CHARMANDER", 7: "SQUIRTLE",
            25: "PIKACHU", 133: "EEVEE"
            # Add more as needed
        }
        return pokemon_names.get(species_id, f"POKEMON_{species_id}")