import subprocess
import time
import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
import platform
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Emulator:
    def __init__(self, emulator_path, rom_path):
        self.emulator_path = emulator_path
        self.rom_path = rom_path
        self.process = None
        self.window_handle = None
        self.system = platform.system()
        
    def start(self):
        """Start the emulator process"""
        try:
            # Launch the emulator with the ROM
            cmd = [self.emulator_path, self.rom_path]
            
            # Use appropriate creation flags for the platform
            if self.system == "Windows":
                self.process = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                self.process = subprocess.Popen(cmd)
                
            logger.info(f"Started emulator with PID: {self.process.pid}")
            
            # Allow time for the emulator window to appear
            time.sleep(2)
            
            # Try to get window handle for screenshot capture
            self._get_window_handle()
            
            return True
        except Exception as e:
            logger.error(f"Failed to start emulator: {e}")
            return False
    
    def _get_window_handle(self):
        """Get handle to the emulator window for screenshot capture"""
        try:
            if self.system == "Windows":
                import win32gui
                import win32process
                
                def callback(hwnd, hwnds):
                    if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        if pid == self.process.pid:
                            hwnds.append(hwnd)
                    return True
                
                hwnds = []
                win32gui.EnumWindows(callback, hwnds)
                
                if hwnds:
                    self.window_handle = hwnds[0]
                    logger.info(f"Found emulator window handle: {self.window_handle}")
            
            # For macOS and Linux, we would use different approaches
            # This would need to be implemented based on the specific emulator
            
        except Exception as e:
            logger.error(f"Failed to get window handle: {e}")
            
    def stop(self):
        """Stop the emulator process"""
        if self.process:
            try:
                self.process.terminate()
                time.sleep(1)
                # Force kill if not terminated
                if self.process.poll() is None:
                    self.process.kill()
                logger.info("Emulator stopped")
            except Exception as e:
                logger.error(f"Error stopping emulator: {e}")
            
    def send_input(self, input_command):
        """
        Send keyboard input to the emulator window.
        In a complete implementation, this would use platform-specific APIs.
        """
        try:
            if self.system == "Windows":
                import win32api
                import win32con
                
                # Map button names to virtual key codes
                key_map = {
                    "UP": win32con.VK_UP,
                    "DOWN": win32con.VK_DOWN,
                    "LEFT": win32con.VK_LEFT,
                    "RIGHT": win32con.VK_RIGHT,
                    "A": ord('Z'),     # Typical key mapping for GBA emulators
                    "B": ord('X'),
                    "START": win32con.VK_RETURN,
                    "SELECT": win32con.VK_RSHIFT
                }
                
                if input_command in key_map and self.window_handle:
                    # Set focus to emulator window
                    import win32gui
                    win32gui.SetForegroundWindow(self.window_handle)
                    
                    # Send keypress
                    key = key_map[input_command]
                    win32api.keybd_event(key, 0, 0, 0)  # Key down
                    time.sleep(0.1)
                    win32api.keybd_event(key, 0, win32con.KEYEVENTF_KEYUP, 0)  # Key up
                    logger.info(f"Sent key: {input_command}")
                    return True
            
            # For now, just log the input for other platforms
            logger.info(f"Would send input to emulator: {input_command}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending input: {e}")
            return False


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
        if platform.system() == "Windows":
            try:
                pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            except:
                logger.warning("Couldn't set Tesseract path. Make sure it's installed.")
        
    def capture(self):
        """
        Capture a screenshot of the emulator window.
        In a complete implementation, this would capture the actual window.
        """
        try:
            if self.emulator.window_handle and platform.system() == "Windows":
                import win32gui
                import win32ui
                from ctypes import windll
                from PIL import Image
                
                # Get window dimensions
                left, top, right, bot = win32gui.GetClientRect(self.emulator.window_handle)
                width = right - left
                height = bot - top
                
                # Get device context
                hwnd_dc = win32gui.GetWindowDC(self.emulator.window_handle)
                mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
                save_dc = mfc_dc.CreateCompatibleDC()
                
                # Create bitmap
                save_bitmap = win32ui.CreateBitmap()
                save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
                save_dc.SelectObject(save_bitmap)
                
                # Copy screen to bitmap
                result = windll.user32.PrintWindow(self.emulator.window_handle, save_dc.GetSafeHdc(), 0)
                
                # Convert to PIL Image
                bmpinfo = save_bitmap.GetInfo()
                bmpstr = save_bitmap.GetBitmapBits(True)
                img = Image.frombuffer(
                    'RGB',
                    (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                    bmpstr, 'raw', 'BGRX', 0, 1)
                
                # Convert to numpy array for OpenCV processing
                self.last_image = np.array(img)
                
                # Clean up
                save_dc.DeleteDC()
                mfc_dc.DeleteDC()
                win32gui.ReleaseDC(self.emulator.window_handle, hwnd_dc)
                win32gui.DeleteObject(save_bitmap.GetHandle())
                
                logger.info(f"Captured screenshot: {width}x{height}")
                return self.last_image
        
        except Exception as e:
            logger.error(f"Error capturing screenshot: {e}")
        
        # Fallback: return a dummy image
        dummy_image = np.zeros((144, 160, 3), dtype=np.uint8)
        self.last_image = dummy_image
        return dummy_image
        
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