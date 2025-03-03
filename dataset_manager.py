import os
import time
from pathlib import Path
import logging
import json
import shutil
from PIL import Image
import numpy as np
from datasets import Dataset, load_from_disk
from huggingface_hub import HfApi, Repository

logger = logging.getLogger(__name__)

class DatasetManager:
    """Manages the collection and uploading of gameplay data to Hugging Face datasets"""
    
    def __init__(self, 
                 dataset_name: str,
                 dataset_dir: Path = None, 
                 push_interval: int = 50,
                 hf_token: str = None,
                 hf_repo_id: str = None):
        """
        Initialize the dataset manager
        
        Args:
            dataset_name: Name for the dataset
            dataset_dir: Directory to store dataset locally
            push_interval: How many samples to collect before pushing to HF
            hf_token: Hugging Face API token
            hf_repo_id: Full Hugging Face repo ID (username/repo-name)
        """
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir or Path(f"dataset_{dataset_name}")
        self.images_dir = self.dataset_dir / "images"
        self.push_interval = push_interval
        self.hf_token = hf_token
        self.hf_repo_id = hf_repo_id
        
        self.samples_since_push = 0
        self.total_samples = 0
        self.enabled = bool(hf_repo_id and hf_token)
        
        # Statistics
        self.start_time = time.time()
        
        if self.enabled:
            self._initialize_dataset()
    
    def _initialize_dataset(self):
        """Initialize the dataset structure"""
        # Create directories
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        try:
            # Try to load existing dataset
            self.dataset = load_from_disk(str(self.dataset_dir))
            self.total_samples = len(self.dataset)
            logger.info(f"Loaded existing dataset with {self.total_samples} samples")
        except Exception:
            # Create new dataset if loading fails
            self.dataset = Dataset.from_dict({
                "image_path": [],
                "game_state": [],
                "agent_reasoning": [],
                "action_taken": [],
                "result": [],
                "timestamp": []
            })
            self.dataset.save_to_disk(str(self.dataset_dir))
            logger.info(f"Created new dataset at {self.dataset_dir}")
            
        # Initialize Hugging Face repo if needed
        if self.enabled and self.hf_repo_id:
            try:
                self.api = HfApi(token=self.hf_token)
                
                # Check if repo exists, create if it doesn't
                try:
                    self.api.repo_info(repo_id=self.hf_repo_id)
                    logger.info(f"Connected to existing HF repo: {self.hf_repo_id}")
                    # Clone repo
                    self.repo = Repository(
                        local_dir=str(self.dataset_dir),
                        clone_from=self.hf_repo_id,
                        use_auth_token=self.hf_token
                    )
                except Exception:
                    # Create new repo
                    self.api.create_repo(
                        repo_id=self.hf_repo_id,
                        exist_ok=True,
                        private=True
                    )
                    logger.info(f"Created new HF repo: {self.hf_repo_id}")
                    self.repo = Repository(
                        local_dir=str(self.dataset_dir),
                        repo_type="dataset",
                        use_auth_token=self.hf_token
                    )
                    # Initialize git
                    self.repo.git_init()
            except Exception as e:
                logger.error(f"Failed to initialize HF repo: {e}")
                self.enabled = False
    
    def add_sample(self, screenshot, game_state, reasoning, action, result):
        """Add a new sample to the dataset"""
        if not self.enabled:
            return
        
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(screenshot, np.ndarray):
                screenshot = Image.fromarray(screenshot)
            
            # Save image
            image_filename = f"sample_{self.total_samples:06d}.png"
            image_path = self.images_dir / image_filename
            screenshot.save(image_path)
            
            # Add to dataset
            self.dataset = self.dataset.add_item({
                "image_path": str(image_path.relative_to(self.dataset_dir)),
                "game_state": game_state[:1000] if game_state else "",  # Limit length
                "agent_reasoning": reasoning[:2000] if reasoning else "",  # Limit length
                "action_taken": str(action)[:500] if action else "",
                "result": result[:500] if result else "",
                "timestamp": time.time()
            })
            
            self.samples_since_push += 1
            self.total_samples += 1
            
            # Save dataset locally
            self.dataset.save_to_disk(str(self.dataset_dir))
            
            # Push to HF if interval reached
            if self.samples_since_push >= self.push_interval:
                self.push_to_hub()
                
            return True
        except Exception as e:
            logger.error(f"Error adding sample to dataset: {e}")
            return False
    
    def push_to_hub(self):
        """Push the dataset to Hugging Face Hub"""
        if not self.enabled or not self.hf_repo_id:
            return False
        
        try:
            # Add all files to git
            self.repo.git_add(".")
            
            # Commit
            self.repo.git_commit(f"Update dataset: {self.samples_since_push} new samples")
            
            # Push
            self.repo.git_push()
            
            logger.info(f"Pushed {self.samples_since_push} samples to {self.hf_repo_id}")
            self.samples_since_push = 0
            return True
        except Exception as e:
            logger.error(f"Error pushing to Hugging Face Hub: {e}")
            return False
    
    def get_stats(self):
        """Get statistics about the dataset"""
        return {
            "total_samples": self.total_samples,
            "samples_since_push": self.samples_since_push,
            "runtime_seconds": time.time() - self.start_time,
            "dataset_directory": str(self.dataset_dir),
            "huggingface_repo": self.hf_repo_id if self.enabled else None
        }