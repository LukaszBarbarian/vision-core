import os
import yaml
from dotenv import load_dotenv
from typing import Dict
from src.config.models.detection import CameraConfiguration

class Settings:
    def __init__(self, config_path: str = "nvr_registry.yaml"):
        self.config_path = config_path
        self.registry: Dict[str, CameraConfiguration] = {}
        
        # Load environment variables from .env file into os.environ
        load_dotenv()
        
        self.load_registry()

    def load_registry(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")
        
        # Read the raw template config file
        with open(self.config_path, "r") as f:
            raw_content = f.read()
        
        # Dynamically inject secret tokens from environment variables
        expanded_content = os.path.expandvars(raw_content)
        
        # Safely parse the complete configuration
        raw_data = yaml.safe_load(expanded_content)
        
        cameras_list = raw_data.get("cameras", []) if raw_data else []
        for item in cameras_list:
            config = CameraConfiguration(**item)
            self.registry[config.camera.id] = config

# Shared singleton instance
settings = Settings()