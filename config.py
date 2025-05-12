import os
from pathlib import Path
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
SAVED_MODELS_DIR = os.path.join(ARTIFACTS_DIR, 'saved_models')
LOGS_DIR = os.path.join(ARTIFACTS_DIR, 'logs')

# Device Configuration
DEVICE_TYPE = os.getenv('DEVICE_TYPE', 'auto')  # Options: auto, mps, cuda, cpu
NUM_WORKERS = int(os.getenv('NUM_WORKERS', '4'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '64'))
PIN_MEMORY = os.getenv('PIN_MEMORY', 'true').lower() == 'true'

# Training Configuration
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', '50'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.001'))
MODEL_TYPE = os.getenv('MODEL_TYPE', 'modern')  # Options: modern, regularized, original, transfer
MODEL_TYPE_PROD = os.getenv('MODEL_TYPE_PROD', 'original')  # Model type for production/inference
MODEL_PATH = os.getenv('MODEL_PATH', os.path.join(SAVED_MODELS_DIR, 'original_best_model_20250511_153151.pth'))

# Dataset Configuration
DATA_DIR = os.getenv('DATA_DIR', 'data/fer2013')
CSV_FILE = os.getenv('CSV_FILE', 'fer2013.csv')

# Model Configuration
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))  # Minimum confidence for predictions
SAVE_BEST_ONLY = os.getenv('SAVE_BEST_ONLY', 'true').lower() == 'true'

# Video Analysis Configuration
ANALYSIS_INTERVAL = float(os.getenv('ANALYSIS_INTERVAL', '1.0'))  # Time between frame analyses in seconds

# Create necessary directories
Path(SAVED_MODELS_DIR).mkdir(parents=True, exist_ok=True)
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

def get_device():
    """Get the appropriate device based on configuration and availability."""
    if DEVICE_TYPE == 'auto':
        if torch.backends.mps.is_available():
            return torch.device("mps"), "MPS (Metal Performance Shaders)"
        elif torch.cuda.is_available():
            return torch.device("cuda"), "CUDA"
        else:
            return torch.device("cpu"), "CPU"
    elif DEVICE_TYPE == 'mps' and torch.backends.mps.is_available():
        return torch.device("mps"), "MPS (Metal Performance Shaders)"
    elif DEVICE_TYPE == 'cuda' and torch.cuda.is_available():
        return torch.device("cuda"), "CUDA"
    else:
        return torch.device("cpu"), "CPU"
