import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import logging
from config import MODEL_TYPE, MODEL_PATH, CONFIDENCE_THRESHOLD

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionCNN(nn.Module):
    """Original CNN model for emotion detection."""
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EmotionCNNModern(nn.Module):
    """Modern CNN model with improved architecture."""
    def __init__(self, num_classes=7):
        super(EmotionCNNModern, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Second convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 12 * 12, 512)  # Changed to 12x12 after two pooling layers
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))  # (batch, 32, 48, 48)
        x = F.relu(self.bn2(self.conv2(x)))  # (batch, 64, 48, 48)
        x = self.pool(x)  # (batch, 64, 24, 24)

        # Second block
        x = F.relu(self.bn3(self.conv3(x)))  # (batch, 128, 24, 24)
        x = self.pool(x)  # (batch, 128, 12, 12)

        # Flatten and fully connected layers
        x = x.view(-1, 128 * 12 * 12)  # Flatten to (batch, 128*12*12)
        x = F.relu(self.fc1(x))  # (batch, 512)
        x = self.dropout(x)
        x = self.fc2(x)  # (batch, num_classes)
        return x

class EmotionCNNRegularized(nn.Module):
    """Regularized CNN model with additional regularization techniques."""
    def __init__(self, num_classes=7):
        super(EmotionCNNRegularized, self).__init__()
        # First convolutional block with regularization
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Second convolutional block with regularization
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        # Third convolutional block with regularization
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers with regularization
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        # Third block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)

        # Flatten and fully connected layers
        x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransferLearningModel(nn.Module):
    """Transfer learning model using ResNet50."""
    def __init__(self, num_classes=7):
        super(TransferLearningModel, self).__init__()
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(pretrained=True)

        # Modify first layer to accept grayscale input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

def load_model(model_path, model_type='modern'):
    """Load the appropriate model based on type.

    This is the single source of truth for loading emotion detection models.
    It handles:
    - Model initialization based on type
    - State dictionary loading and preprocessing
    - Metadata extraction
    - Device placement

    Args:
        model_path (str): Path to the saved model file
        model_type (str): Type of model to load ('modern', 'regularized', 'transfer', or 'original')

    Returns:
        tuple: (model, metadata) where model is the loaded PyTorch model and metadata is a dict
    """
    logger.info(f"Loading {model_type} model from {model_path}")

    # Initialize the appropriate model
    if model_type == 'modern':
        model = EmotionCNNModern()
    elif model_type == 'regularized':
        model = EmotionCNNRegularized()
    elif model_type == 'transfer':
        model = TransferLearningModel()
    else:
        model = EmotionCNN()

    try:
        # Load the state dictionary
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        # Handle nested state dictionary structure
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # Remove 'module.' prefix if present (from DataParallel)
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                metadata = checkpoint.get('metadata', {})
            else:
                state_dict = checkpoint
                metadata = {}
        else:
            state_dict = checkpoint
            metadata = {}

        # Load the state dictionary into the model
        model.load_state_dict(state_dict)
        model.eval()

        logger.info(f"Successfully loaded {model_type} model")
        return model, {'type': model_type, **metadata}
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def load_model_for_detection(model_path, model_type='modern'):
    """Load model specifically for emotion detection."""
    return load_model(model_path, model_type)

def predict_emotion(model, face_tensor, confidence_threshold=0.5):
    """Predict emotion from face tensor with confidence threshold."""
    with torch.no_grad():
        outputs = model(face_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        is_reliable = confidence.item() > confidence_threshold

        return {
            'emotion': predicted.item(),
            'confidence': confidence.item(),
            'is_reliable': is_reliable,
            'probabilities': probabilities[0].tolist()
        }

def preprocess_face(face_image):
    """Preprocess face image for model input."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    return transform(face_image).unsqueeze(0)

def save_model(model, model_path, model_type, metadata=None):
    """Save model with metadata.

    Args:
        model: PyTorch model to save
        model_path (str): Path to save the model
        model_type (str): Type of model being saved
        metadata (dict, optional): Additional metadata to save with the model
    """
    if metadata is None:
        metadata = {}

    # Add model type to metadata
    metadata['model_type'] = model_type

    # Save model with metadata
    torch.save({
        'state_dict': model.state_dict(),
        'metadata': metadata
    }, model_path)
    logger.info(f"Saved {model_type} model to {model_path}")
