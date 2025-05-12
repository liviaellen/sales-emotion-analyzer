import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
from .emotion_model import EmotionClassifier

def save_pretrained_model():
    """Save a pre-trained ResNet18 model for emotion detection."""
    print("Creating PyTorch model...")
    model = EmotionClassifier()

    # Save the pre-trained model
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'emotion_model.pt')
    torch.save(model.state_dict(), output_path)
    print(f"Saved pre-trained ResNet18 model to {output_path}")
    print("Note: This model will need to be fine-tuned for emotion detection.")

if __name__ == "__main__":
    save_pretrained_model()
