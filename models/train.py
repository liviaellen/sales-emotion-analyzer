import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from datetime import datetime
import json
import sys
from tqdm import tqdm
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.emotion_model import EmotionCNN, EmotionCNNRegularized, EmotionCNNModern, save_model
from config import (
    get_device, NUM_WORKERS, BATCH_SIZE, PIN_MEMORY,
    NUM_EPOCHS, LEARNING_RATE, DATA_DIR, CSV_FILE,
    SAVED_MODELS_DIR, LOGS_DIR, MODEL_TYPE, DEVICE_TYPE
)

class EmotionDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        print(f"Loading dataset from {csv_file}...")
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        print(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image pixels and emotion label
        pixels = self.data.iloc[idx, 1]
        emotion = self.data.iloc[idx, 0]

        # Convert pixels to numpy array
        image = np.array(pixels.split(), dtype=np.float32)
        image = image.reshape(48, 48)

        # Convert to tensor and normalize
        image = torch.from_numpy(image).float()
        image = image.unsqueeze(0)  # Add channel dimension (1 channel for grayscale)
        image = image / 255.0  # Normalize

        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)

        return image, emotion

class EmotionTransferModel(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionTransferModel, self).__init__()

        # Load pre-trained ResNet18
        self.base_model = models.resnet18(pretrained=True)

        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Modify the first layer to accept grayscale input
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace the final fully connected layer
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, model_type):
    best_val_acc = 0.0
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': [],
        'model_type': model_type,
        'training_start': datetime.now().strftime('%Y%m%d_%H%M%S')
    }

    # Create timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create log filename with model type prefix
    log_file = os.path.join(LOGS_DIR, f'{model_type}_training_{timestamp}.json')

    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Create progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]')
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{train_loss/train_total:.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # Create progress bar for validation
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Val]')
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f'{val_loss/val_total:.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })

        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']

        # Update learning rate with validation loss
        scheduler.step(val_loss)

        # Log metrics
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        metrics['learning_rates'].append(current_lr)

        # Save metrics to file
        with open(log_file, 'w') as f:
            json.dump(metrics, f, indent=4)

        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS} Summary:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {current_lr:.6f}')

        # Save best model with metadata
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_save_path = os.path.join(SAVED_MODELS_DIR, f'{model_type}_best_model_{timestamp}.pth')
            metadata = {
                'epoch': epoch + 1,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'learning_rate': current_lr,
                'best_val_acc': best_val_acc
            }
            save_model(model, model_save_path, model_type, metadata)
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')

    # Add training end time to metrics
    metrics['training_end'] = datetime.now().strftime('%Y%m%d_%H%M%S')
    return model, metrics

def main():
    # Set up device
    device, device_name = get_device()
    print(f"\n{'='*50}")
    print(f"Training Configuration:")
    print(f"{'='*50}")
    print(f"Device: {device_name}")
    print(f"Model Type: {MODEL_TYPE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Number of Epochs: {NUM_EPOCHS}")
    print(f"Number of Workers: {NUM_WORKERS}")
    print(f"{'='*50}\n")

    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    dataset = EmotionDataset(os.path.join(DATA_DIR, CSV_FILE))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # Initialize model based on type
    print(f"\nInitializing {MODEL_TYPE} model...")
    if MODEL_TYPE == 'transfer':
        model = EmotionTransferModel()
        print("Using Transfer Learning Model (ResNet18-based)")
        print("- Pre-trained ResNet18 as base")
        print("- Modified for grayscale input")
        print("- Fine-tuned final layers")
    elif MODEL_TYPE == 'modern':
        model = EmotionCNNModern()
        print("Using Modern CNN Architecture")
        print("- Enhanced feature extraction")
        print("- Improved regularization")
    elif MODEL_TYPE == 'regularized':
        model = EmotionCNNRegularized()
        print("Using Regularized CNN Architecture")
        print("- Additional dropout layers")
        print("- L2 regularization")
    else:
        model = EmotionCNN()
        print("Using Original CNN Architecture")
        print("- Basic CNN structure")
        print("- Standard training approach")

    model = model.to(device)
    print(f"\nModel moved to {device_name}")
    print(f"Model architecture:\n{model}\n")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    print("\nStarting training...")
    print(f"{'='*50}")
    # Train the model
    model, metrics = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, MODEL_TYPE)

    # Save final model with metadata
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_model_path = os.path.join(SAVED_MODELS_DIR, f'{MODEL_TYPE}_final_model_{timestamp}.pth')
    metadata = {
        'final_epoch': NUM_EPOCHS,
        'final_val_acc': metrics['val_acc'][-1],
        'final_val_loss': metrics['val_loss'][-1],
        'final_train_acc': metrics['train_acc'][-1],
        'final_train_loss': metrics['train_loss'][-1],
        'best_val_acc': max(metrics['val_acc']),
        'training_duration': f"{metrics['training_start']} to {metrics['training_end']}",
        'model_type': MODEL_TYPE,
        'device': device_name,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE
    }
    save_model(model, final_model_path, MODEL_TYPE, metadata)
    print(f'\nTraining completed!')
    print(f'Final model saved to {final_model_path}')
    print(f'Best validation accuracy: {max(metrics["val_acc"]):.2f}%')
    print(f'Training duration: {metrics["training_start"]} to {metrics["training_end"]}')
    print(f"{'='*50}")

if __name__ == '__main__':
    main()
