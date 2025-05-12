import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

def organize_dataset():
    """Organize the dataset into train and validation sets."""
    print("Organizing dataset...")

    # Create train and val directories
    for split in ['train', 'val']:
        for emotion in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']:
            os.makedirs(f'data/fer2013/{split}/{emotion}', exist_ok=True)

    # Read the training CSV file
    train_csv_path = 'data/fer2013/train.csv'
    if not os.path.exists(train_csv_path):
        print(f"Error: Could not find {train_csv_path}")
        return

    # Read the test CSV file
    test_csv_path = 'data/fer2013/test.csv'
    if not os.path.exists(test_csv_path):
        print(f"Error: Could not find {test_csv_path}")
        return

    # Process training data
    print("Processing training data...")
    train_df = pd.read_csv(train_csv_path)
    process_dataframe(train_df, 'train')

    # Process test data
    print("Processing test data...")
    test_df = pd.read_csv(test_csv_path)
    process_dataframe(test_df, 'val')

def process_dataframe(df, split):
    """Process a dataframe and save images to appropriate directories."""
    # Emotion mapping
    emotion_map = {
        0: 'angry',
        1: 'disgust',
        2: 'fear',
        3: 'happy',
        4: 'sad',
        5: 'surprise',
        6: 'neutral'
    }

    # Process each image
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Get emotion (if available) and pixels
        emotion = row['emotion'] if 'emotion' in row else None
        pixels = row['pixels']

        # Convert pixels to image
        pixels = np.array(pixels.split(), dtype=np.uint8)
        image = pixels.reshape(48, 48)
        image = Image.fromarray(image)

        # Save image
        if emotion is not None:
            emotion_name = emotion_map[emotion]
            image.save(f'data/fer2013/{split}/{emotion_name}/{idx}.png')
        else:
            # For test set without labels, save to a separate directory
            os.makedirs(f'data/fer2013/{split}/unlabeled', exist_ok=True)
            image.save(f'data/fer2013/{split}/unlabeled/{idx}.png')

def main():
    try:
        organize_dataset()
        print("Dataset preparation complete!")
    except Exception as e:
        print(f"Error during dataset preparation: {str(e)}")
        raise

if __name__ == '__main__':
    main()
