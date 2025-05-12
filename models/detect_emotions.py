import cv2
import torch
import numpy as np
from .emotion_model import EmotionCNN, EmotionCNNModern, EmotionCNNRegularized, TransferLearningModel
from PIL import Image
import torchvision.transforms as transforms
import os
from collections import deque
import time
from config import ANALYSIS_INTERVAL, MODEL_TYPE, MODEL_PATH, MODEL_TYPE_PROD, CONFIDENCE_THRESHOLD

class SalesCallAnalyzer:
    """A class for analyzing emotions in sales call videos.

    This class provides functionality for:
    - Face detection and tracking in video frames
    - Emotion analysis using a trained deep learning model
    - Engagement scoring based on emotional patterns
    - Real-time visualization of analysis results

    Attributes:
        device (torch.device): Device to run the model on (CPU/GPU)
        model (nn.Module): Loaded emotion detection model
        metadata (dict): Model metadata including architecture and training info
        face_cascade: OpenCV face detector
        transform (transforms.Compose): Image preprocessing pipeline
        selected_face (tuple): Coordinates of the selected face
        tracked_face (tuple): Current tracked face coordinates
        tracker: OpenCV face tracker
        window_size (int): Number of frames to analyze for engagement scoring
        emotion_history (deque): Recent emotion predictions
        start_time (float): Analysis start timestamp
        last_analysis_time (float): Last analysis timestamp
        analysis_interval (float): Analysis interval in seconds
        metrics (dict): Analysis metrics including emotion durations and changes
    """

    def __init__(self, model_path=MODEL_PATH, model_type_prod='original', window_size=30):
        """Initialize the SalesCallAnalyzer.

        Args:
            model_path (str): Path to the trained model file
            model_type_prod (str): Type of model to use ('modern', 'regularized', 'transfer', or 'original')
            window_size (int): Number of frames to consider for engagement scoring
        """
        # Initialize timing variables
        self.start_time = time.time()
        self.last_analysis_time = 0
        self.analysis_interval = ANALYSIS_INTERVAL

        # Initialize face tracking variables
        self.selected_face = None
        self.tracked_face = None

        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.metadata = self.load_model(model_path, model_type_prod)
        self.model.to(self.device)
        print(f"Loaded {model_type_prod} model from {model_path}")
        print(f"Model metadata: {self.metadata}")

        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Define transform based on model type
        self.transform = self.get_preprocessing_pipeline(model_type_prod)

        # Analysis metrics
        self.window_size = window_size  # Number of frames to analyze
        self.emotion_history = deque(maxlen=window_size)
        self.metrics = {
            'positive_time': 0,
            'negative_time': 0,
            'neutral_time': 0,
            'uncertain_time': 0,
            'last_emotion': None,
            'emotion_changes': 0
        }

    def get_model_class(self, model_type_prod):
        """Get the appropriate model class based on production model type.

        Args:
            model_type_prod (str): Type of model to use ('modern', 'regularized', 'transfer', or 'original')

        Returns:
            nn.Module: The model class to use
        """
        model_classes = {
            'modern': EmotionCNNModern,
            'regularized': EmotionCNNRegularized,
            'transfer': TransferLearningModel,
            'original': EmotionCNN
        }

        if model_type_prod not in model_classes:
            raise ValueError(f"Invalid model_type_prod: {model_type_prod}. Must be one of {list(model_classes.keys())}")

        return model_classes[model_type_prod]

    def get_preprocessing_pipeline(self, model_type_prod):
        """Get the appropriate preprocessing pipeline based on model type.

        Args:
            model_type_prod (str): Type of model to use ('modern', 'regularized', 'transfer', or 'original')

        Returns:
            transforms.Compose: The preprocessing pipeline
        """
        # Base transforms for all models
        base_transforms = [
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
        ]

        # Model-specific transforms
        if model_type_prod == 'transfer':
            # Transfer learning model uses ImageNet normalization
            return transforms.Compose(base_transforms + [
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])
        else:
            # All other models use standard normalization
            return transforms.Compose(base_transforms + [
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])

    def load_model(self, model_path, model_type_prod):
        """Load a trained model for prediction.

        Args:
            model_path (str): Path to the saved model file
            model_type_prod (str): Type of model to use ('modern', 'regularized', 'transfer', or 'original')

        Returns:
            tuple: (model, metadata) where model is the loaded PyTorch model and metadata is a dict
        """
        print(f"Loading {model_type_prod} model from {model_path}")

        # Get the appropriate model class
        model_class = self.get_model_class(model_type_prod)
        model = model_class()

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

            # Load the state dictionary
            model.load_state_dict(state_dict)
            model.eval()

            print(f"Successfully loaded {model_type_prod} model")
            return model, metadata

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def preprocess_face(self, face_img):
        """Preprocess face image for model input.

        Args:
            face_img (numpy.ndarray): Face image in BGR or grayscale format

        Returns:
            torch.Tensor: Preprocessed face tensor ready for model input
        """
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_tensor = self.transform(face_img)
        return face_tensor.unsqueeze(0)

    def detect_faces(self, frame):
        """Detect faces in a video frame.

        Args:
            frame (numpy.ndarray): Video frame in BGR format

        Returns:
            tuple: (faces, gray) where faces is a list of face coordinates
                  and gray is the grayscale version of the frame
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,  # Reduced from 1.3 to detect more faces
            minNeighbors=3,   # Reduced from 5 to be more lenient
            minSize=(20, 20)  # Reduced from 30x30 to detect smaller faces
        )

        # Print debug information
        print(f"Detected {len(faces)} faces in frame")
        for (x, y, w, h) in faces:
            print(f"Face at ({x}, {y}) with size {w}x{h}")

        return faces, gray

    def select_face(self, frame, faces):
        """Let user select a face to track from detected faces.

        Args:
            frame (numpy.ndarray): Video frame to display
            faces (list): List of detected face coordinates

        Returns:
            bool: True if a face was selected, False otherwise
        """
        if len(faces) == 0:
            return False

        # For Streamlit, we'll just use the first face
        self.selected_face = faces[0]
        self.tracked_face = self.selected_face  # Store the face coordinates
        self.start_time = time.time()
        return True

    def track_face(self, frame):
        """Track the selected face in subsequent frames.

        Args:
            frame (numpy.ndarray): Current video frame

        Returns:
            tuple: Face coordinates if tracking successful, None otherwise
        """
        if self.tracked_face is None:
            return None

        # Detect faces in the current frame
        faces, _ = self.detect_faces(frame)
        if len(faces) == 0:
            return None

        # Find the face closest to the tracked face
        tracked_x, tracked_y, tracked_w, tracked_h = self.tracked_face
        tracked_center = (tracked_x + tracked_w/2, tracked_y + tracked_h/2)

        min_dist = float('inf')
        closest_face = None

        for face in faces:
            x, y, w, h = face
            center = (x + w/2, y + h/2)
            dist = ((center[0] - tracked_center[0])**2 + (center[1] - tracked_center[1])**2)**0.5

            if dist < min_dist:
                min_dist = dist
                closest_face = face

        # Update tracked face if a close enough face is found
        if min_dist < max(tracked_w, tracked_h):
            self.tracked_face = closest_face
            return closest_face

        return None

    def update_metrics(self, analysis):
        """Update analysis metrics based on emotion prediction.

        Args:
            analysis (dict): Emotion analysis results including:
                - emotion (str): Detected emotion
                - confidence (float): Prediction confidence
                - is_reliable (bool): Whether prediction meets threshold
                - category (str): Emotion category (positive/negative/neutral)
        """
        # Get the time since last analysis
        current_time = time.time()
        time_diff = current_time - self.last_analysis_time

        print(f"\nUpdating metrics:")
        print(f"Time since last analysis: {time_diff:.2f}s")
        print(f"Current emotion: {analysis['emotion']}")
        print(f"Confidence: {analysis['confidence']:.2f}")
        print(f"Is reliable: {analysis['is_reliable']}")

        # Update metrics regardless of reliability
        if analysis['category'] == 'positive':
            self.metrics['positive_time'] += time_diff
            print(f"Added {time_diff:.2f}s to positive time")
        elif analysis['category'] == 'negative':
            self.metrics['negative_time'] += time_diff
            print(f"Added {time_diff:.2f}s to negative time")
        else:  # neutral
            self.metrics['neutral_time'] += time_diff
            print(f"Added {time_diff:.2f}s to neutral time")

        # Update emotion changes if the emotion has changed
        if self.metrics['last_emotion'] != analysis['emotion']:
            self.metrics['emotion_changes'] += 1
            self.metrics['last_emotion'] = analysis['emotion']
            print(f"Emotion changed to: {analysis['emotion']}")

        # Update last analysis time
        self.last_analysis_time = current_time

        # Print current metrics
        print(f"\nCurrent metrics:")
        print(f"Positive time: {self.metrics['positive_time']:.2f}s")
        print(f"Negative time: {self.metrics['negative_time']:.2f}s")
        print(f"Neutral time: {self.metrics['neutral_time']:.2f}s")
        print(f"Emotion changes: {self.metrics['emotion_changes']}")

    def get_engagement_score(self):
        """Calculate engagement score based on emotional patterns.

        The score is calculated using a weighted combination of:
        - Positive emotions (weight: 1.0)
        - Neutral emotions (weight: 0.5)
        - Negative emotions (weight: -0.5)

        Returns:
            float: Engagement score between 0 and 100
        """
        total_time = (self.metrics['positive_time'] + self.metrics['negative_time'] +
                     self.metrics['neutral_time'] + self.metrics['uncertain_time'])

        if total_time == 0:
            return 0

        # Weight positive emotions higher, negative emotions lower
        engagement_score = (
            (self.metrics['positive_time'] * 1.0 +
             self.metrics['neutral_time'] * 0.5 -
             self.metrics['negative_time'] * 0.5) / total_time
        ) * 100

        return min(max(engagement_score, 0), 100)

    def detect_emotion(self, frame):
        """Process a video frame for emotion detection and analysis.

        This method:
        1. Checks if enough time has passed since last analysis
        2. Detects/tracks faces in the frame
        3. Analyzes emotions using the model
        4. Updates engagement metrics
        5. Visualizes results on the frame

        Args:
            frame (numpy.ndarray): Video frame to process

        Returns:
            numpy.ndarray: Frame with visualization overlays
        """
        current_time = time.time()

        # Detect faces in the frame
        faces, gray = self.detect_faces(frame)

        # Draw face boxes for all detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Only analyze if enough time has passed since last analysis
        if current_time - self.last_analysis_time < self.analysis_interval:
            # Just draw the last known emotion if available
            if self.metrics['last_emotion'] is not None:
                cv2.putText(frame, f"Last Emotion: {self.metrics['last_emotion']}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                engagement = self.get_engagement_score()
                cv2.putText(frame, f"Engagement: {engagement:.1f}%", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            return frame

        self.last_analysis_time = current_time

        # Process each detected face
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]

            # Preprocess face
            face_tensor = self.preprocess_face(face_roi)
            face_tensor = face_tensor.to(self.device)

            # Get prediction
            with torch.no_grad():
                outputs = self.model(face_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probabilities, 1)
                emotion_idx = predicted.item()
                confidence = probabilities[0][emotion_idx].item()

                # Map emotion index to label
                emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                emotion = emotion_labels[emotion_idx]

                # Determine emotion category
                if emotion in ['Happy', 'Surprise']:
                    category = 'positive'
                elif emotion in ['Angry', 'Disgust', 'Fear', 'Sad']:
                    category = 'negative'
                else:
                    category = 'neutral'

                # Print detailed prediction information
                print(f"\nEmotion Analysis:")
                print(f"Detected emotion: {emotion}")
                print(f"Confidence: {confidence:.2f}")
                print(f"Category: {category}")
                print(f"Threshold: {CONFIDENCE_THRESHOLD}")
                print(f"Is reliable: {confidence >= CONFIDENCE_THRESHOLD}")

                # Update metrics
                analysis = {
                    'emotion': emotion,
                    'confidence': confidence,
                    'is_reliable': confidence >= CONFIDENCE_THRESHOLD,
                    'category': category
                }
                self.update_metrics(analysis)

                # Draw emotion label
                label = f"{emotion}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Add engagement score
        engagement = self.get_engagement_score()
        cv2.putText(frame, f"Engagement: {engagement:.1f}%", (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame

def process_video(video_path):
    """Process a video file for emotion analysis.
    Takes one frame every ANALYSIS_INTERVAL seconds, analyzes emotions,
    and creates an annotated video showing the results.

    Args:
        video_path (str): Path to the input video file
    """
    try:
        # Initialize analyzer with production model type
        analyzer = SalesCallAnalyzer(
            model_path=MODEL_PATH,
            model_type_prod=MODEL_TYPE_PROD,
            window_size=30
        )
        print(f"Initialized analyzer with model type: {MODEL_TYPE_PROD}")
        print(f"Using model path: {MODEL_PATH}")
        print(f"Analysis interval: {ANALYSIS_INTERVAL}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video file {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        print(f"Video duration: {duration:.1f} seconds")

        # Create videos directory if it doesn't exist
        videos_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'videos')
        os.makedirs(videos_dir, exist_ok=True)

        # Save output in videos directory
        output_filename = f"output_{os.path.basename(video_path)}"
        output_path = os.path.join(videos_dir, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use H.264 codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            raise ValueError(f"Error: Could not create output video file {output_path}")

        # Initialize emotion time tracking
        emotion_times = {
            'Angry': 0.0, 'Disgust': 0.0, 'Fear': 0.0, 'Happy': 0.0,
            'Sad': 0.0, 'Surprise': 0.0, 'Neutral': 0.0
        }

        frame_count = 0
        last_emotion = None
        last_emotion_time = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_count / fps

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = analyzer.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(20, 20)
            )

            # Process each detected face
            for (x, y, w, h) in faces:
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Extract face ROI
                face_roi = gray[y:y+h, x:x+w]

                # Preprocess face
                face_tensor = analyzer.preprocess_face(face_roi)
                face_tensor = face_tensor.to(analyzer.device)

                # Get prediction
                with torch.no_grad():
                    outputs = analyzer.model(face_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(probabilities, 1)
                    emotion_idx = predicted.item()
                    confidence = probabilities[0][emotion_idx].item()

                    # Map emotion index to label
                    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                    emotion = emotion_labels[emotion_idx]

                    # Draw emotion label
                    label = f"{emotion}: {confidence:.2f}"
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Update emotion times
                    if last_emotion is not None and last_emotion != emotion:
                        time_diff = current_time - last_emotion_time
                        emotion_times[last_emotion] += time_diff
                        print(f"Added {time_diff:.1f}s to {last_emotion}")

                    last_emotion = emotion
                    last_emotion_time = current_time

            # Add time stamp
            cv2.putText(frame, f"Time: {current_time:.1f}s", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Write frame to output video
            out.write(frame)
            frame_count += 1

        # Add the last emotion's time
        if last_emotion is not None:
            final_time = duration - last_emotion_time
            emotion_times[last_emotion] += final_time
            print(f"Added final {final_time:.1f}s to {last_emotion}")

        cap.release()
        out.release()

        # Print final statistics
        print("\nEmotion Analysis Results:")
        print(f"Total Duration: {duration:.1f} seconds")
        total_emotion_time = sum(emotion_times.values())
        print(f"Total time with emotions detected: {total_emotion_time:.1f} seconds")
        print("\nEmotion Distribution:")
        for emotion, time in emotion_times.items():
            percentage = (time / total_emotion_time) * 100 if total_emotion_time > 0 else 0
            print(f"{emotion}: {time:.1f} seconds ({percentage:.1f}%)")

        print(f"\nAnnotated video saved as: {output_path}")

        return {
            'emotion_times': emotion_times,
            'output_video': output_path
        }

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'out' in locals() and out.isOpened():
            out.release()
        raise

def main():
    """Main entry point for the script.

    Prompts user for video path and initiates analysis.
    """
    video_path = input("Enter the path to your sales call video: ")
    process_video(video_path)

if __name__ == '__main__':
    main()
