import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from .face_detector import FaceDetector
from models.emotion_model import EmotionClassifier, predict_emotion

class EmotionPredictor:
    def __init__(self, model_path: str):
        """
        Initialize emotion predictor.

        Args:
            model_path: Path to trained PyTorch model
        """
        self.face_detector = FaceDetector()
        self.model = EmotionClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

        # Emotion labels
        self.emotions = {
            0: "Angry",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Neutral",
            5: "Sad",
            6: "Surprise"
        }

        # Store emotion history for each tracked face
        self.emotion_history: Dict[int, List[Dict]] = {}

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Process a single frame to detect faces and predict emotions.

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of dictionaries containing face locations and emotion predictions
        """
        results = []

        # Detect and track faces
        faces = self.face_detector.detect_faces(frame)

        # Get current tracked face IDs
        current_face_ids = list(self.face_detector.tracked_faces.keys())

        # Update emotion history for each tracked face
        for face_id in current_face_ids:
            if face_id not in self.emotion_history:
                self.emotion_history[face_id] = []

        for face_id, face_rect in zip(current_face_ids, faces):
            # Extract face
            face_img = self.face_detector.extract_face(frame, face_rect)
            if face_img is None:
                continue

            # Predict emotion
            emotion_idx, probabilities = predict_emotion(self.model, face_img)

            # Store results
            result = {
                'face_id': face_id,
                'face_rect': face_rect,
                'emotion': self.emotions[emotion_idx],
                'emotion_idx': emotion_idx,
                'probabilities': probabilities
            }

            results.append(result)
            self.emotion_history[face_id].append(result)

        return results

    def get_emotion_summary(self, results: List[Dict], face_id: Optional[int] = None) -> Dict:
        """
        Generate summary statistics for emotion predictions.

        Args:
            results: List of emotion prediction results
            face_id: Optional face ID to get summary for specific person

        Returns:
            Dictionary containing emotion statistics
        """
        if not results:
            return {
                'total_frames': 0,
                'emotion_counts': {emotion: 0 for emotion in self.emotions.values()},
                'emotion_percentages': {emotion: 0.0 for emotion in self.emotions.values()}
            }

        # Filter results by face_id if specified
        if face_id is not None:
            results = [r for r in results if r['face_id'] == face_id]

        # Count emotions
        emotion_counts = {emotion: 0 for emotion in self.emotions.values()}
        for result in results:
            emotion_counts[result['emotion']] += 1

        # Calculate percentages
        total_frames = len(results)
        emotion_percentages = {
            emotion: (count / total_frames) * 100
            for emotion, count in emotion_counts.items()
        }

        return {
            'total_frames': total_frames,
            'emotion_counts': emotion_counts,
            'emotion_percentages': emotion_percentages
        }

    def get_emotion_history(self, face_id: int) -> List[Dict]:
        """
        Get emotion history for a specific tracked face.

        Args:
            face_id: ID of the tracked face

        Returns:
            List of emotion predictions for the face
        """
        return self.emotion_history.get(face_id, [])
