import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

class FaceDetector:
    def __init__(self, cascade_path: str = 'haarcascade_files/haarcascade_frontalface_default.xml'):
        """
        Initialize face detector with tracking capabilities.

        Args:
            cascade_path: Path to Haar cascade XML file
        """
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise ValueError(f"Could not load cascade classifier from {cascade_path}")

        # Initialize face trackers
        self.trackers = {}  # Dictionary to store trackers
        self.next_face_id = 0
        self.max_faces = 3  # Maximum number of faces to track
        self.tracking_threshold = 0.3  # Minimum tracking confidence

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a frame and update tracking.

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of face rectangles (x, y, w, h) with tracking IDs
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Update existing trackers
        active_trackers = {}
        for face_id, tracker in list(self.trackers.items()):
            # Update tracker
            success, bbox = tracker.update(frame)

            if success:
                # Convert bbox to (x, y, w, h)
                x, y, w, h = [int(v) for v in bbox]
                active_trackers[face_id] = (tracker, (x, y, w, h))

        # Detect new faces if we have room for more
        if len(active_trackers) < self.max_faces:
            new_faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            # Filter out faces that overlap with existing tracked faces
            for (x, y, w, h) in new_faces:
                overlap = False
                for _, (_, (tx, ty, tw, th)) in active_trackers.items():
                    if self._calculate_iou((x, y, w, h), (tx, ty, tw, th)) > 0.3:
                        overlap = True
                        break

                if not overlap:
                    # Create new tracker
                    tracker = cv2.TrackerCSRT_create()
                    bbox = (x, y, w, h)
                    tracker.init(frame, bbox)
                    active_trackers[self.next_face_id] = (tracker, (x, y, w, h))
                    self.next_face_id += 1

                    if len(active_trackers) >= self.max_faces:
                        break

        # Update trackers
        self.trackers = {face_id: tracker for face_id, (tracker, _) in active_trackers.items()}

        # Return list of face rectangles
        return [(x, y, w, h) for _, (_, (x, y, w, h)) in active_trackers.items()]

    def _calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0

    def extract_face(self, frame: np.ndarray, face_rect: Tuple[int, int, int, int],
                    target_size: Tuple[int, int] = (48, 48)) -> Optional[np.ndarray]:
        """
        Extract and preprocess a face region.

        Args:
            frame: Input frame (BGR format)
            face_rect: Face rectangle (x, y, w, h)
            target_size: Target size for face image

        Returns:
            Preprocessed face image or None if face is too small
        """
        x, y, w, h = face_rect

        # Skip if face is too small
        if w < 30 or h < 30:
            return None

        # Extract face region
        face_img = frame[y:y+h, x:x+w]

        # Convert to grayscale
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # Resize to target size
        face_resized = cv2.resize(face_gray, target_size)

        return face_resized
