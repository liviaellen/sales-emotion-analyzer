import cv2
import numpy as np
from typing import Generator, Tuple

class VideoReader:
    def __init__(self, video_path: str, sample_rate: int = 1):
        """
        Initialize video reader.

        Args:
            video_path: Path to the video file
            sample_rate: Process every nth frame (default: 1)
        """
        self.video_path = video_path
        self.sample_rate = sample_rate
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps

    def __iter__(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Iterate through video frames."""
        frame_idx = 0

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_idx % self.sample_rate == 0:
                yield frame_idx, frame

            frame_idx += 1

    def __del__(self):
        """Release video capture when object is destroyed."""
        if hasattr(self, 'cap'):
            self.cap.release()

    def get_frame_at_time(self, timestamp: float) -> np.ndarray:
        """
        Get frame at specific timestamp.

        Args:
            timestamp: Time in seconds

        Returns:
            Frame at specified timestamp
        """
        frame_idx = int(timestamp * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if not ret:
            raise ValueError(f"Could not read frame at timestamp {timestamp}")

        return frame
