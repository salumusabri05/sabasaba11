import numpy as np
import mediapipe as mp
import cv2
import time
from dataclasses import dataclass

@dataclass
class CameraMetadata:
    """Metadata about the camera source for proper frame preprocessing."""
    facing: str = "user"        # "user" (front) or "environment" (back)
    rotation: int = 0           # Device rotation: 0, 90, 180, 270
    mirrored: bool = False      # Whether the browser already mirrored the frame
    width: int = 640
    height: int = 480

class FramePreprocessor:
    """
    Fixes the 3 critical camera variation issues:
    1. Rotation - phones deliver rotated frames depending on orientation
    2. Mirroring - front cameras may be pre-mirrored by the browser/OS
    3. Aspect ratio - portrait vs landscape changes landmark distributions
    """

    @staticmethod
    def correct_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
        """Correct frame rotation from device orientation."""
        if rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    @staticmethod
    def correct_mirror(frame: np.ndarray, facing: str, already_mirrored: bool) -> np.ndarray:
        """
        Apply correct mirroring based on camera source.
        Front camera ("user"): we WANT a mirror view for natural UX.
        Back camera ("environment"): we DON'T want mirror.
        """
        need_flip = False
        if facing == "user":
            if not already_mirrored:
                need_flip = True
        else:
            if already_mirrored:
                need_flip = True

        return cv2.flip(frame, 1) if need_flip else frame

    @staticmethod
    def normalize_aspect_ratio(frame: np.ndarray, target_ratio: float = 4 / 3) -> np.ndarray:
        """
        Pad frame to match training-data aspect ratio (landscape 4:3).
        This ensures MediaPipe sees the hand in a similar spatial context.
        """
        h, w = frame.shape[:2]
        current_ratio = w / h

        if abs(current_ratio - target_ratio) < 0.05:
            return frame

        if current_ratio < target_ratio:
            # Too tall (portrait) -> pad width
            new_w = int(h * target_ratio)
            pad = (new_w - w) // 2
            frame = cv2.copyMakeBorder(
                frame, 0, 0, pad, new_w - w - pad,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        else:
            # Too wide -> pad height
            new_h = int(w / target_ratio)
            pad = (new_h - h) // 2
            frame = cv2.copyMakeBorder(
                frame, pad, new_h - h - pad, 0, 0,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        return frame

    def preprocess(self, frame: np.ndarray, meta: CameraMetadata) -> np.ndarray:
        """Full preprocessing: rotation -> mirror -> aspect ratio."""
        frame = self.correct_rotation(frame, meta.rotation)
        frame = self.correct_mirror(frame, meta.facing, meta.mirrored)
        frame = self.normalize_aspect_ratio(frame)
        return frame

class SignPredictor:
    """Sign language word prediction with camera-robust preprocessing and MediaPipe Holistic."""

    def __init__(self, model_path: str):
        # Configure Keras to use JAX backend (fast and lightweight CPU execution)
        import os
        os.environ['KERAS_BACKEND'] = 'jax'
        os.environ['JAX_PLATFORMS'] = 'cpu'
        import keras
        self.model = keras.models.load_model(model_path)
        self.classes = ['ALAMA', 'ASUBUHI', 'HABARI', 'JINA', 'JIONI', 'KUJITAMBULISHA', 'LANGU', 'LUGHA', 'MCHANA', 'SHIKAMOO', 'YAKO']

        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.preprocessor = FramePreprocessor()

    @staticmethod
    def normalize_frame(vec: np.ndarray) -> np.ndarray:
        """Normalizes coordinates relative to the body (shoulder distance and center)."""
        vec = vec.copy()
        pose = vec[:15].reshape(5, 3)
        lh = vec[15:78].reshape(21, 3)
        rh = vec[78:].reshape(21, 3)

        if np.any(pose):
            ls = pose[1]  # Left shoulder (index 1)
            rs = pose[2]  # Right shoulder (index 2)

            center = (ls + rs) / 2
            dist = np.linalg.norm(ls - rs)
            if dist > 1e-6:
                pose = (pose - center) / dist
                if np.any(lh):
                    lh = (lh - center) / dist
                if np.any(rh):
                    rh = (rh - center) / dist

        return np.concatenate([
            pose.flatten(),
            lh.flatten(),
            rh.flatten()
        ])

    def extract_features(self, frame: np.ndarray, meta: CameraMetadata = None) -> tuple:
        """Process a frame using MediaPipe Holistic and extract normalized 141-dim feature vector."""
        if meta:
            frame = self.preprocessor.preprocess(frame, meta)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)

        pose = []
        if results.pose_landmarks:
            # We only need pose landmarks: 0, 11, 12, 13, 14
            for idx in [0, 11, 12, 13, 14]:
                lm = results.pose_landmarks.landmark[idx]
                pose.extend([lm.x, lm.y, lm.z])
        else:
            pose = [0] * 15

        lh = []
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                lh.extend([lm.x, lm.y, lm.z])
        else:
            lh = [0] * 63

        rh = []
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                rh.extend([lm.x, lm.y, lm.z])
        else:
            rh = [0] * 63

        raw_vec = np.array(pose + lh + rh, dtype=np.float32)
        features = self.normalize_frame(raw_vec)

        hands_detected = (results.left_hand_landmarks is not None) or (results.right_hand_landmarks is not None)
        return features, hands_detected

    def process_sequence(self, seq: list, target_length: int = 90) -> np.ndarray:
        """
        Process a single landmark sequence according to training rules:
        1. Forward fill internal missing frames to avoid [0,0,0] artificial motion.
        2. Standardize to target_length (pad with zeros if short, temporal sample if long).
        """
        if len(seq) == 0:
            return np.zeros((target_length, 141), dtype=np.float32)
            
        seq = np.array(seq, dtype=np.float32)
        
        # 1. Identify the actual length of the sequence before any trailing zero-padding
        non_zero_frames = [i for i, frame in enumerate(seq) if np.any(frame != 0)]
        if len(non_zero_frames) == 0:
            return np.zeros((target_length, 141), dtype=np.float32)
            
        actual_len = non_zero_frames[-1] + 1
        seq = seq[:actual_len].copy()
        
        # 2. Forward fill internal missing frames
        for i in range(1, len(seq)):
            if np.all(seq[i] == 0):
                seq[i] = seq[i-1]
                
        # 3. Standardize length to target_length
        if len(seq) < target_length:
            padding = np.zeros((target_length - len(seq), 141), dtype=np.float32)
            seq = np.vstack([seq, padding])
        elif len(seq) > target_length:
            indices = np.linspace(0, len(seq) - 1, target_length).astype(int)
            seq = seq[indices]
            
        return seq

    def predict_sequence(self, sequence_buffer: list) -> np.ndarray:
        """Predict probability distribution for a given sequence of features."""
        processed_seq = self.process_sequence(sequence_buffer, target_length=90)
        input_data = np.expand_dims(processed_seq, axis=0)
        pred_probs = self.model.predict(input_data, verbose=0)[0]
        return pred_probs

class TextBuilder:
    """Manages Swahili word-based sentence translation logs and state."""

    def __init__(self):
        self.sentence = ""

    def add_word(self, word: str):
        if not self.sentence:
            self.sentence = word
        else:
            self.sentence += " " + word

    def delete_letter(self):
        # Maps backspace action to removing the last word in a word-based model
        self.delete_word()

    def delete_word(self):
        if self.sentence:
            words = self.sentence.strip().split()
            if words:
                self.sentence = " ".join(words[:-1])

    def clear(self):
        self.sentence = ""

    def get_full_text(self) -> str:
        return self.sentence.strip()
