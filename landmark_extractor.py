"""
landmark_extractor.py
Wraps MediaPipe Hands to extract 21-point 3D hand landmarks per frame.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional


class LandmarkExtractor:
    """
    Detects hands in a BGR frame and returns normalised landmark vectors.

    Each hand produces 21 landmarks × 3 coordinates (x, y, z) = 63 floats.
    Coordinates are normalised relative to the bounding box of the hand,
    making the representation scale- and position-invariant.
    """

    def __init__(
        self,
        max_num_hands:       int   = 1,
        min_detection_conf:  float = 0.7,
        min_tracking_conf:   float = 0.6,
    ):
        self._mp_hands    = mp.solutions.hands
        self._mp_drawing  = mp.solutions.drawing_utils
        self._mp_styles   = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode       = False,
            max_num_hands           = max_num_hands,
            min_detection_confidence= min_detection_conf,
            min_tracking_confidence = min_tracking_conf,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract(
        self,
        frame: np.ndarray,
        visualise: bool = False,
    ) -> tuple[Optional[np.ndarray], np.ndarray]:
        """
        Process a single BGR frame.

        Returns
        -------
        landmarks : np.ndarray of shape (63,) or None if no hand detected.
        annotated : BGR frame (with or without skeleton overlay).
        """
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        annotated = frame.copy()

        if not results.multi_hand_landmarks:
            return None, annotated

        # Use first detected hand only
        hand_landmarks = results.multi_hand_landmarks[0]

        if visualise:
            self._mp_drawing.draw_landmarks(
                annotated,
                hand_landmarks,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_styles.get_default_hand_landmarks_style(),
                self._mp_styles.get_default_hand_connections_style(),
            )

        landmarks = self._normalise(hand_landmarks)
        return landmarks, annotated

    # ── Private ────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(hand_landmarks) -> np.ndarray:
        """
        Convert MediaPipe landmarks to a normalised flat vector.

        Steps:
          1. Extract raw (x, y, z) from all 21 landmarks.
          2. Translate so wrist (landmark 0) is the origin.
          3. Scale by the maximum absolute coordinate so values ∈ [-1, 1].
        """
        raw = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32,
        )  # shape: (21, 3)

        # Centre on wrist
        raw -= raw[0]

        # Normalise scale
        scale = np.max(np.abs(raw))
        if scale > 1e-6:
            raw /= scale

        return raw.flatten()   # shape: (63,)
