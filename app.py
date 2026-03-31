"""
SignSpeak — Real-time Sign Language to Speech
Main application entry point.

Pipeline: Camera → MediaPipe Hand Landmarks → Gesture Classifier → Text → TTS
"""

import cv2
import time
import argparse
import numpy as np
from collections import deque

from landmark_extractor import LandmarkExtractor
from gesture_classifier import GestureClassifier
from tts_engine import TTSEngine
from display import Display


# ─── Configuration ────────────────────────────────────────────────────────────

PREDICTION_BUFFER_SIZE = 10       # Frames to smooth prediction over
CONFIDENCE_THRESHOLD   = 0.75     # Minimum confidence to accept a prediction
WORD_COMMIT_FRAMES     = 20       # Stable frames before committing a word
SENTENCE_PAUSE_SEC     = 2.5      # Seconds of no gesture before speaking sentence
CAMERA_INDEX           = 0        # Default webcam


# ─── Main Loop ────────────────────────────────────────────────────────────────

def run(camera_index: int = CAMERA_INDEX, speak: bool = True, demo: bool = False):
    """
    Launch the real-time sign-language-to-speech pipeline.

    Args:
        camera_index: OpenCV camera device index.
        speak:        Whether to output audio via TTS.
        demo:         If True, overlays landmark visualisation (useful for demos).
    """
    extractor   = LandmarkExtractor()
    classifier  = GestureClassifier(model_path="models/gesture_classifier.pkl")
    tts         = TTSEngine(enabled=speak)
    display     = Display()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {camera_index}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prediction_buffer: deque[str] = deque(maxlen=PREDICTION_BUFFER_SIZE)
    committed_words:   list[str]  = []
    current_word:      str        = ""
    stable_count:      int        = 0
    last_gesture_time: float      = time.time()

    print("\n✋  SignSpeak is running — press [Q] to quit, [C] to clear sentence.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: dropped frame.")
            continue

        frame = cv2.flip(frame, 1)   # Mirror for natural feel

        # ── 1. Extract hand landmarks ──────────────────────────────────────────
        landmarks, annotated_frame = extractor.extract(frame, visualise=demo)

        # ── 2. Classify gesture ────────────────────────────────────────────────
        label, confidence = None, 0.0
        if landmarks is not None:
            label, confidence = classifier.predict(landmarks)
            last_gesture_time = time.time()

        # ── 3. Smooth predictions with a majority-vote buffer ─────────────────
        if label and confidence >= CONFIDENCE_THRESHOLD:
            prediction_buffer.append(label)
        else:
            prediction_buffer.append("")   # empty slot keeps buffer rolling

        smoothed_label = _majority_vote(prediction_buffer)

        # ── 4. Word commitment logic ───────────────────────────────────────────
        if smoothed_label:
            if smoothed_label == current_word:
                stable_count += 1
            else:
                current_word = smoothed_label
                stable_count = 1

            if stable_count == WORD_COMMIT_FRAMES:
                committed_words.append(current_word)
                stable_count = 0
                print(f"  ✔  Committed: {current_word}")
        else:
            stable_count = 0

        # ── 5. Sentence flush — speak after pause ──────────────────────────────
        idle_secs = time.time() - last_gesture_time
        if committed_words and idle_secs >= SENTENCE_PAUSE_SEC:
            sentence = " ".join(committed_words)
            print(f"\n🔊  Speaking: \"{sentence}\"\n")
            tts.speak(sentence)
            committed_words.clear()
            current_word = ""

        # ── 6. Render HUD ──────────────────────────────────────────────────────
        display.render(
            frame          = annotated_frame,
            current_label  = smoothed_label,
            confidence     = confidence,
            stable_count   = stable_count,
            commit_frames  = WORD_COMMIT_FRAMES,
            committed_words= committed_words,
            idle_secs      = idle_secs,
            pause_threshold= SENTENCE_PAUSE_SEC,
        )

        # ── 7. Keyboard controls ───────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            committed_words.clear()
            current_word = ""
            print("  ↺  Sentence cleared.")

    cap.release()
    cv2.destroyAllWindows()
    print("SignSpeak closed.")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _majority_vote(buffer: deque) -> str:
    """Return the most common non-empty label in the buffer."""
    counts: dict[str, int] = {}
    for item in buffer:
        if item:
            counts[item] = counts.get(item, 0) + 1
    return max(counts, key=counts.get) if counts else ""


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignSpeak — Sign Language to Speech")
    parser.add_argument("--camera", type=int, default=0,        help="Camera device index (default: 0)")
    parser.add_argument("--no-speak",  action="store_true",     help="Disable TTS audio output")
    parser.add_argument("--demo",      action="store_true",     help="Show landmark skeleton overlay")
    args = parser.parse_args()

    run(camera_index=args.camera, speak=not args.no_speak, demo=args.demo)
