import cv2
import os
import time
import argparse
import numpy as np

# Add src/ to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from landmark_extractor import LandmarkExtractor


def collect(label: str, n_samples: int, data_dir: str, camera_index: int):
    out_dir = os.path.join(data_dir, label)
    os.makedirs(out_dir, exist_ok=True)

    # Start numbering from where we left off
    existing = [f for f in os.listdir(out_dir) if f.endswith(".npy")]
    start_idx = len(existing)

    extractor = LandmarkExtractor()
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    collected = 0
    recording = False
    countdown = 3
    countdown_start = None

    print(f"\n  Collecting {n_samples} samples for gesture: '{label}'")
    print("    Press [SPACE] to start/stop recording, [Q] to quit early.\n")

    while collected < n_samples:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        landmarks, annotated = extractor.extract(frame, visualise=True)

        # Countdown overlay
        if countdown_start is not None:
            elapsed = time.time() - countdown_start
            remaining = countdown - int(elapsed)
            if remaining > 0:
                cv2.putText(annotated, f"Starting in {remaining}...", (50, 100),
                            cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 255, 200), 4)
            else:
                recording = True
                countdown_start = None

        # Status overlay
        status_color = (0, 220, 80) if recording else (0, 140, 255)
        status_text  = f"● REC  {collected}/{n_samples}" if recording else f"READY — SPACE to record"
        cv2.putText(annotated, status_text, (20, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, status_color, 2)
        cv2.putText(annotated, f"Gesture: {label.upper()}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Save sample while recording
        if recording and landmarks is not None:
            idx  = start_idx + collected
            path = os.path.join(out_dir, f"{idx:04d}.npy")
            np.save(path, landmarks)
            collected += 1

        cv2.imshow("Data Collector — SignSpeak", annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" ") and not recording and countdown_start is None:
            countdown_start = time.time()
        elif key == ord(" ") and recording:
            recording = False
            print(f"  ⏸  Paused at {collected}/{n_samples}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Saved {collected} samples for '{label}' → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect gesture training data")
    parser.add_argument("--label",   required=True,             help="Gesture name (e.g. 'hello')")
    parser.add_argument("--samples", type=int, default=200,     help="Number of samples to collect")
    parser.add_argument("--data-dir",default="data/samples",    help="Root data directory")
    parser.add_argument("--camera",  type=int, default=0,       help="Camera device index")
    args = parser.parse_args()

    collect(args.label, args.samples, args.data_dir, args.camera)
