"""
display.py
Renders the real-time HUD overlay on the OpenCV window.
"""

import cv2
import numpy as np
from typing import Optional


# Colour palette (BGR)
C_GREEN  = (80,  200, 120)
C_AMBER  = (0,   180, 255)
C_WHITE  = (255, 255, 255)
C_DARK   = (20,  20,  20)
C_RED    = (60,  60,  220)
C_TEAL   = (180, 210, 80)


class Display:
    """Draws all HUD elements directly onto the OpenCV frame."""

    WINDOW_NAME = "SignSpeak ✋→🔊"

    def __init__(self):
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)

    def render(
        self,
        frame:           np.ndarray,
        current_label:   Optional[str],
        confidence:      float,
        stable_count:    int,
        commit_frames:   int,
        committed_words: list,
        idle_secs:       float,
        pause_threshold: float,
    ) -> None:
        h, w = frame.shape[:2]

        # ── Semi-transparent top bar ───────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), C_DARK, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # ── App title ─────────────────────────────────────────────────────────
        cv2.putText(frame, "SignSpeak", (16, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, C_TEAL, 2, cv2.LINE_AA)

        # ── Current gesture label ─────────────────────────────────────────────
        label_text = current_label.upper() if current_label else "—"
        label_color = C_GREEN if current_label else (120, 120, 120)
        cv2.putText(frame, label_text, (w // 2 - 120, 55),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, label_color, 2, cv2.LINE_AA)

        # ── Confidence bar ────────────────────────────────────────────────────
        if current_label:
            bar_x, bar_y, bar_w, bar_h = w - 220, 20, 200, 18
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
            filled = int(bar_w * confidence)
            bar_color = C_GREEN if confidence > 0.8 else C_AMBER
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), bar_color, -1)
            cv2.putText(frame, f"{confidence:.0%}", (bar_x + bar_w + 8, bar_y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_WHITE, 1, cv2.LINE_AA)

        # ── Stability progress bar ────────────────────────────────────────────
        if stable_count > 0:
            prog_x, prog_y = w - 220, 48
            prog_w = int(200 * stable_count / commit_frames)
            cv2.rectangle(frame, (prog_x, prog_y), (prog_x + 200, prog_y + 10), (40, 40, 40), -1)
            cv2.rectangle(frame, (prog_x, prog_y), (prog_x + prog_w, prog_y + 10), C_TEAL, -1)
            cv2.putText(frame, "stability", (prog_x + 205, prog_y + 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

        # ── Sentence strip (bottom) ───────────────────────────────────────────
        bottom_overlay = frame.copy()
        cv2.rectangle(bottom_overlay, (0, h - 80), (w, h), C_DARK, -1)
        cv2.addWeighted(bottom_overlay, 0.65, frame, 0.35, 0, frame)

        sentence = " ".join(committed_words) if committed_words else "Start signing..."
        sentence_color = C_WHITE if committed_words else (90, 90, 90)
        cv2.putText(frame, sentence, (20, h - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, sentence_color, 2, cv2.LINE_AA)

        # ── Pause countdown arc ───────────────────────────────────────────────
        if committed_words and idle_secs > 0:
            progress = min(idle_secs / pause_threshold, 1.0)
            angle    = int(360 * progress)
            cx, cy   = w - 45, h - 45
            cv2.ellipse(frame, (cx, cy), (22, 22), -90, 0, angle,
                        C_AMBER if progress < 1.0 else C_GREEN, 3)
            cv2.putText(frame, "🔊" if progress >= 1.0 else f"{pause_threshold - idle_secs:.1f}s",
                        (cx - 14, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_WHITE, 1)

        # ── Controls hint ─────────────────────────────────────────────────────
        cv2.putText(frame, "[Q] Quit   [C] Clear", (16, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)

        cv2.imshow(self.WINDOW_NAME, frame)
