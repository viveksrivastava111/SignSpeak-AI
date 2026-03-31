"""
tts_engine.py
Text-to-speech output using pyttsx3 (offline) with gTTS fallback.
"""

import threading


class TTSEngine:
    """
    Speaks text asynchronously so the video loop is never blocked.

    Primary engine  : pyttsx3  (offline, no API key needed)
    Fallback engine : gTTS     (online, better voice quality)
    """

    def __init__(self, enabled: bool = True, rate: int = 160, volume: float = 1.0):
        self._enabled = enabled
        self._lock    = threading.Lock()

        if not enabled:
            print("[TTS] Audio output disabled.")
            return

        self._engine = self._init_pyttsx3(rate, volume)

    # ── Public ─────────────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Speak *text* in a background thread (non-blocking)."""
        if not self._enabled or not text.strip():
            return
        thread = threading.Thread(target=self._speak_blocking, args=(text,), daemon=True)
        thread.start()

    # ── Private ────────────────────────────────────────────────────────────────

    def _speak_blocking(self, text: str) -> None:
        with self._lock:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as e:
                print(f"[TTS] pyttsx3 error: {e}. Trying gTTS fallback...")
                self._gtts_fallback(text)

    def _gtts_fallback(self, text: str) -> None:
        try:
            from gtts import gTTS
            import tempfile, os
            tts = gTTS(text=text, lang="en")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                os.system(f"mpg123 -q {fp.name}")
                os.unlink(fp.name)
        except Exception as e:
            print(f"[TTS] gTTS fallback also failed: {e}")

    @staticmethod
    def _init_pyttsx3(rate: int, volume: float):
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate",   rate)
        engine.setProperty("volume", volume)
        # Prefer a female voice if available
        voices = engine.getProperty("voices")
        for voice in voices:
            if "female" in voice.name.lower() or "zira" in voice.id.lower():
                engine.setProperty("voice", voice.id)
                break
        return engine
