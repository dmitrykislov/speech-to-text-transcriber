from pathlib import Path
from typing import Union
import numpy as np
from faster_whisper import WhisperModel as FasterWhisperModel


class WhisperModel:
    """Wrapper for Faster-Whisper model with ~/.cache/huggingface caching."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._cache_dir = Path.home() / ".cache" / "huggingface"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._model = FasterWhisperModel(
            model_size_or_path="large",
            download_root=str(self._cache_dir),
            device="cpu",
            compute_type="default"
        )
        self._initialized = True

    @property
    def cache_dir(self) -> Path:
        """Return the cache directory path."""
        return self._cache_dir

    @property
    def model(self) -> FasterWhisperModel:
        """Return the underlying Faster-Whisper model instance."""
        return self._model

    def transcribe(self, audio: Union[str, Path, np.ndarray], language: str = None) -> dict:
        """Transcribe audio and detect language, including Russian.

        Args:
            audio: Path to audio file (str or Path) or numpy array of audio samples.
            language: Optional language code (e.g. 'ru' for Russian). If None, auto-detect.

        Returns:
            dict with keys:
            - 'text': Transcribed text
            - 'language': Detected language code (e.g. 'ru', 'en')
            - 'code_switching': bool indicating if multiple languages detected in segments
        """
        segments, info = self._model.transcribe(audio, language=language)

        full_text = ''.join(segment.text for segment in segments)
        detected_language = info.language

        languages_in_segments = set()
        for segment in segments:
            if hasattr(segment, 'language') and segment.language:
                languages_in_segments.add(segment.language)
        code_switching = len(languages_in_segments) > 1

        return {
            'text': full_text,
            'language': detected_language,
            'code_switching': code_switching
        }
