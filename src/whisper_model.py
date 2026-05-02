from pathlib import Path
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
