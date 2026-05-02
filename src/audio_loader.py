import numpy as np
import librosa
from pathlib import Path


class AudioLoader:
    """Load audio files and extract samples as numpy arrays."""

    @staticmethod
    def load_ogg(file_path: str | Path) -> np.ndarray:
        """Load OGG Vorbis audio file and return audio samples.

        Args:
            file_path: Path to OGG audio file.

        Returns:
            numpy array of audio samples (mono, float32).

        Raises:
            FileNotFoundError: If file does not exist.
            RuntimeError: If file cannot be decoded as OGG audio.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        try:
            audio, _ = librosa.load(str(path), sr=None, mono=True)
            return audio.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to load OGG file {path}: {e}") from e
