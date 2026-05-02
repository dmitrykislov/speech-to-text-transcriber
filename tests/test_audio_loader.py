import pytest
import numpy as np
from pathlib import Path
import tempfile
import soundfile as sf
from unittest.mock import patch
from src.audio_loader import AudioLoader


class TestAudioLoader:
    """Test AudioLoader.load_ogg() functionality."""

    def test_load_ogg_returns_numpy_array(self):
        """load_ogg returns numpy array of audio samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "test.ogg"
            sample_rate = 16000
            duration = 1
            samples = np.random.randn(sample_rate * duration).astype(np.float32)
            sf.write(str(audio_path), samples, sample_rate, subtype="PCM_16")

            result = AudioLoader.load_ogg(audio_path)

            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float32
            assert len(result) > 0

    def test_load_ogg_file_not_found(self):
        """load_ogg raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            AudioLoader.load_ogg("/nonexistent/path/audio.ogg")

    def test_load_ogg_with_pathlib_path(self):
        """load_ogg accepts pathlib.Path objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "test.ogg"
            sample_rate = 16000
            samples = np.random.randn(sample_rate).astype(np.float32)
            sf.write(str(audio_path), samples, sample_rate, subtype="PCM_16")

            result = AudioLoader.load_ogg(audio_path)

            assert isinstance(result, np.ndarray)

    def test_load_ogg_corrupted_file_error(self):
        """load_ogg raises RuntimeError when librosa.load() fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "corrupted.ogg"
            audio_path.write_text("not audio data")

            with patch('librosa.load') as mock_load:
                mock_load.side_effect = Exception("Corrupted audio file")
                with pytest.raises(RuntimeError, match="Failed to load OGG file"):
                    AudioLoader.load_ogg(audio_path)
