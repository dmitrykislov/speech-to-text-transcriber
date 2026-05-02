import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.whisper_model import WhisperModel


class TestWhisperModel:
    """Test WhisperModel initialization and caching behavior."""

    def setup_method(self):
        """Reset singleton before each test."""
        WhisperModel._instance = None

    def test_whisper_model_initializes_with_cache_dir(self):
        """WhisperModel initializes with ~/.cache/huggingface cache directory."""
        with patch('src.whisper_model.FasterWhisperModel'):
            model = WhisperModel()
            expected_cache = Path.home() / ".cache" / "huggingface"
            assert model.cache_dir == expected_cache

    def test_whisper_model_downloads_on_first_instantiation(self):
        """WhisperModel downloads model on first instantiation."""
        with patch('src.whisper_model.FasterWhisperModel') as mock_whisper:
            mock_instance = MagicMock()
            mock_whisper.return_value = mock_instance

            model = WhisperModel()

            mock_whisper.assert_called_once()
            call_kwargs = mock_whisper.call_args[1]
            assert call_kwargs['model_size_or_path'] == 'large'
            assert 'download_root' in call_kwargs
            assert model.model == mock_instance

    def test_whisper_model_caches_on_subsequent_instantiation(self):
        """Subsequent instantiations load from cache without re-downloading."""
        with patch('src.whisper_model.FasterWhisperModel') as mock_whisper:
            mock_instance = MagicMock()
            mock_whisper.return_value = mock_instance

            model1 = WhisperModel()
            model2 = WhisperModel()

            assert model1 is model2
            mock_whisper.assert_called_once()
