import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
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

    def test_transcribe_returns_dict_with_text_and_language(self):
        """transcribe() returns dict with 'text' and 'language' fields."""
        with patch('src.whisper_model.FasterWhisperModel') as mock_whisper_class:
            mock_model = MagicMock()
            mock_whisper_class.return_value = mock_model

            mock_segment = MagicMock()
            mock_segment.text = 'Hello world'
            mock_segment.language = 'en'
            mock_info = MagicMock()
            mock_info.language = 'en'
            mock_model.transcribe.return_value = ([mock_segment], mock_info)

            model = WhisperModel()
            result = model.transcribe('test.ogg')

            assert isinstance(result, dict)
            assert 'text' in result
            assert 'language' in result
            assert result['text'] == 'Hello world'
            assert result['language'] == 'en'

    def test_transcribe_detects_russian_language(self):
        """transcribe() correctly detects and preserves Russian language."""
        with patch('src.whisper_model.FasterWhisperModel') as mock_whisper_class:
            mock_model = MagicMock()
            mock_whisper_class.return_value = mock_model

            mock_segment = MagicMock()
            mock_segment.text = 'Привет мир'
            mock_segment.language = 'ru'
            mock_info = MagicMock()
            mock_info.language = 'ru'
            mock_model.transcribe.return_value = ([mock_segment], mock_info)

            model = WhisperModel()
            result = model.transcribe('russian_audio.ogg')

            assert result['language'] == 'ru'
            assert result['text'] == 'Привет мир'

    def test_transcribe_detects_code_switching(self):
        """transcribe() detects code-switching when multiple languages present."""
        with patch('src.whisper_model.FasterWhisperModel') as mock_whisper_class:
            mock_model = MagicMock()
            mock_whisper_class.return_value = mock_model

            mock_segment1 = MagicMock()
            mock_segment1.text = 'Hello '
            mock_segment1.language = 'en'
            mock_segment2 = MagicMock()
            mock_segment2.text = 'привет'
            mock_segment2.language = 'ru'
            mock_info = MagicMock()
            mock_info.language = 'en'
            mock_model.transcribe.return_value = ([mock_segment1, mock_segment2], mock_info)

            model = WhisperModel()
            result = model.transcribe('mixed_audio.ogg')

            assert result['code_switching'] is True
            assert result['text'] == 'Hello привет'
