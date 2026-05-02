import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
from src.main import main


class TestMain:
    """Test main() entry point orchestration."""

    def test_main_accepts_input_file_and_output_dir_args(self):
        """main() accepts input_file and output_dir command-line arguments."""
        with patch('src.main.AudioLoader.load_ogg') as mock_load, \
             patch('src.main.WhisperModel') as mock_whisper_class, \
             patch('sys.argv', ['main.py', '/path/to/audio.ogg', '/path/to/output']):
            mock_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            mock_load.return_value = mock_audio
            mock_model = MagicMock()
            mock_whisper_class.return_value = mock_model
            mock_model.transcribe.return_value = {
                'text': 'Hello world',
                'language': 'en',
                'code_switching': False
            }

            result = main()

            assert isinstance(result, dict)
            assert 'text' in result

    def test_main_calls_audio_loader_with_input_file(self):
        """main() calls AudioLoader.load_ogg() with input_file argument."""
        with patch('src.main.AudioLoader.load_ogg') as mock_load, \
             patch('src.main.WhisperModel') as mock_whisper_class, \
             patch('sys.argv', ['main.py', '/path/to/audio.ogg', '/path/to/output']):
            mock_audio = np.array([0.1, 0.2], dtype=np.float32)
            mock_load.return_value = mock_audio
            mock_model = MagicMock()
            mock_whisper_class.return_value = mock_model
            mock_model.transcribe.return_value = {'text': 'test', 'language': 'en', 'code_switching': False}

            main()

            mock_load.assert_called_once()
            call_arg = mock_load.call_args[0][0]
            assert str(call_arg) == '/path/to/audio.ogg'

    def test_main_passes_audio_to_whisper_transcribe(self):
        """main() passes AudioLoader result to WhisperModel.transcribe()."""
        with patch('src.main.AudioLoader.load_ogg') as mock_load, \
             patch('src.main.WhisperModel') as mock_whisper_class, \
             patch('sys.argv', ['main.py', '/path/to/audio.ogg', '/path/to/output']):
            mock_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            mock_load.return_value = mock_audio
            mock_model = MagicMock()
            mock_whisper_class.return_value = mock_model
            mock_model.transcribe.return_value = {'text': 'test', 'language': 'en', 'code_switching': False}

            main()

            mock_model.transcribe.assert_called_once()
            call_arg = mock_model.transcribe.call_args[0][0]
            assert np.array_equal(call_arg, mock_audio)

    def test_main_returns_transcription_dict(self):
        """main() returns transcription output dict with text, language, code_switching."""
        with patch('src.main.AudioLoader.load_ogg') as mock_load, \
             patch('src.main.WhisperModel') as mock_whisper_class, \
             patch('sys.argv', ['main.py', '/path/to/audio.ogg', '/path/to/output']):
            mock_audio = np.array([0.1, 0.2], dtype=np.float32)
            mock_load.return_value = mock_audio
            mock_model = MagicMock()
            mock_whisper_class.return_value = mock_model
            expected_result = {
                'text': 'Привет мир',
                'language': 'ru',
                'code_switching': False
            }
            mock_model.transcribe.return_value = expected_result

            result = main()

            assert result == expected_result
            assert result['text'] == 'Привет мир'
            assert result['language'] == 'ru'
