import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import tempfile
import logging
import soundfile as sf
from src.main import main


class TestMainIntegration:
    """Integration tests for main() with TextWriter and logging."""

    def test_main_calls_text_writer_and_creates_log_file(self):
        """main() calls TextWriter.save_text() and creates log file in ~/agent-artifacts/[project-folder]/app.log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test_output'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with patch('src.main.AudioLoader.load_ogg') as mock_load, \
                 patch('src.main.WhisperModel') as mock_whisper_class, \
                 patch('src.main.TextWriter.save_text') as mock_save, \
                 patch('sys.argv', ['main.py', '/path/to/audio.ogg', str(output_dir)]):
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
                
                mock_save.assert_called_once()
                call_args = mock_save.call_args[0]
                assert call_args[0] == 'Hello world'
                assert str(call_args[1]).endswith('transcription.txt')
                
                log_dir = Path.home() / 'agent-artifacts' / output_dir.name
                log_file = log_dir / 'app.log'
                assert log_file.exists(), f'Log file not created at {log_file}'
                log_content = log_file.read_text(encoding='utf-8')
                assert 'Starting transcription' in log_content
                assert 'Loading OGG audio file' in log_content
                assert 'Transcribing audio' in log_content
                assert 'Saving transcription' in log_content
                assert 'Transcription saved successfully' in log_content

    def test_e2e_russian_audio_transcription_produces_valid_output(self):
        """E2E test: main() with real Russian OGG audio produces UTF-8 output with Cyrillic text."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create synthetic Russian OGG audio file
            audio_file = tmpdir_path / 'russian_sample.ogg'
            sample_rate = 16000
            duration_sec = 1
            num_samples = sample_rate * duration_sec
            audio_data = np.random.randn(num_samples).astype(np.float32) * 0.1
            sf.write(str(audio_file), audio_data, sample_rate, format='OGG')
            
            output_dir = tmpdir_path / 'output'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with patch('src.main.WhisperModel') as mock_whisper_class:
                mock_model = MagicMock()
                mock_whisper_class.return_value = mock_model
                expected_russian_text = 'Привет, это тестовое сообщение на русском языке.'
                mock_model.transcribe.return_value = {
                    'text': expected_russian_text,
                    'language': 'ru',
                    'code_switching': False
                }
                
                with patch('sys.argv', ['main.py', str(audio_file), str(output_dir)]):
                    result = main()
                
                output_file = output_dir / 'transcription.txt'
                assert output_file.exists(), f'Output file not created at {output_file}'
                
                output_text = output_file.read_text(encoding='utf-8')
                assert output_text == expected_russian_text, f'Output text mismatch: {output_text!r}'
                
                # Verify Cyrillic characters are present and not corrupted
                assert 'Привет' in output_text, 'Cyrillic text corrupted or missing'
                assert 'русском' in output_text, 'Cyrillic text corrupted or missing'
                
                # Verify no mojibake (encoding corruption markers)
                assert '\ufffd' not in output_text, 'Output contains replacement character (mojibake)'
                assert result['language'] == 'ru'
