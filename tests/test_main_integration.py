import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import tempfile
import logging
import soundfile as sf
from src.main import main


def _reset_root_logger():
    """Remove handlers from root logger between tests so log files don't bleed."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass


class TestMainIntegration:
    """Integration tests covering TextWriter + logging + multi-file orchestration."""

    def setup_method(self):
        _reset_root_logger()

    def teardown_method(self):
        _reset_root_logger()

    def test_single_file_writes_txt_alongside_and_creates_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = Path(tmpdir) / 'voicenotes'
            audio_dir.mkdir(parents=True)
            audio_file = audio_dir / 'msg.ogg'
            audio_file.write_bytes(b'')

            with patch('src.main.AudioLoader.load_ogg') as mock_load, \
                 patch('src.main.WhisperModel') as mock_whisper_class, \
                 patch('sys.argv', ['main.py', str(audio_file)]):
                mock_load.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
                mock_model = MagicMock()
                mock_whisper_class.return_value = mock_model
                mock_model.transcribe.return_value = {
                    'text': 'Hello world',
                    'language': 'en',
                    'code_switching': False,
                }

                results = main()

            output_file = audio_file.with_suffix('.txt')
            assert output_file.exists(), f'Output file not created at {output_file}'
            assert output_file.read_text(encoding='utf-8') == 'Hello world'
            assert results[0]['output'] == output_file

            log_dir = Path.home() / 'agent-artifacts' / audio_dir.name
            log_file = log_dir / 'app.log'
            assert log_file.exists(), f'Log file not created at {log_file}'
            log_content = log_file.read_text(encoding='utf-8')
            assert 'Found 1 audio file(s) to transcribe' in log_content
            assert 'Loading audio' in log_content
            assert 'Transcribing audio' in log_content
            assert 'Saving transcription' in log_content
            assert 'Transcription saved successfully' in log_content

    def test_directory_input_transcribes_all_audio_files_alongside_themselves(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir) / 'batch'
            d.mkdir()
            audio_files = [d / 'first.ogg', d / 'second.mp3', d / 'third.wav']
            for f in audio_files:
                f.write_bytes(b'')
            (d / 'unrelated.txt').write_text('keep me')

            with patch('src.main.AudioLoader.load_ogg') as mock_load, \
                 patch('src.main.WhisperModel') as mock_whisper_class, \
                 patch('sys.argv', ['main.py', str(d)]):
                mock_load.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
                mock_model = MagicMock()
                mock_whisper_class.return_value = mock_model
                texts = iter(['transcript one', 'transcript two', 'transcript three'])
                mock_model.transcribe.side_effect = lambda audio, language=None: {
                    'text': next(texts), 'language': 'en', 'code_switching': False,
                }

                results = main()

            assert len(results) == 3
            for f in audio_files:
                txt = f.with_suffix('.txt')
                assert txt.exists(), f'Missing output for {f}'
                assert txt.read_text(encoding='utf-8').startswith('transcript')

            assert (d / 'unrelated.txt').read_text() == 'keep me'

    def test_e2e_russian_audio_transcription_produces_valid_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir) / 'ru_batch'
            d.mkdir()

            audio_file = d / 'russian_sample.ogg'
            sample_rate = 16000
            num_samples = sample_rate * 1
            audio_data = np.random.randn(num_samples).astype(np.float32) * 0.1
            sf.write(str(audio_file), audio_data, sample_rate, format='OGG')

            with patch('src.main.WhisperModel') as mock_whisper_class:
                mock_model = MagicMock()
                mock_whisper_class.return_value = mock_model
                expected_russian_text = 'Привет, это тестовое сообщение на русском языке.'
                mock_model.transcribe.return_value = {
                    'text': expected_russian_text,
                    'language': 'ru',
                    'code_switching': False,
                }

                with patch('sys.argv', ['main.py', str(audio_file)]):
                    results = main()

            output_file = audio_file.with_suffix('.txt')
            assert output_file.exists(), f'Output file not created at {output_file}'

            output_text = output_file.read_text(encoding='utf-8')
            assert output_text == expected_russian_text
            assert 'Привет' in output_text
            assert 'русском' in output_text
            assert '�' not in output_text
            assert results[0]['language'] == 'ru'
