import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import tempfile
from src.main import main, _discover_audio_files, AUDIO_EXTENSIONS


def _make_transcribe_side_effect(text_by_filename=None, default_text='hello'):
    """Return a transcribe() side_effect that returns deterministic per-call output."""
    calls = {'i': 0}

    def _side_effect(audio, language=None):
        calls['i'] += 1
        return {
            'text': f'{default_text} {calls["i"]}',
            'language': 'en',
            'code_switching': False,
        }

    return _side_effect


class TestDiscoverAudioFiles:
    """Test _discover_audio_files() helper."""

    def test_single_file_returns_list_with_that_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio = Path(tmpdir) / 'clip.ogg'
            audio.write_bytes(b'')
            assert _discover_audio_files(audio) == [audio]

    def test_directory_returns_only_audio_files_sorted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / 'b.ogg').write_bytes(b'')
            (d / 'a.mp3').write_bytes(b'')
            (d / 'c.wav').write_bytes(b'')
            (d / 'notes.txt').write_text('not audio')
            (d / 'image.png').write_bytes(b'')

            result = _discover_audio_files(d)

            assert result == [d / 'a.mp3', d / 'b.ogg', d / 'c.wav']

    def test_directory_is_non_recursive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / 'top.ogg').write_bytes(b'')
            sub = d / 'sub'
            sub.mkdir()
            (sub / 'nested.ogg').write_bytes(b'')

            result = _discover_audio_files(d)

            assert result == [d / 'top.ogg']

    def test_empty_directory_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert _discover_audio_files(Path(tmpdir)) == []

    def test_directory_with_no_audio_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / 'a.txt').write_text('x')
            (d / 'b.json').write_text('{}')
            assert _discover_audio_files(d) == []

    def test_extension_match_is_case_insensitive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / 'A.OGG').write_bytes(b'')
            (d / 'B.Mp3').write_bytes(b'')
            result = _discover_audio_files(d)
            assert sorted(p.name for p in result) == ['A.OGG', 'B.Mp3']

    def test_missing_path_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            _discover_audio_files(Path('/definitely/does/not/exist/xyz'))

    def test_supported_extensions_cover_common_formats(self):
        for ext in ('.ogg', '.mp3', '.wav', '.flac', '.m4a', '.opus'):
            assert ext in AUDIO_EXTENSIONS


class TestMainSingleFile:
    """main() called with a single audio-file path."""

    def test_writes_txt_next_to_audio_file_with_same_basename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_file = Path(tmpdir) / 'clip.ogg'
            audio_file.write_bytes(b'')

            with patch('src.main.AudioLoader.load_ogg') as mock_load, \
                 patch('src.main.WhisperModel') as mock_whisper_class, \
                 patch('src.main.TextWriter.save_text') as mock_save, \
                 patch('sys.argv', ['main.py', str(audio_file)]):
                mock_load.return_value = np.array([0.1], dtype=np.float32)
                mock_model = MagicMock()
                mock_whisper_class.return_value = mock_model
                mock_model.transcribe.return_value = {
                    'text': 'hello world', 'language': 'en', 'code_switching': False,
                }

                results = main()

            mock_save.assert_called_once()
            saved_text, saved_path = mock_save.call_args[0]
            assert saved_text == 'hello world'
            assert Path(saved_path) == audio_file.with_suffix('.txt')

            assert len(results) == 1
            assert results[0]['input'] == audio_file
            assert results[0]['output'] == audio_file.with_suffix('.txt')
            assert results[0]['language'] == 'en'


class TestMainDirectory:
    """main() called with a directory containing audio files."""

    def test_transcribes_every_audio_file_in_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            files = [d / 'a.ogg', d / 'b.mp3', d / 'c.wav']
            for f in files:
                f.write_bytes(b'')
            (d / 'readme.txt').write_text('skip me')

            with patch('src.main.AudioLoader.load_ogg') as mock_load, \
                 patch('src.main.WhisperModel') as mock_whisper_class, \
                 patch('src.main.TextWriter.save_text') as mock_save, \
                 patch('sys.argv', ['main.py', str(d)]):
                mock_load.return_value = np.array([0.1], dtype=np.float32)
                mock_model = MagicMock()
                mock_whisper_class.return_value = mock_model
                mock_model.transcribe.side_effect = _make_transcribe_side_effect()

                results = main()

            assert mock_save.call_count == 3
            saved_paths = {Path(c.args[1]) for c in mock_save.call_args_list}
            expected_paths = {f.with_suffix('.txt') for f in files}
            assert saved_paths == expected_paths

            assert len(results) == 3
            assert {r['output'] for r in results} == expected_paths

    def test_whisper_model_loaded_only_once_for_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / 'a.ogg').write_bytes(b'')
            (d / 'b.ogg').write_bytes(b'')

            with patch('src.main.AudioLoader.load_ogg') as mock_load, \
                 patch('src.main.WhisperModel') as mock_whisper_class, \
                 patch('src.main.TextWriter.save_text'), \
                 patch('sys.argv', ['main.py', str(d)]):
                mock_load.return_value = np.array([0.1], dtype=np.float32)
                mock_model = MagicMock()
                mock_whisper_class.return_value = mock_model
                mock_model.transcribe.return_value = {
                    'text': 't', 'language': 'en', 'code_switching': False,
                }

                main()

            assert mock_whisper_class.call_count == 1

    def test_empty_directory_returns_empty_and_does_not_load_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('src.main.WhisperModel') as mock_whisper_class, \
                 patch('src.main.AudioLoader.load_ogg') as mock_load, \
                 patch('src.main.TextWriter.save_text') as mock_save, \
                 patch('sys.argv', ['main.py', tmpdir]):

                results = main()

            assert results == []
            mock_whisper_class.assert_not_called()
            mock_load.assert_not_called()
            mock_save.assert_not_called()

    def test_skips_non_audio_files_in_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / 'a.ogg').write_bytes(b'')
            (d / 'notes.txt').write_text('x')
            (d / 'image.png').write_bytes(b'')
            (d / 'readme.md').write_text('x')

            with patch('src.main.AudioLoader.load_ogg') as mock_load, \
                 patch('src.main.WhisperModel') as mock_whisper_class, \
                 patch('src.main.TextWriter.save_text') as mock_save, \
                 patch('sys.argv', ['main.py', str(d)]):
                mock_load.return_value = np.array([0.1], dtype=np.float32)
                mock_model = MagicMock()
                mock_whisper_class.return_value = mock_model
                mock_model.transcribe.return_value = {
                    'text': 't', 'language': 'en', 'code_switching': False,
                }

                results = main()

            assert mock_save.call_count == 1
            assert len(results) == 1
            assert results[0]['input'] == d / 'a.ogg'
