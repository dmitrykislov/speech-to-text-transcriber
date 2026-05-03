import pytest
import numpy as np
from pathlib import Path
import tempfile
import soundfile as sf
from unittest.mock import patch
from src.audio_loader import AudioLoader


def _write_audio(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write samples to path; soundfile picks format/subtype from the extension."""
    sf.write(str(path), samples, sample_rate)


class TestLoadOggBasics:
    """Core return-shape and dtype guarantees of AudioLoader.load_ogg."""

    def test_returns_numpy_float32_array(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / 'test.ogg'
            sample_rate = 16000
            samples = np.random.randn(sample_rate).astype(np.float32) * 0.1
            _write_audio(audio_path, samples, sample_rate)

            result = AudioLoader.load_ogg(audio_path)

            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float32
            assert len(result) > 0

    def test_accepts_pathlib_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / 'test.ogg'
            samples = np.random.randn(16000).astype(np.float32) * 0.1
            _write_audio(audio_path, samples, 16000)

            result = AudioLoader.load_ogg(audio_path)

            assert isinstance(result, np.ndarray)

    def test_accepts_string_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / 'test.ogg'
            samples = np.random.randn(16000).astype(np.float32) * 0.1
            _write_audio(audio_path, samples, 16000)

            result = AudioLoader.load_ogg(str(audio_path))

            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float32

    def test_stereo_input_is_downmixed_to_mono(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / 'stereo.ogg'
            sample_rate = 16000
            stereo = (np.random.randn(sample_rate, 2).astype(np.float32) * 0.1)
            _write_audio(audio_path, stereo, sample_rate)

            result = AudioLoader.load_ogg(audio_path)

            assert result.ndim == 1, 'Expected 1-D mono array after downmix'

    def test_preserves_native_sample_rate_length(self):
        """librosa.load(sr=None) keeps the file's native sample rate, so length should equal samples written."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / 'test.ogg'
            sample_rate = 22050
            duration_s = 2
            samples = np.random.randn(sample_rate * duration_s).astype(np.float32) * 0.1
            _write_audio(audio_path, samples, sample_rate)

            result = AudioLoader.load_ogg(audio_path)

            tolerance = sample_rate // 10
            assert abs(len(result) - sample_rate * duration_s) <= tolerance


class TestLoadOggOtherFormats:
    """Despite the name, load_ogg dispatches to librosa.load and works for other formats too."""

    @pytest.mark.parametrize('extension', ['.wav', '.flac', '.ogg'])
    def test_loads_common_lossless_formats(self, extension):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / f'sample{extension}'
            sample_rate = 16000
            samples = np.random.randn(sample_rate).astype(np.float32) * 0.1
            _write_audio(audio_path, samples, sample_rate)

            result = AudioLoader.load_ogg(audio_path)

            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float32
            assert len(result) > 0


class TestLoadOggErrors:
    """Error-path behaviour."""

    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            AudioLoader.load_ogg('/nonexistent/path/audio.ogg')

    def test_missing_file_error_includes_path(self):
        bad_path = '/nonexistent/audio.ogg'
        with pytest.raises(FileNotFoundError, match='nonexistent'):
            AudioLoader.load_ogg(bad_path)

    def test_corrupted_file_raises_runtime_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / 'corrupted.ogg'
            audio_path.write_text('not audio data')

            with patch('librosa.load') as mock_load:
                mock_load.side_effect = Exception('Corrupted audio file')
                with pytest.raises(RuntimeError, match='Failed to load OGG file'):
                    AudioLoader.load_ogg(audio_path)

    def test_runtime_error_chains_original_exception(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / 'corrupted.ogg'
            audio_path.write_text('garbage')

            original = ValueError('decoder blew up')
            with patch('librosa.load') as mock_load:
                mock_load.side_effect = original
                with pytest.raises(RuntimeError) as excinfo:
                    AudioLoader.load_ogg(audio_path)

            assert excinfo.value.__cause__ is original
