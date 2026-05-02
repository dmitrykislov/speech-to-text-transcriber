import pytest
from pathlib import Path
import tempfile
from src.text_writer import TextWriter


class TestTextWriter:
    """Test TextWriter.save_text() functionality."""

    def test_save_text_creates_file(self):
        """save_text creates output.txt with correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.txt"
            text = "Hello, World!"

            TextWriter.save_text(text, output_path)

            assert output_path.exists()
            assert output_path.read_text(encoding="utf-8") == text

    def test_save_text_creates_directories(self):
        """save_text auto-creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dirs" / "output.txt"
            text = "Test content"

            TextWriter.save_text(text, output_path)

            assert output_path.exists()
            assert output_path.parent.exists()
            assert output_path.read_text(encoding="utf-8") == text

    def test_save_text_utf8_encoding(self):
        """save_text preserves UTF-8 encoded text (Cyrillic)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.txt"
            text = "Привет, мир! Hello, world!"

            TextWriter.save_text(text, output_path)

            assert output_path.read_text(encoding="utf-8") == text

    def test_save_text_with_pathlib_path(self):
        """save_text accepts pathlib.Path objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.txt"
            text = "Content"

            TextWriter.save_text(text, output_path)

            assert output_path.exists()

    def test_save_text_overwrites_existing_file(self):
        """save_text overwrites existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.txt"
            output_path.write_text("Old content", encoding="utf-8")
            new_text = "New content"

            TextWriter.save_text(new_text, output_path)

            assert output_path.read_text(encoding="utf-8") == new_text
