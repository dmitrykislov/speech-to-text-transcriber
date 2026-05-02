from pathlib import Path


class TextWriter:
    """Write text content to files with UTF-8 encoding."""

    @staticmethod
    def save_text(text: str, output_path: str | Path) -> None:
        """Save text to file with UTF-8 encoding, creating directories as needed.

        Args:
            text: Text content to write.
            output_path: Path where text file will be saved.

        Raises:
            IOError: If file cannot be written.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            path.write_text(text, encoding="utf-8")
        except Exception as e:
            raise IOError(f"Failed to write text to {path}: {e}") from e
