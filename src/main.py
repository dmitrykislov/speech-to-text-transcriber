import argparse
import sys
from pathlib import Path
from src.audio_loader import AudioLoader
from src.whisper_model import WhisperModel


def main():
    """Main entry point: load OGG file and transcribe to text."""
    parser = argparse.ArgumentParser(
        description='Convert OGG audio file to text'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to OGG audio file'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Directory to write transcription output'
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_dir)

    audio = AudioLoader.load_ogg(input_path)
    model = WhisperModel()
    transcription = model.transcribe(audio)

    return transcription


if __name__ == '__main__':
    result = main()
    print(result)
