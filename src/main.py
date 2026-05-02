import argparse
import sys
import logging
from pathlib import Path
from src.audio_loader import AudioLoader
from src.whisper_model import WhisperModel
from src.text_writer import TextWriter


def _setup_logging(output_dir: Path) -> None:
    """Configure file and console logging to ~/agent-artifacts/[project-folder]/app.log."""
    log_dir = Path.home() / 'agent-artifacts' / output_dir.name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'app.log'
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


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
    
    _setup_logging(output_path)
    logger = logging.getLogger()
    
    logger.info(f'Starting transcription: input={input_path}, output_dir={output_path}')
    
    logger.info('Loading OGG audio file')
    audio = AudioLoader.load_ogg(input_path)
    logger.info(f'Audio loaded: {len(audio)} samples')
    
    logger.info('Initializing Whisper model')
    model = WhisperModel()
    logger.info('Transcribing audio')
    transcription = model.transcribe(audio)
    logger.info(f'Transcription complete: language={transcription.get("language")}, text_length={len(transcription.get("text", ""))}')
    
    output_file = output_path / 'transcription.txt'
    logger.info(f'Saving transcription to {output_file}')
    TextWriter.save_text(transcription['text'], output_file)
    logger.info('Transcription saved successfully')
    
    return transcription


if __name__ == '__main__':
    result = main()
    print(result)
