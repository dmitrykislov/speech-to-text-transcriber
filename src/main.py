import argparse
import logging
from pathlib import Path
from src.audio_loader import AudioLoader
from src.whisper_model import WhisperModel
from src.text_writer import TextWriter


AUDIO_EXTENSIONS = {
    '.ogg', '.oga', '.opus',
    '.mp3', '.wav', '.flac',
    '.m4a', '.aac', '.aiff', '.au', '.wma',
}


def _setup_logging(input_path: Path) -> None:
    """Configure file and console logging to ~/agent-artifacts/<name>/app.log."""
    log_name = input_path.name if input_path.is_dir() else input_path.parent.name
    log_dir = Path.home() / 'agent-artifacts' / (log_name or 'transcriber')
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


def _discover_audio_files(input_path: Path) -> list[Path]:
    """Return audio files at input_path. If a file: [input_path]. If a dir: sorted audio files inside (non-recursive)."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        return [input_path]

    return sorted(
        p for p in input_path.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def main():
    """Transcribe an audio file or every audio file in a directory."""
    parser = argparse.ArgumentParser(
        description='Transcribe an audio file or every audio file in a directory'
    )
    parser.add_argument(
        'input_path',
        type=str,
        help='Audio file or directory containing audio files'
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)

    _setup_logging(input_path)
    logger = logging.getLogger()

    audio_files = _discover_audio_files(input_path)
    if not audio_files:
        logger.warning(f'No audio files found at {input_path}')
        return []

    logger.info(f'Found {len(audio_files)} audio file(s) to transcribe')
    logger.info('Initializing Whisper model')
    model = WhisperModel()

    results = []
    for i, audio_file in enumerate(audio_files, 1):
        logger.info(f'[{i}/{len(audio_files)}] Loading audio: {audio_file}')
        audio = AudioLoader.load_ogg(audio_file)
        logger.info(f'Audio loaded: {len(audio)} samples')

        logger.info('Transcribing audio')
        transcription = model.transcribe(audio)
        logger.info(
            f'Transcription complete: language={transcription.get("language")}, '
            f'text_length={len(transcription.get("text", ""))}'
        )

        output_file = audio_file.with_suffix('.txt')
        logger.info(f'Saving transcription to {output_file}')
        TextWriter.save_text(transcription['text'], output_file)
        logger.info('Transcription saved successfully')

        results.append({
            'input': audio_file,
            'output': output_file,
            **transcription,
        })

    return results


if __name__ == '__main__':
    results = main()
    for r in results:
        print(f"{r['input']} -> {r['output']} ({r.get('language')})")
