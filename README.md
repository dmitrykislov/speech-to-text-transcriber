# speech-to-text-transcriber

Offline speech-to-text CLI using OpenAI's Whisper (large-v3) via faster-whisper. Drop in an audio file and get a UTF-8 transcript. Auto-detects 99 languages, runs on CPU (Apple Silicon friendly), no GPU required.

## Features

- **Single file or whole directory**: pass a path; if it's a directory every audio file inside is transcribed in one batch (Whisper model loaded once and reused).
- **Multi-format input**: OGG, MP3, WAV, FLAC, M4A, Opus, AAC, AIFF, AU, WMA — anything `librosa.load()` can decode.
- **Transcripts saved next to the source audio**: `clip.ogg` → `clip.txt` in the same directory.
- **Multilingual**: Whisper auto-detects 99 languages; Russian (Cyrillic) verified end-to-end.
- **Code-switching field**: reports whether multiple languages were detected.
- **CPU-only**: runs on Apple Silicon and other CPUs, no GPU required; model cached after first download.
- **UTF-8 output**: Cyrillic and other non-ASCII text round-trip cleanly.

## Architecture

```
Input: file or directory
      ↓
_discover_audio_files (filters by extension)
      ↓
For each audio file:
    AudioLoader (librosa) → numpy float32 mono
        ↓
    WhisperModel (faster-whisper, large-v3, CPU)
        ↓
    {text, language, code_switching}
        ↓
    TextWriter (UTF-8) → <audio_basename>.txt next to source
```

## Technology Stack

- **Language**: Python 3.14
- **Audio decoding**: librosa ≥0.10.0 (libsndfile + audioread/ffmpeg fallback)
- **Transcription model**: faster-whisper ≥0.10.0 running OpenAI Whisper **large-v3**
- **Numerical computing**: numpy ≥1.24.0
- **Model caching**: `~/.cache/huggingface` (auto-downloaded on first run)
- **Logging**: Python logging module (file + console)

## Installation

### Prerequisites
- Python 3.14
- pip or uv package manager

### Setup

```bash
# Clone or navigate to project directory
cd speech-to-text-transcriber

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m pytest -v
```

## Usage

### Basic Usage

```bash
# Single file: transcript saved to <audio_basename>.txt next to it
python -m src.main <audio_file>

# Directory: every audio file inside (non-recursive) is transcribed,
# each transcript saved as <audio_basename>.txt next to its source
python -m src.main <directory>
```

### Examples

```bash
# Single file
python -m src.main ~/Downloads/voicenote.ogg
# → ~/Downloads/voicenote.txt

# Batch a directory of voice notes
python -m src.main ~/Downloads/voicenotes/
# → ~/Downloads/voicenotes/<each>.txt
```

### Command-line Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `input_path` | str (path) | Audio file OR directory containing audio files (required). Directory scan is non-recursive. |

## Supported Audio Formats

The CLI dispatches to `librosa.load()`, so any format librosa can decode works. The discovered extensions for directory mode are:

`.ogg`, `.oga`, `.opus`, `.mp3`, `.wav`, `.flac`, `.m4a`, `.aac`, `.aiff`, `.au`, `.wma`

- **Sample rate**: Any (librosa keeps native rate; faster-whisper resamples internally)
- **Channels**: Mono or stereo (converted to mono automatically)
- **Bit depth**: 16-bit, 24-bit, 32-bit float supported

> Note: MP3 / M4A / WMA may need `ffmpeg` on PATH (audioread fallback). OGG / WAV / FLAC work out of the box via libsndfile.

## Output

Each audio file produces a sibling `.txt` file with the same basename:

```
~/Downloads/clip.ogg          →  ~/Downloads/clip.txt
~/Downloads/voicenotes/a.mp3  →  ~/Downloads/voicenotes/a.txt
~/Downloads/voicenotes/b.wav  →  ~/Downloads/voicenotes/b.txt
```

Files are UTF-8 plain text.

**Example output** (Russian audio):
```
Привет, это тестовое сообщение на русском языке.
```

### Return Value (Python API)

`main()` returns a list of dicts, one per transcribed file:

```python
[
    {
        'input': Path,            # Source audio path
        'output': Path,           # Path to written .txt
        'text': str,              # Transcribed text
        'language': str,          # Detected language code (e.g., 'ru', 'en')
        'code_switching': bool,   # True if multiple languages detected
    },
    ...
]
```

Empty list if the directory contains no recognized audio files.

## Logging

### Log File Location

```
~/agent-artifacts/<dir_name>/app.log
```

`<dir_name>` is the input directory's name (or the parent directory's name when a single file is passed).

### Log Levels

- **File**: DEBUG (all messages)
- **Console**: INFO (progress messages only)

### Example Log Output

```
2026-05-03 14:22:15 - INFO - Found 3 audio file(s) to transcribe
2026-05-03 14:22:15 - INFO - Initializing Whisper model
2026-05-03 14:22:45 - INFO - [1/3] Loading audio: /path/to/voicenotes/a.ogg
2026-05-03 14:22:45 - INFO - Audio loaded: 160000 samples
2026-05-03 14:22:45 - INFO - Transcribing audio
2026-05-03 14:23:10 - INFO - Transcription complete: language=ru, text_length=245
2026-05-03 14:23:10 - INFO - Saving transcription to /path/to/voicenotes/a.txt
2026-05-03 14:23:10 - INFO - Transcription saved successfully
2026-05-03 14:23:10 - INFO - [2/3] Loading audio: /path/to/voicenotes/b.mp3
...
```

## Language Support

- **Auto-detection**: Whisper automatically detects language from audio
- **Russian (ru)**: Fully supported with Cyrillic character handling
- **English (en)**: Fully supported
- **99+ languages**: Supported by Whisper large-v3
- **Code-switching**: Detects and reports when multiple languages appear in same audio

## Model Details

- **Model**: OpenAI Whisper **large-v3** (HF id `Systran/faster-whisper-large-v3`)
- **Size**: ~3 GB (downloaded to `~/.cache/huggingface` on first run)
- **Device**: CPU (works on Apple Silicon; CUDA users can change `device` in `whisper_model.py`)
- **Compute type**: `default` — on CPU this is converted from fp16 to fp32 at load time
- **First run**: model download may take 2–5 minutes depending on network

## Troubleshooting

### FileNotFoundError: Input path not found
- Verify the path (file or directory) is correct and exists
- Use absolute paths if relative paths fail

### Directory passed but nothing transcribed
- The directory scan is non-recursive — files in subdirectories are ignored
- Check that file extensions match the supported list (case-insensitive)

### RuntimeError: Failed to load OGG file
- For non-OGG formats (MP3, M4A, WMA), make sure `ffmpeg` is on PATH
- Check the file isn't corrupted: `file <filename>` should report a recognizable audio container

### IOError: Failed to write text to ...
- Verify the audio file's parent directory is writable
- Check disk space is available

### Model download fails
- Check internet connection
- Verify ~/.cache/huggingface directory is writable
- Model will be cached after first successful download

## Testing

```bash
# Run all tests
pytest -v

# Run integration tests only
pytest -v tests/test_main_integration.py
```

## Requirements

- **REQ-001**: Convert OGG audio file to text _(critical)_ ✓
- **REQ-002**: Support Russian language speech recognition _(critical)_ ✓
- **REQ-003**: Achieve highest quality transcription _(critical)_ ✓
- **REQ-004**: Run on M1 Max hardware _(critical)_ ✓
- **REQ-005**: Select optimal model for M1 Max _(high)_ ✓
- **REQ-006**: Output as text file _(critical)_ ✓
- **REQ-007**: Handle OGG audio format _(critical)_ ✓
- **REQ-008**: Minimize external dependencies _(high)_ ✓
