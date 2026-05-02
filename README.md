# OGG-to-Text Converter

Convert OGG Vorbis audio files to text using OpenAI's Whisper model with support for Russian language and code-switching detection.

## Features

- **OGG Vorbis audio support**: Decode OGG audio files using librosa
- **Multilingual transcription**: Automatic language detection with explicit Russian support
- **Code-switching detection**: Identifies when multiple languages appear in the same audio
- **High-quality output**: Uses Whisper's large model for optimal transcription accuracy
- **CPU-optimized**: Runs on CPU (M1 Max compatible) with cached model downloads
- **UTF-8 text output**: Proper encoding for Cyrillic and international characters

## Architecture

```
Input OGG File
      ↓
AudioLoader (librosa)
      ↓
numpy float32 array (mono, original sample rate)
      ↓
WhisperModel (faster-whisper, large model, CPU)
      ↓
Transcription dict: {text, language, code_switching}
      ↓
TextWriter (UTF-8)
      ↓
transcription.txt in output_dir
```

## Technology Stack

- **Language**: Python 3.11
- **Audio processing**: librosa ≥0.10.0 (OGG decoding via libsndfile)
- **Transcription model**: faster-whisper ≥0.10.0 (OpenAI Whisper large model)
- **Numerical computing**: numpy ≥1.24.0
- **Model caching**: ~/.cache/huggingface (auto-downloaded on first run)
- **Logging**: Python logging module (file + console)

## Installation

### Prerequisites
- Python 3.11
- pip or uv package manager

### Setup

```bash
# Clone or navigate to project directory
cd ogg-to-text

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m pytest -v
```

## Usage

### Basic Usage

```bash
python -m src.main <input_ogg_file> <output_directory>
```

### Example

```bash
# Transcribe Russian audio
python -m src.main ./samples/russian_speech.ogg ./output

# Output: ./output/transcription.txt
```

### Command-line Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `input_file` | str (path) | Path to OGG Vorbis audio file (required) |
| `output_dir` | str (path) | Directory where transcription.txt will be written (required) |

## Supported Audio Formats

- **OGG Vorbis** (.ogg, .oga): Primary supported format
- **Sample rate**: Any (librosa resamples to model's native rate internally)
- **Channels**: Mono or stereo (converted to mono automatically)
- **Bit depth**: 16-bit, 24-bit, 32-bit float supported

## Output

### Output File Location

```
<output_dir>/transcription.txt
```

### Output File Format

Plain text file (UTF-8 encoding) containing the transcribed text.

**Example output** (Russian audio):
```
Привет, это тестовое сообщение на русском языке.
```

### Return Value (Python API)

The `main()` function returns a dictionary:

```python
{
    'text': str,              # Transcribed text
    'language': str,          # Detected language code (e.g., 'ru', 'en')
    'code_switching': bool    # True if multiple languages detected
}
```

## Logging

### Log File Location

```
~/agent-artifacts/<output_dir_name>/app.log
```

### Log Levels

- **File**: DEBUG (all messages)
- **Console**: INFO (progress messages only)

### Example Log Output

```
2025-02-03 14:22:15,123 - INFO - Starting transcription: input=/path/to/audio.ogg, output_dir=/path/to/output
2025-02-03 14:22:15,456 - INFO - Loading OGG audio file
2025-02-03 14:22:15,789 - INFO - Audio loaded: 160000 samples
2025-02-03 14:22:15,890 - INFO - Initializing Whisper model
2025-02-03 14:22:45,123 - INFO - Transcription complete: language=ru, text_length=245
2025-02-03 14:22:45,456 - INFO - Saving transcription to /path/to/output/transcription.txt
2025-02-03 14:22:45,789 - INFO - Transcription saved successfully
```

## Language Support

- **Auto-detection**: Whisper automatically detects language from audio
- **Russian (ru)**: Fully supported with Cyrillic character handling
- **English (en)**: Fully supported
- **99+ languages**: Supported by Whisper large model
- **Code-switching**: Detects and reports when multiple languages appear in same audio

## Model Details

- **Model**: OpenAI Whisper (large variant)
- **Size**: ~3 GB (downloaded to ~/.cache/huggingface on first run)
- **Device**: CPU (optimized for M1 Max)
- **Compute type**: default (float32)
- **First run**: Model download may take 2–5 minutes depending on network

## Troubleshooting

### FileNotFoundError: Audio file not found
- Verify the input file path is correct and file exists
- Use absolute paths if relative paths fail

### RuntimeError: Failed to load OGG file
- Ensure file is valid OGG Vorbis format (not MP3, WAV, etc.)
- Check file is not corrupted: `file <filename>` should show "Ogg Vorbis audio"

### IOError: Failed to write text to ...
- Verify output directory exists or is writable
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
