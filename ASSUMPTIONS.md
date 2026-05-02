# ASSUMPTIONS.md

Auto-maintained by the agent platform. Each line below is one
assumption surfaced (or discovered) during the pipeline run.

Status legend:
- **RESOLVED** — the user / clarification answered the question.
- **OPEN** — flagged but not yet resolved; reviewer should surface.
- **DISCOVERED** — a worker made a non-obvious choice during execution.

## Entries

- [RESOLVED] Q: When you say 'highest quality,' do you mean transcription accuracy (fewest errors), or do you also care about preserving speaker tone, pauses, and punctuation in the output? — A: Fewest errors — _BA_ at 2026-05-02T16:54:58.648827Z
- [RESOLVED] Q: Are you willing to download a 1–2 GB model file once (takes 5–10 minutes), or do you need the program to work immediately with a smaller model? — A: I can download a model yes — _BA_ at 2026-05-02T16:54:58.648827Z
- [RESOLVED] Q: How long is acceptable for the program to run? If the audio is 1 hour long, is 10 minutes of processing time acceptable, or do you need it faster? — A: Audio is around 10 min long; any processing time speed is acceptable as long as quality is best. — _BA_ at 2026-05-02T16:54:58.648827Z
- [RESOLVED] Q: Where should the output text file be saved, and what should it be named? Should it go in the same Downloads folder, or somewhere else? — A: Let it go under ~/agent-artifacts/ then this project's folder name - there — _BA_ at 2026-05-02T16:54:58.648827Z
- [RESOLVED] Q: Is the audio file entirely in Russian, or does it mix Russian with other languages (English, code-switching, etc.)? — A: Sometimes could mix languages. — _BA_ at 2026-05-02T16:54:58.648827Z
- [RESOLVED] Q: Have you verified that the file '2026-04-20 at 21.20.37.ogg' is a valid, uncorrupted OGG audio file? Can you play it in a media player? — A: Yes, i can — _BA_ at 2026-05-02T16:54:58.648827Z
- [RESOLVED] Q: Do you have internet access available during the first run to download the model, or must the program work completely offline from the start? — A: Yes — _BA_ at 2026-05-02T16:54:58.648827Z
- [RESOLVED] Use Faster-Whisper large model for local Russian speech-to-text: Adopt Faster-Whisper with the large model variant, downloaded and cached locally on first run, with CTranslate2 backend for M1 native execution. — Large model maximizes accuracy per REQ-003 while Faster-Whisper's 4x speedup makes 10-minute audio processing acceptable on M1 Max. CTranslate2 provides native ARM64 support without cloud APIs. Russian language support is built-in with code-switching detection. — _SE_ at 2026-05-02T16:54:58.649438Z
- [RESOLVED] Implement single-file Python script with no external service calls: Create a single Python script that orchestrates OGG decoding, model inference, and text file output with no external API calls or network dependencies. — Simplicity and offline-first design align with REQ-004 and REQ-008. Single script eliminates deployment complexity, configuration management overhead, and network latency. All processing is deterministic and reproducible on M1 Max. — _SE_ at 2026-05-02T16:54:58.649438Z
- [RESOLVED] Use pathlib for file I/O with UTF-8 encoding and automatic directory creation: Use pathlib.Path for all file operations, pathlib.Path.mkdir(parents=True, exist_ok=True) for directory creation, and open with encoding='utf-8' for text output. — pathlib handles spaces and special characters automatically without shell escaping, mkdir with parents=True creates nested directories in one call, and UTF-8 encoding preserves Cyrillic characters without data loss. — _SE_ at 2026-05-02T16:54:58.649438Z
- [RESOLVED] Implement model caching strategy with ~/.cache/huggingface directory: Use Hugging Face transformers library default caching behavior (HF_HOME environment variable) to store model weights in ~/.cache/huggingface, with explicit logging of cache location and download progress. — Hugging Face caching is standard practice in ML ecosystems, automatically deduplicates model versions, and integrates seamlessly with Faster-Whisper. Explicit logging provides visibility into first-run download and subsequent cache hits. — _SE_ at 2026-05-02T16:54:58.649438Z
- [RESOLVED] Use Python logging module with file and console handlers for offline debugging: Configure Python logging with both file handler (logs to ~/agent-artifacts/[project-folder]/transcription.log) and console handler (INFO level), with timestamps and structured format. — Built-in logging module requires no external dependencies, provides structured output for debugging, and file logs persist for post-execution analysis. Console output gives real-time feedback during inference. — _SE_ at 2026-05-02T16:54:58.649438Z
