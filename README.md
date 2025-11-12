# Enhanced Speech Diarization Pipeline

A professional-grade pipeline for transcribing and diarizing audio with intelligent chunking, multilingual support, and speaker identification.

## Key Improvements Over Original Script

### 1. **Silence-Based Chunking**
- Chunks audio at natural speech boundaries instead of arbitrary time intervals
- Preserves semantic coherence of speech segments
- Configurable silence detection parameters

### 2. **Speaker Count Optimization**
- Support for specifying exact number of speakers (significantly improves accuracy)
- Option to set speaker count ranges
- Automatic speaker detection when count is unknown

### 3. **Professional Architecture**
- Modular, class-based design
- Comprehensive error handling
- Detailed logging
- Clean separation of concerns

### 4. **Enhanced Robustness**
- Fallback mechanisms for failed operations
- Proper resource cleanup
- Type hints for better code maintainability
- Structured configuration management

## Installation

```bash
# Install required packages
pip install openai yt-dlp pydub pyannote.audio torch torchaudio huggingface_hub

# For audio processing
pip install ffmpeg-python

# Note: You may need to install ffmpeg separately on your system
# Ubuntu/Debian: sudo apt-get install ffmpeg
# macOS: brew install ffmpeg
```

## Quick Start

### Basic Usage

```python
from improved_diarization_pipeline import DiarizationPipeline, PipelineConfig

# Configure the pipeline
config = PipelineConfig(
    hf_token="your_huggingface_token_here",
    salt_lang_code="lug",  # Luganda
    num_speakers=2,  # Set if you know the number of speakers
)

# Initialize pipeline
pipeline = DiarizationPipeline(config)

# Process a YouTube video
results = pipeline.process(
    "https://www.youtube.com/watch?v=VIDEO_ID",
    source_type="youtube"
)

# Or process a local audio file
results = pipeline.process(
    "path/to/audio.wav",
    source_type="file"
)
```

## Configuration Guide

### Essential Parameters

#### 1. Hugging Face Token
Required for accessing the pyannote diarization model.

```python
config = PipelineConfig(
    hf_token="hf_xxxxxxxxxxxxx"  # Get from https://huggingface.co/settings/tokens
)
```

#### 2. Language Selection
Set the language for better transcription accuracy.

```python
# Specific language
config = PipelineConfig(
    salt_lang_code="lug"  # Options: lug, eng, swa, ach, lgg, nyn, teo, xog, ttj, kin, myx
)

# Auto-detection
config = PipelineConfig(
    salt_lang_code=None
)
```

#### 3. Speaker Count (IMPORTANT!)
Specifying the number of speakers significantly improves diarization accuracy.

```python
# Known exact number
config = PipelineConfig(
    num_speakers=2  # For interviews, dialogues
)

# Speaker range (when uncertain)
config = PipelineConfig(
    min_speakers=2,
    max_speakers=5
)

# Automatic detection (less accurate)
config = PipelineConfig()  # No speaker parameters
```

### Chunking Configuration

#### Silence-Based Chunking (Recommended)

```python
config = PipelineConfig(
    chunk_strategy="silence",
    min_silence_len=700,      # Minimum silence duration (ms)
    silence_thresh=-40,       # Silence threshold (dBFS)
    max_chunk_length=30000,   # Maximum chunk size (ms)
    min_chunk_length=5000,    # Minimum chunk size (ms)
)
```

**Parameter Tuning:**

| Speech Type | min_silence_len | silence_thresh | Use Case |
|-------------|----------------|----------------|----------|
| Fast conversation | 300-500ms | -35 to -40 | Casual chat, debate |
| Normal speech | 500-800ms | -40 to -45 | Interview, podcast |
| Formal speech | 800-1200ms | -45 to -50 | Presentation, lecture |
| Segmented content | 1200ms+ | -40 to -45 | Radio show, documentary |

#### Fixed-Length Chunking

```python
config = PipelineConfig(
    chunk_strategy="fixed",
    max_chunk_length=30000  # 30 seconds
)
```

Use fixed chunking when:
- Audio has consistent background noise
- Silence detection is unreliable
- You need predictable processing time

## Usage Examples

### Example 1: Luganda Interview (2 speakers)

```python
from improved_diarization_pipeline import DiarizationPipeline, PipelineConfig

config = PipelineConfig(
    hf_token="hf_xxxxxxxxxxxxx",
    salt_lang_code="lug",
    num_speakers=2,
    chunk_strategy="silence",
    min_silence_len=800,  # Natural pauses in conversation
    silence_thresh=-40,
    output_dir="luganda_interviews"
)

pipeline = DiarizationPipeline(config)
results = pipeline.process("interview.wav", source_type="file")
```

### Example 2: English News Broadcast (1 speaker)

```python
config = PipelineConfig(
    hf_token="hf_xxxxxxxxxxxxx",
    salt_lang_code="eng",
    num_speakers=1,
    chunk_strategy="silence",
    min_silence_len=500,  # Professional speech with shorter pauses
    silence_thresh=-35,
)

pipeline = DiarizationPipeline(config)
results = pipeline.process(
    "https://www.youtube.com/watch?v=NEWS_VIDEO",
    source_type="youtube"
)
```

### Example 3: Conference Call (Multiple speakers)

```python
config = PipelineConfig(
    hf_token="hf_xxxxxxxxxxxxx",
    salt_lang_code="eng",
    min_speakers=3,
    max_speakers=8,
    chunk_strategy="silence",
    min_silence_len=600,
    silence_thresh=-40,
)

pipeline = DiarizationPipeline(config)
results = pipeline.process("meeting.mp3", source_type="file")
```

### Example 4: Swahili Radio Show

```python
config = PipelineConfig(
    hf_token="hf_xxxxxxxxxxxxx",
    salt_lang_code="swa",
    min_speakers=2,
    max_speakers=6,
    chunk_strategy="silence",
    min_silence_len=1000,  # Longer silences for segment breaks
    silence_thresh=-45,
)

pipeline = DiarizationPipeline(config)
results = pipeline.process("radio_show.mp3", source_type="file")
```

### Example 5: Auto-detect Language

```python
config = PipelineConfig(
    hf_token="hf_xxxxxxxxxxxxx",
    salt_lang_code=None,  # Auto-detect
    num_speakers=2,
)

pipeline = DiarizationPipeline(config)
results = pipeline.process("unknown_language.wav", source_type="file")
```

## Output Format

The pipeline generates a structured text file with three sections:

### 1. Chunk-by-Chunk Transcripts
```
[Chunk 1] 0.00s - 15.43s
----------------------------------------------------------------------
[0.00s - 5.23s] Webale nnyo okutuyingiramu ku pulogulamu yaffe...
[5.23s - 12.89s] Leero tujja kwogera ku nsonga...
```

### 2. Diarized Transcript (Speaker-Attributed)
```
SPEAKER_00 [0:00:00]: Webale nnyo okutuyingiramu ku pulogulamu yaffe. Leero tujja kwogera ku nsonga...
SPEAKER_01 [0:00:15]: Weebale nnyo. Nze nina essanyu...
SPEAKER_00 [0:00:45]: Kale kino kiba kitya...
```

### 3. Full Plain Transcript
```
Webale nnyo okutuyingiramu ku pulogulamu yaffe. Leero tujja kwogera ku nsonga enkulu ezitukwatako. Weebale nnyo. Nze nina essanyu...
```

## Results Dictionary

```python
results = {
    'chunk_transcripts': [
        {
            'chunk_id': 1,
            'start_time': 0.0,
            'end_time': 15.43,
            'text': 'Transcribed text...',
            'segments': [...]  # Detailed segments if available
        },
        ...
    ],
    'aligned_segments': [
        ('SPEAKER_00', 0.0, 15.2, 'Speaker text...'),
        ('SPEAKER_01', 15.2, 28.5, 'Speaker text...'),
        ...
    ],
    'full_text': 'Complete transcript...'
}
```

## Supported Languages

SALT language codes mapped to Whisper:

| Language | SALT Code | Whisper Code |
|----------|-----------|--------------|
| English (Ugandan) | eng | en |
| Swahili | swa | sw |
| Acholi | ach | su |
| Lugbara | lgg | jw |
| Luganda | lug | ba |
| Runyankole | nyn | ha |
| Ateso | teo | ln |
| Lusoga | xog | haw |
| Rutooro | ttj | tt |
| Kinyarwanda | kin | as |
| Lumasaba | myx | mg |

## Troubleshooting

### Poor Diarization Results

**Problem:** Speakers are not correctly identified or frequently mixed up.

**Solutions:**
1. **Set the speaker count explicitly:**
   ```python
   config = PipelineConfig(num_speakers=2)  # If you know there are 2 speakers
   ```

2. **Adjust silence detection:**
   ```python
   # For faster speech with less silence
   config = PipelineConfig(min_silence_len=500, silence_thresh=-35)
   
   # For slower speech with more pauses
   config = PipelineConfig(min_silence_len=1000, silence_thresh=-45)
   ```

3. **Use fixed chunking for very noisy audio:**
   ```python
   config = PipelineConfig(chunk_strategy="fixed")
   ```

### Transcription Errors

**Problem:** Words are incorrectly transcribed.

**Solutions:**
1. **Specify the correct language:**
   ```python
   config = PipelineConfig(salt_lang_code="lug")
   ```

2. **Ensure audio quality is adequate:**
   - Sample rate: 16kHz or higher
   - Format: WAV, MP3, FLAC
   - Minimal background noise

### Memory Issues

**Problem:** Out of memory during processing.

**Solutions:**
1. **Reduce chunk size:**
   ```python
   config = PipelineConfig(max_chunk_length=20000)  # 20 seconds
   ```

2. **Use CPU instead of GPU:**
   ```python
   config = PipelineConfig(device="cpu")
   ```

### Chunking Issues

**Problem:** Audio is chunked at awkward points, cutting sentences.

**Solutions:**
1. **Adjust silence parameters:**
   ```python
   # More lenient silence detection
   config = PipelineConfig(
       min_silence_len=600,
       silence_thresh=-45
   )
   ```

2. **Adjust chunk size constraints:**
   ```python
   config = PipelineConfig(
       max_chunk_length=45000,  # Allow longer chunks
       min_chunk_length=3000     # Allow shorter chunks
   )
   ```

## Performance Tips

1. **GPU Acceleration:** Ensure CUDA is available for faster processing
   ```python
   import torch
   print(torch.cuda.is_available())  # Should return True
   ```

2. **Batch Processing:** Process multiple files efficiently
   ```python
   files = ["file1.wav", "file2.wav", "file3.wav"]
   for file in files:
       results = pipeline.process(file, source_type="file")
   ```

3. **Monitor Logs:** Check `diarization_pipeline.log` for detailed processing info

## Advanced Usage

### Custom Output Processing

```python
# Process and extract specific information
results = pipeline.process("audio.wav", source_type="file")

# Get speaker statistics
speakers = {}
for speaker, start, end, text in results['aligned_segments']:
    if speaker not in speakers:
        speakers[speaker] = {'duration': 0, 'turns': 0}
    speakers[speaker]['duration'] += (end - start)
    speakers[speaker]['turns'] += 1

print("Speaker Statistics:")
for speaker, stats in speakers.items():
    print(f"{speaker}: {stats['turns']} turns, {stats['duration']:.2f}s total")
```

### Integration with Other Tools

```python
# Export to JSON
import json

with open('results.json', 'w') as f:
    json.dump({
        'chunks': results['chunk_transcripts'],
        'speakers': results['aligned_segments'],
        'full_text': results['full_text']
    }, f, indent=2)
```

## Logging

The pipeline logs all operations to `diarization_pipeline.log`:

```
2025-11-12 10:30:15 - INFO - Starting Diarization Pipeline
2025-11-12 10:30:16 - INFO - Loading audio file: audio.wav
2025-11-12 10:30:17 - INFO - Detecting speech segments based on silence...
2025-11-12 10:30:18 - INFO - Created 8 optimized chunks
2025-11-12 10:30:20 - INFO - Processing chunk 1/8 (0.00s - 15.43s)
...
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- ffmpeg installed on system
- Hugging Face account with access to pyannote models

## License

This pipeline uses the following models:
- Whisper (OpenAI) - via faster-whisper-server
- pyannote.audio (MIT License)

Ensure you comply with the terms of use for each model.

## Credits

- pyannote.audio for speaker diarization
- OpenAI Whisper for speech recognition
- Sunbird AI for SALT language models

## Support

For issues specific to:
- **Whisper transcription:** Check your faster-whisper-server configuration
- **Diarization:** Ensure you have a valid Hugging Face token with pyannote access
- **Audio processing:** Verify ffmpeg installation

## Contributing

To extend this pipeline:
1. Inherit from base classes for custom behavior
2. Add new language mappings to `SALT_TO_WHISPER_LANG`
3. Implement custom chunking strategies in `AudioChunker`
4. Add post-processing in `TranscriptAligner`
