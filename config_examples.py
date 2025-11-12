"""
Configuration Template for Diarization Pipeline

Copy this file and customize the settings for your use case.
"""

from improved_diarization_pipeline import PipelineConfig

# ============================================================================
# BASIC CONFIGURATION
# ============================================================================

basic_config = PipelineConfig(
    # Replace with your actual Hugging Face token
    hf_token="hf_xxxxxxxxxxxxx",
    
    # Language: Set to specific SALT code or None for auto-detection
    # Options: 'lug', 'eng', 'swa', 'ach', 'lgg', 'nyn', 'teo', 'xog', 'ttj', 'kin', 'myx'
    salt_lang_code="lug",
    
    # Number of speakers (IMPORTANT for accuracy!)
    num_speakers=2,  # Set if you know the exact number
)

# ============================================================================
# ADVANCED CHUNKING CONFIGURATION
# ============================================================================

silence_based_config = PipelineConfig(
    hf_token="hf_xxxxxxxxxxxxx",
    
    # Chunking strategy
    chunk_strategy="silence",  # "silence" or "fixed"
    
    # Silence detection parameters
    min_silence_len=700,  # Minimum silence duration in ms (700ms = 0.7s)
                          # Increase for longer pauses, decrease for faster speech
    
    silence_thresh=-40,   # Silence threshold in dBFS
                          # More negative = quieter sounds treated as silence
                          # Typical range: -30 (strict) to -50 (lenient)
    
    # Chunk size constraints
    max_chunk_length=30000,  # Maximum 30 seconds per chunk
    min_chunk_length=5000,   # Minimum 5 seconds per chunk
)

# ============================================================================
# SPEAKER CONFIGURATION OPTIONS
# ============================================================================

# Option 1: Known exact number of speakers (BEST for accuracy)
known_speakers_config = PipelineConfig(
    hf_token="hf_xxxxxxxxxxxxx",
    num_speakers=3,  # e.g., for a 3-person panel discussion
)

# Option 2: Speaker range (when uncertain)
speaker_range_config = PipelineConfig(
    hf_token="hf_xxxxxxxxxxxxx",
    min_speakers=2,
    max_speakers=5,
)

# Option 3: Unknown speakers (least accurate)
unknown_speakers_config = PipelineConfig(
    hf_token="hf_xxxxxxxxxxxxx",
    # Don't set num_speakers, min_speakers, or max_speakers
)

# ============================================================================
# LANGUAGE-SPECIFIC CONFIGURATIONS
# ============================================================================

# Luganda podcast/interview
luganda_config = PipelineConfig(
    hf_token="hf_xxxxxxxxxxxxx",
    salt_lang_code="lug",
    num_speakers=2,
    chunk_strategy="silence",
    min_silence_len=800,  # Longer pauses for natural speech
    silence_thresh=-45,
)

# Multi-language detection (auto)
multilingual_config = PipelineConfig(
    hf_token="hf_xxxxxxxxxxxxx",
    salt_lang_code=None,  # Auto-detect language
    num_speakers=2,
)

# English (Ugandan) news broadcast
english_news_config = PipelineConfig(
    hf_token="hf_xxxxxxxxxxxxx",
    salt_lang_code="eng",
    num_speakers=1,  # Single anchor
    chunk_strategy="silence",
    min_silence_len=500,  # Shorter pauses for professional speech
    silence_thresh=-35,
)

# ============================================================================
# USE CASE SPECIFIC CONFIGURATIONS
# ============================================================================

# Conference call / Meeting (multiple speakers, cross-talk)
meeting_config = PipelineConfig(
    hf_token="hf_xxxxxxxxxxxxx",
    salt_lang_code="eng",
    min_speakers=3,
    max_speakers=10,
    chunk_strategy="silence",
    min_silence_len=600,
    silence_thresh=-40,
)

# Interview / Podcast (2 speakers, clear speech)
interview_config = PipelineConfig(
    hf_token="hf_xxxxxxxxxxxxx",
    salt_lang_code="lug",
    num_speakers=2,
    chunk_strategy="silence",
    min_silence_len=800,  # Longer pauses between turns
    silence_thresh=-40,
    max_chunk_length=45000,  # Longer chunks for sustained speech
)

# Lecture / Presentation (single speaker)
lecture_config = PipelineConfig(
    hf_token="hf_xxxxxxxxxxxxx",
    salt_lang_code="eng",
    num_speakers=1,
    chunk_strategy="fixed",  # Fixed chunking might work better
    max_chunk_length=30000,
)

# Radio show (multiple segments, music/ads)
radio_config = PipelineConfig(
    hf_token="hf_xxxxxxxxxxxxx",
    salt_lang_code="swa",
    min_speakers=2,
    max_speakers=6,
    chunk_strategy="silence",
    min_silence_len=1000,  # Longer silences to separate segments
    silence_thresh=-45,
)

# ============================================================================
# PERFORMANCE OPTIMIZATION
# ============================================================================

# Fast processing (lower accuracy)
fast_config = PipelineConfig(
    hf_token="hf_xxxxxxxxxxxxx",
    chunk_strategy="fixed",  # Faster than silence detection
    max_chunk_length=25000,
)

# High accuracy (slower processing)
accurate_config = PipelineConfig(
    hf_token="hf_xxxxxxxxxxxxx",
    chunk_strategy="silence",
    min_silence_len=600,
    silence_thresh=-42,
    max_chunk_length=20000,  # Smaller chunks for better alignment
    min_chunk_length=3000,
)

# ============================================================================
# AUDIO QUALITY HANDLING
# ============================================================================

# Poor audio quality (noisy, low quality recording)
poor_audio_config = PipelineConfig(
    hf_token="hf_xxxxxxxxxxxxx",
    chunk_strategy="fixed",  # Silence detection may fail with noise
    silence_thresh=-35,  # Stricter threshold
    num_speakers=2,  # Explicitly set if known
)

# High quality studio recording
studio_config = PipelineConfig(
    hf_token="hf_xxxxxxxxxxxxx",
    chunk_strategy="silence",
    min_silence_len=500,
    silence_thresh=-50,  # Can use lenient threshold with clean audio
)

# ============================================================================
# RECOMMENDED PARAMETER GUIDELINES
# ============================================================================

"""
SILENCE DETECTION TUNING:

min_silence_len (Minimum silence duration):
- 300-500ms: Fast speech, conversations with quick turns
- 500-800ms: Normal speech, interviews
- 800-1200ms: Formal speech, presentations
- 1200ms+: Separate distinct sections, radio shows

silence_thresh (Silence threshold in dBFS):
- -30 to -35: Very strict, only complete silence
- -35 to -45: Standard range, works for most audio
- -45 to -50: Lenient, includes quiet background sounds
- -50+: Very lenient, may include speech as silence

CHUNK SIZE:
- max_chunk_length: Keep at 20-30 seconds for API limits
- min_chunk_length: 3-5 seconds minimum for meaningful transcription

SPEAKER COUNT:
Priority:
1. Set num_speakers if you know exact count (BEST)
2. Set min/max range if uncertain
3. Leave unset for automatic detection (least accurate)

CHUNKING STRATEGY:
- "silence": Better for natural speech boundaries, more accurate alignment
- "fixed": Faster, more predictable, works with noisy audio
"""
