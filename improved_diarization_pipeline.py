"""
Enhanced Speech Diarization and Transcription Pipeline

This script provides an improved pipeline for transcribing and diarizing audio from
various sources (YouTube, local files) with intelligent chunking based on silence
detection and optimized speaker diarization.

Features:
- Silence-based audio segmentation for natural speech boundaries
- Configurable speaker count for improved diarization accuracy
- Robust error handling and logging
- Support for multiple audio sources
- SALT language code mapping for multilingual support
"""

import os
import torch
import torchaudio
import datetime
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import yt_dlp
from openai import OpenAI
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from pyannote.audio import Pipeline
from huggingface_hub import login

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diarization_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# -------------------------------------------------
# Configuration
# -------------------------------------------------
@dataclass
class PipelineConfig:
    """Configuration for the diarization pipeline"""
    
    # Whisper API settings
    whisper_api_key: str = "local"
    whisper_base_url: str = "http://135.181.63.183:9000/v1/"
    whisper_model: str = "crestai/whisper_salt-large-v3-ct2"
    
    # Hugging Face settings
    hf_token: str = "hf_xxxxxxxxxxxxx"  # Replace with your token
    
    # Chunking settings
    chunk_strategy: str = "silence"  # "silence" or "fixed"
    min_silence_len: int = 700  # Minimum silence length in ms
    silence_thresh: int = -40  # Silence threshold in dBFS
    max_chunk_length: int = 30000  # Maximum chunk length in ms (30 seconds)
    min_chunk_length: int = 5000  # Minimum chunk length in ms (5 seconds)
    
    # Diarization settings
    num_speakers: Optional[int] = None  # Set to specific number if known
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    
    # Output settings
    output_dir: str = "transcriptions"
    temp_dir: str = "temp_audio"
    
    # Language settings
    salt_lang_code: Optional[str] = None  # e.g., 'lug', 'eng', 'swa', or None for auto
    
    # Processing settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------
# Language Mapping
# -------------------------------------------------
SALT_TO_WHISPER_LANG = {
    'eng': 'en',   # English (Ugandan)
    'swa': 'sw',   # Swahili
    'ach': 'su',   # Acholi
    'lgg': 'jw',   # Lugbara
    'lug': 'ba',   # Luganda
    'nyn': 'ha',   # Runyankole
    'teo': 'ln',   # Ateso
    'xog': 'haw',  # Lusoga
    'ttj': 'tt',   # Rutooro
    'kin': 'as',   # Kinyarwanda
    'myx': 'mg',   # Lumasaba
}


# -------------------------------------------------
# Audio Processing Classes
# -------------------------------------------------
class AudioChunker:
    """Handles intelligent audio chunking based on silence detection"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
    def chunk_by_silence(self, audio: AudioSegment) -> List[Tuple[AudioSegment, float, float]]:
        """
        Split audio into chunks based on silence detection.
        
        Args:
            audio: AudioSegment to chunk
            
        Returns:
            List of tuples (chunk, start_time, end_time) in seconds
        """
        logger.info("Detecting speech segments based on silence...")
        
        # Detect non-silent chunks
        nonsilent_ranges = detect_nonsilent(
            audio,
            min_silence_len=self.config.min_silence_len,
            silence_thresh=self.config.silence_thresh,
            seek_step=1
        )
        
        if not nonsilent_ranges:
            logger.warning("No speech detected, using full audio")
            return [(audio, 0, len(audio) / 1000)]
        
        logger.info(f"Detected {len(nonsilent_ranges)} initial speech segments")
        
        # Merge segments that are too short or split those that are too long
        chunks = []
        current_chunk_start = None
        current_chunk_end = None
        
        for start_ms, end_ms in nonsilent_ranges:
            segment_length = end_ms - start_ms
            
            if current_chunk_start is None:
                # Start new chunk
                current_chunk_start = start_ms
                current_chunk_end = end_ms
            else:
                # Check if we should merge with current chunk
                potential_length = end_ms - current_chunk_start
                
                if potential_length <= self.config.max_chunk_length:
                    # Merge segments
                    current_chunk_end = end_ms
                else:
                    # Save current chunk and start new one
                    if current_chunk_end - current_chunk_start >= self.config.min_chunk_length:
                        chunk = audio[current_chunk_start:current_chunk_end]
                        chunks.append((
                            chunk,
                            current_chunk_start / 1000,
                            current_chunk_end / 1000
                        ))
                    current_chunk_start = start_ms
                    current_chunk_end = end_ms
        
        # Add final chunk
        if current_chunk_start is not None and current_chunk_end is not None:
            if current_chunk_end - current_chunk_start >= self.config.min_chunk_length:
                chunk = audio[current_chunk_start:current_chunk_end]
                chunks.append((
                    chunk,
                    current_chunk_start / 1000,
                    current_chunk_end / 1000
                ))
        
        logger.info(f"Created {len(chunks)} optimized chunks")
        return chunks
    
    def chunk_fixed(self, audio: AudioSegment) -> List[Tuple[AudioSegment, float, float]]:
        """
        Split audio into fixed-length chunks (fallback method).
        
        Args:
            audio: AudioSegment to chunk
            
        Returns:
            List of tuples (chunk, start_time, end_time) in seconds
        """
        chunks = []
        chunk_length_ms = self.config.max_chunk_length
        
        for i in range(0, len(audio), chunk_length_ms):
            chunk = audio[i:i + chunk_length_ms]
            start_time = i / 1000
            end_time = (i + len(chunk)) / 1000
            chunks.append((chunk, start_time, end_time))
        
        logger.info(f"Created {len(chunks)} fixed-length chunks")
        return chunks
    
    def chunk_audio(self, audio: AudioSegment) -> List[Tuple[AudioSegment, float, float]]:
        """
        Chunk audio using configured strategy.
        
        Args:
            audio: AudioSegment to chunk
            
        Returns:
            List of tuples (chunk, start_time, end_time) in seconds
        """
        if self.config.chunk_strategy == "silence":
            try:
                return self.chunk_by_silence(audio)
            except Exception as e:
                logger.error(f"Silence-based chunking failed: {e}. Falling back to fixed chunking.")
                return self.chunk_fixed(audio)
        else:
            return self.chunk_fixed(audio)


class AudioSourceHandler:
    """Handles downloading and loading audio from various sources"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.temp_dir = Path(config.temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
    
    def download_youtube(self, url: str) -> str:
        """
        Download audio from YouTube video.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Path to downloaded audio file
        """
        logger.info(f"Downloading audio from YouTube: {url}")
        
        output_path = self.temp_dir / "youtube_audio.mp3"
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': str(self.temp_dir / 'youtube_audio.%(ext)s'),
            'quiet': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            logger.info("YouTube audio downloaded successfully")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to download YouTube audio: {e}")
            raise
    
    def load_audio(self, path: str) -> AudioSegment:
        """
        Load audio file.
        
        Args:
            path: Path to audio file
            
        Returns:
            AudioSegment object
        """
        logger.info(f"Loading audio file: {path}")
        
        try:
            # Try to determine format from extension
            file_ext = Path(path).suffix.lower().lstrip('.')
            if file_ext in ['mp3', 'wav', 'flac', 'ogg', 'm4a']:
                audio = AudioSegment.from_file(path, format=file_ext)
            else:
                audio = AudioSegment.from_file(path)
            
            logger.info(f"Audio loaded: {len(audio)/1000:.2f}s duration, {audio.frame_rate}Hz sample rate")
            return audio
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise


class TranscriptionEngine:
    """Handles transcription using Whisper API"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.whisper_api_key,
            base_url=config.whisper_base_url
        )
        self.whisper_lang = self._get_whisper_language()
    
    def _get_whisper_language(self) -> Optional[str]:
        """Get Whisper language code from SALT code"""
        if self.config.salt_lang_code:
            return SALT_TO_WHISPER_LANG.get(self.config.salt_lang_code)
        return None
    
    def transcribe_chunk(self, chunk_path: str) -> Dict[str, Any]:
        """
        Transcribe a single audio chunk.
        
        Args:
            chunk_path: Path to audio chunk
            
        Returns:
            Dictionary with transcription results
        """
        try:
            with open(chunk_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model=self.config.whisper_model,
                    file=audio_file,
                    language=self.whisper_lang,
                    response_format="verbose_json",
                    temperature=0.0
                )
                
                result = {
                    'text': transcript.text,
                    'segments': None
                }
                
                if hasattr(transcript, 'segments') and transcript.segments:
                    result['segments'] = transcript.segments
                
                return result
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {'text': '[ERROR]', 'segments': None}


class DiarizationEngine:
    """Handles speaker diarization"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize pipeline
        logger.info("Initializing diarization pipeline...")
        login(token=config.hf_token)
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=config.hf_token
        )
        self.pipeline.to(self.device)
    
    def diarize(self, audio_path: str) -> Any:
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Diarization results
        """
        logger.info("Performing speaker diarization...")
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.to(self.device)
        
        # Prepare diarization parameters
        diarization_params = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }
        
        # Add speaker count constraints if provided
        if self.config.num_speakers is not None:
            diarization_params["num_speakers"] = self.config.num_speakers
            logger.info(f"Using fixed speaker count: {self.config.num_speakers}")
        elif self.config.min_speakers is not None or self.config.max_speakers is not None:
            diarization_params["min_speakers"] = self.config.min_speakers
            diarization_params["max_speakers"] = self.config.max_speakers
            logger.info(f"Using speaker range: {self.config.min_speakers}-{self.config.max_speakers}")
        
        # Perform diarization
        diarization = self.pipeline(diarization_params)
        
        return diarization


class TranscriptAligner:
    """Aligns transcription segments with speaker diarization"""
    
    @staticmethod
    def align(segments: List[Tuple[float, float, str]], 
              diarization: Any) -> List[Tuple[str, float, float, str]]:
        """
        Align transcription segments with diarization results.
        
        Args:
            segments: List of (start, end, text) tuples
            diarization: Diarization results
            
        Returns:
            List of (speaker, start, end, text) tuples
        """
        logger.info("Aligning transcription with diarization...")
        
        aligned = []
        annotation = diarization.speaker_diarization if hasattr(diarization, 'speaker_diarization') else diarization
        
        for start, end, text in segments:
            # Find best overlapping speaker
            best_speaker, max_overlap = None, 0
            
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                overlap_start = max(start, turn.start)
                overlap_end = min(end, turn.end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = speaker
            
            if best_speaker is None:
                best_speaker = "UNKNOWN"
            
            aligned.append((best_speaker, start, end, text))
        
        # Merge consecutive same-speaker segments
        merged = []
        prev = None
        
        for curr in aligned:
            if prev and prev[0] == curr[0]:
                # Merge with previous segment
                prev = (prev[0], prev[1], curr[2], prev[3] + " " + curr[3])
            else:
                if prev:
                    merged.append(prev)
                prev = curr
        
        if prev:
            merged.append(prev)
        
        logger.info(f"Aligned {len(merged)} speaker segments")
        return merged


# -------------------------------------------------
# Main Pipeline
# -------------------------------------------------
class DiarizationPipeline:
    """Main pipeline orchestrating the entire process"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir = Path(config.temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.audio_handler = AudioSourceHandler(config)
        self.chunker = AudioChunker(config)
        self.transcriber = TranscriptionEngine(config)
        self.diarizer = DiarizationEngine(config)
        self.aligner = TranscriptAligner()
    
    def process(self, source: str, source_type: str = "file") -> Dict[str, Any]:
        """
        Process audio source through the complete pipeline.
        
        Args:
            source: Audio source (file path or YouTube URL)
            source_type: Type of source ("file" or "youtube")
            
        Returns:
            Dictionary containing all results
        """
        logger.info("="*70)
        logger.info("Starting Diarization Pipeline")
        logger.info("="*70)
        
        # Step 1: Load or download audio
        if source_type == "youtube":
            audio_path = self.audio_handler.download_youtube(source)
        else:
            audio_path = source
        
        audio = self.audio_handler.load_audio(audio_path)
        
        # Step 2: Perform diarization on full audio
        diarization = self.diarizer.diarize(audio_path)
        
        # Step 3: Chunk audio intelligently
        chunks = self.chunker.chunk_audio(audio)
        logger.info(f"Processing {len(chunks)} audio chunks")
        
        # Step 4: Transcribe each chunk
        chunk_transcripts = []
        all_segments = []
        
        for idx, (chunk, start_time, end_time) in enumerate(chunks):
            logger.info(f"Processing chunk {idx + 1}/{len(chunks)} ({start_time:.2f}s - {end_time:.2f}s)")
            
            # Save chunk temporarily
            chunk_path = self.temp_dir / f"chunk_{idx}.mp3"
            chunk.export(str(chunk_path), format="mp3")
            
            # Transcribe
            result = self.transcriber.transcribe_chunk(str(chunk_path))
            
            chunk_transcripts.append({
                'chunk_id': idx + 1,
                'start_time': start_time,
                'end_time': end_time,
                'text': result['text'],
                'segments': result['segments']
            })
            
            # Collect segments for alignment
            if result['segments']:
                for seg in result['segments']:
                    seg_start = start_time + (seg.start if seg.start is not None else 0)
                    seg_end = start_time + (seg.end if seg.end is not None else (seg_start + 5))
                    seg_text = (seg.text or '').strip()
                    if seg_text:
                        all_segments.append((seg_start, seg_end, seg_text))
            elif result['text'] and result['text'] != '[ERROR]':
                all_segments.append((start_time, end_time, result['text'].strip()))
            
            # Clean up
            chunk_path.unlink()
        
        # Step 5: Align transcription with diarization
        aligned_segments = self.aligner.align(all_segments, diarization)
        
        # Step 6: Generate outputs
        results = {
            'chunk_transcripts': chunk_transcripts,
            'aligned_segments': aligned_segments,
            'full_text': ' '.join([text for _, _, text in all_segments])
        }
        
        # Step 7: Save results
        self._save_results(results, source)
        
        # Cleanup
        if source_type == "youtube":
            Path(audio_path).unlink(missing_ok=True)
        
        logger.info("="*70)
        logger.info(f"Pipeline complete: {len(chunk_transcripts)} chunks, {len(aligned_segments)} speaker segments")
        logger.info("="*70)
        
        return results
    
    def _save_results(self, results: Dict[str, Any], source: str):
        """Save results to files"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"transcript_{timestamp}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("="*70 + "\n")
            f.write("SPEECH TRANSCRIPTION AND DIARIZATION RESULTS\n")
            f.write("="*70 + "\n")
            f.write(f"Source: {source}\n")
            f.write(f"Processed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Language: {self.config.salt_lang_code or 'Auto-detected'}\n")
            if self.config.num_speakers:
                f.write(f"Speakers: {self.config.num_speakers}\n")
            f.write("="*70 + "\n\n")
            
            # Chunk-by-chunk transcripts
            f.write("CHUNK-BY-CHUNK TRANSCRIPTS\n")
            f.write("="*70 + "\n\n")
            for chunk in results['chunk_transcripts']:
                f.write(f"[Chunk {chunk['chunk_id']}] {chunk['start_time']:.2f}s - {chunk['end_time']:.2f}s\n")
                f.write("-" * 70 + "\n")
                if chunk['segments']:
                    for seg in chunk['segments']:
                        seg_start = seg.start if seg.start is not None else 0
                        seg_end = seg.end if seg.end is not None else 0
                        seg_text = seg.text or ''
                        f.write(f"[{seg_start:.2f}s - {seg_end:.2f}s] {seg_text}\n")
                else:
                    f.write(chunk['text'] + "\n")
                f.write("\n")
            
            # Diarized transcript
            f.write("\n" + "="*70 + "\n")
            f.write("DIARIZED TRANSCRIPT (SPEAKER-ATTRIBUTED)\n")
            f.write("="*70 + "\n\n")
            for speaker, start, end, text in results['aligned_segments']:
                start_time = str(datetime.timedelta(seconds=int(start)))
                f.write(f"{speaker} [{start_time}]: {text}\n")
            
            # Full plain transcript
            f.write("\n" + "="*70 + "\n")
            f.write("FULL PLAIN TRANSCRIPT\n")
            f.write("="*70 + "\n\n")
            f.write(results['full_text'])
        
        logger.info(f"Results saved to: {output_file}")


# -------------------------------------------------
# Example Usage
# -------------------------------------------------
def main():
    """Example usage of the pipeline"""
    
    # Configure pipeline
    config = PipelineConfig(
        # Whisper settings
        whisper_base_url="http://135.181.63.183:9000/v1/",
        whisper_model="crestai/whisper_salt-large-v3-ct2",
        
        # HuggingFace token
        hf_token="hf_xxxxxxxxxxxxx",  # Replace with your token
        
        # Chunking strategy
        chunk_strategy="silence",  # Use "silence" for intelligent chunking
        min_silence_len=700,  # Minimum silence duration (ms)
        silence_thresh=-40,  # Silence threshold (dBFS)
        max_chunk_length=30000,  # Max chunk size (30s)
        min_chunk_length=5000,  # Min chunk size (5s)
        
        # Speaker settings - IMPORTANT: Set if you know the number of speakers
        num_speakers=2,  # Set to specific number if known (e.g., 2 for interview)
        # OR use min/max if uncertain:
        # min_speakers=2,
        # max_speakers=4,
        
        # Language
        salt_lang_code="swa",  # 'lug', 'eng', 'swa', etc., or None for auto-detect
        
        # Output
        output_dir="transcriptions",
        temp_dir="temp_audio"
    )
    
    # Initialize pipeline
    pipeline = DiarizationPipeline(config)
    
    # Process YouTube video
    youtube_url = "https://www.youtube.com/watch?v=c9bKcRO7enQ"
    results = pipeline.process(youtube_url, source_type="youtube")
    
    # Or process local file
    # results = pipeline.process("path/to/audio.wav", source_type="file")
    
    # Display summary
    print("\n" + "="*70)
    print("PROCESSING SUMMARY")
    print("="*70)
    print(f"Total chunks: {len(results['chunk_transcripts'])}")
    print(f"Speaker segments: {len(results['aligned_segments'])}")
    print(f"Unique speakers: {len(set(s[0] for s in results['aligned_segments']))}")
    print("\nDiarized Transcript Preview:")
    print("-"*70)
    for speaker, start, end, text in results['aligned_segments'][:5]:
        print(f"{speaker} [{start:.2f}s]: {text[:100]}...")


if __name__ == "__main__":
    main()
