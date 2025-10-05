"""
Real-time Voice Conversion Constants for 16kHz Audio Processing

This module defines all timing and buffer constants for real-time streaming
voice conversion, following the architecture outlined in the ChatGPT guide.
"""

# Audio format constants
SAMPLE_RATE = 16000  # 16 kHz audio
CHANNELS = 1         # Mono audio

# Timing constants (all in samples at 16kHz)
HOP_20MS = 320       # 20ms hop = 320 samples at 16kHz (your system's actual hop_size)
CHUNK_80MS = 1280    # 80ms chunk = 1280 samples at 16kHz  
LKA_40MS = 640       # 40ms look-ahead = 640 samples at 16kHz
XFADE_2MS = 32       # 2ms cross-fade = 32 samples at 16kHz
XFADE_5MS = 80       # 5ms cross-fade = 80 samples at 16kHz (alternative)

# Frame-based constants (for mel-spectrogram processing)
# Using your system's hop_size=320 (20ms per frame)
HOP_SIZE = 320       # samples per frame (20ms at 16kHz)
FRAMES_PER_CHUNK = CHUNK_80MS // HOP_SIZE  # 4 frames per 80ms chunk
FRAMES_LOOKAHEAD = LKA_40MS // HOP_SIZE    # 2 frames of lookahead  
TOTAL_FRAMES_INPUT = FRAMES_PER_CHUNK + FRAMES_LOOKAHEAD  # 6 frames total input

# Buffer sizes
RING_BUFFER_SIZE = SAMPLE_RATE * 2  # 2 seconds of audio buffer
OUTPUT_BUFFER_SIZE = SAMPLE_RATE * 1  # 1 second of output buffer

# Processing constants
PRE_ROLL_MS = 120    # 80ms chunk + 40ms lookahead for stability
PRE_ROLL_SAMPLES = int(PRE_ROLL_MS * SAMPLE_RATE / 1000)

# Thread priorities (system dependent)
AUDIO_THREAD_PRIORITY = 95   # High priority for audio callback
INFERENCE_THREAD_PRIORITY = 85  # Lower but still high for inference

# Performance monitoring
MAX_INFERENCE_TIME_MS = 70   # Must complete inference in <70ms for 80ms chunks
XRUN_THRESHOLD_MS = 100      # Consider it an xrun if we exceed this

# Cross-fade windows
CROSSFADE_SAMPLES = XFADE_2MS  # Default to 2ms crossfade

def get_timing_info():
    """Return a dictionary with all timing information for debugging."""
    return {
        'sample_rate': SAMPLE_RATE,
        'hop_20ms_samples': HOP_20MS,
        'chunk_80ms_samples': CHUNK_80MS,
        'lookahead_40ms_samples': LKA_40MS,
        'crossfade_2ms_samples': XFADE_2MS,
        'frames_per_chunk': FRAMES_PER_CHUNK,
        'frames_lookahead': FRAMES_LOOKAHEAD,
        'total_input_frames': TOTAL_FRAMES_INPUT,
        'pre_roll_ms': PRE_ROLL_MS,
        'max_inference_time_ms': MAX_INFERENCE_TIME_MS,
    }

def validate_constants():
    """Validate that all constants are consistent."""
    assert CHUNK_80MS == 4 * HOP_20MS, "Chunk size should be 4 hops (4 * 20ms = 80ms)"
    assert LKA_40MS == 2 * HOP_20MS, "Look-ahead should be 2 hops (2 * 20ms = 40ms)"
    assert FRAMES_PER_CHUNK == CHUNK_80MS // HOP_SIZE, "Frame calculation mismatch"
    assert FRAMES_LOOKAHEAD == LKA_40MS // HOP_SIZE, "Lookahead frame calculation mismatch"
    assert HOP_SIZE == HOP_20MS, "Hop size should match 20ms hop"
    print("✅ All constants validated successfully!")

if __name__ == "__main__":
    validate_constants()
    info = get_timing_info()
    print("Real-time Voice Conversion Constants:")
    for key, value in info.items():
        print(f"  {key}: {value}") 