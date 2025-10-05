"""
Real-time Voice Conversion Gradio Demo

This demo provides a real-time voice conversion interface using the new
RealtimeVoiceConversion class with proper streaming audio I/O.
"""

import gradio as gr
import numpy as np
import threading
import queue
from typing import Optional, Generator
import time
import tempfile
import os
from collections import deque
import torch

from utils.commons.hparams import hparams, set_hparams
from utils.audio.io import save_wav
from realtime_constants import *
from RealtimeConan import RealtimeVoiceConversion

# Global variables for streaming audio processing
input_queue = queue.Queue(maxsize=100)  # Input audio queue
output_queue = queue.Queue(maxsize=100)  # Output audio queue
processing_active = threading.Event()
stop_processing = threading.Event()
sampling_rate = SAMPLE_RATE  # 16000 Hz

# Global voice conversion engine and reference
voice_converter: Optional[RealtimeVoiceConversion] = None
reference_audio_path: Optional[str] = None
temp_dir = tempfile.mkdtemp()

# Audio buffer for accumulating chunks
audio_buffer = deque()
buffer_size = CHUNK_80MS  # Process in 80ms chunks

# Streaming state for real-time processing
current_mel_buffer = deque()
emformer_state = None
total_frames_processed = 0

# Audio-level streaming chunk management variables  
audio_chunk_position = 0  # Current position in the streaming audio
segment_samples = 0  # Samples per segment (will be set from hparams)
lookahead_samples = 0  # Lookahead samples (will be set from hparams)
audio_lookahead_cache = deque()  # Cache for lookahead samples from previous chunks

# Output audio buffer for smooth streaming
output_audio_buffer = deque()  # Buffer for processed audio ready for output
output_chunk_size = SAMPLE_RATE * 40 // 1000  # 40ms output chunks = 640 samples at 16kHz

# New streaming chunk management variables
lookahead_cache = deque()  # Cache for lookahead frames from previous chunks
chunk_position = 0  # Current position in the streaming audio
segment_frames = 0  # Frames per segment (will be set from hparams)
lookahead_frames = 0  # Lookahead frames (will be set from hparams)

def initialize_voice_converter() -> str:
    """Initialize the RealtimeVoiceConversion engine."""
    global voice_converter
    try:
        set_hparams()
        voice_converter = RealtimeVoiceConversion(hparams)
        return "✅ Model initialized successfully!"
    except Exception as e:
        return f"❌ Failed to initialize model: {str(e)}"

def set_reference_voice(reference_audio) -> str:
    """Set the reference voice from uploaded audio."""
    global reference_audio_path, voice_converter
    
    if reference_audio is None:
        return "❌ No reference audio provided"
    
    if voice_converter is None:
        return "❌ Model not initialized"
    
    sample_rate, audio_data = reference_audio
    
    # Convert to 16kHz mono if needed
    if sample_rate != SAMPLE_RATE:
        import librosa
        audio_data = librosa.resample(audio_data.astype(np.float32), 
                                    orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    
    # Ensure mono
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Save reference audio
    reference_audio_path = os.path.join(temp_dir, "reference.wav")
    save_wav(audio_data, reference_audio_path, SAMPLE_RATE)
    
    # Set reference in the voice converter
    try:
        voice_converter.set_reference_audio(reference_audio_path)
        return f"✅ Reference voice set ({len(audio_data)/SAMPLE_RATE:.1f}s audio)"
    except Exception as e:
        return f"❌ Failed to set reference: {str(e)}"

def initialize_streaming_params():
    """Initialize streaming parameters from hparams."""
    global segment_frames, lookahead_frames, segment_samples, lookahead_samples
    if voice_converter is not None:
        # Convert from time-based to frame-based parameters
        # chunk_size is in samples, convert to mel frames (hop_size = 320 samples, 20ms per frame)
        segment_frames = voice_converter.chunk_size  # Already in frames
        lookahead_frames = voice_converter.right_context  # Already in frames
        
        # Convert to audio samples for audio-level processing
        segment_samples = segment_frames * voice_converter.hop_size  # frames * hop_size = samples
        lookahead_samples = lookahead_frames * voice_converter.hop_size  # frames * hop_size = samples
        
        print(f"Streaming params: segment={segment_frames} frames ({segment_samples} samples), lookahead={lookahead_frames} frames ({lookahead_samples} samples)")

def reset_streaming_chunk_state():
    """Reset all streaming chunk management state."""
    global current_mel_buffer, lookahead_cache, chunk_position, emformer_state, total_frames_processed
    global audio_chunk_position, audio_lookahead_cache, output_audio_buffer
    current_mel_buffer.clear()
    lookahead_cache.clear()
    chunk_position = 0
    emformer_state = None
    total_frames_processed = 0
    
    # Reset audio-level streaming state
    audio_chunk_position = 0
    audio_lookahead_cache.clear()
    
    # Reset output audio buffer
    output_audio_buffer.clear()

def prepare_audio_streaming_chunk() -> Optional[np.ndarray]:
    """
    Prepare an audio streaming chunk with proper segment and lookahead management at the raw audio level.
    
    Returns:
        np.ndarray: Prepared audio chunk [segment+lookahead samples] or None if not ready
    """
    global audio_buffer, audio_lookahead_cache, audio_chunk_position, segment_samples, lookahead_samples
    
    if segment_samples == 0 or lookahead_samples == 0:
        return None
    
    # For the first chunk, we need segment + lookahead samples
    if audio_chunk_position == 0:
        required_samples = segment_samples + lookahead_samples
        if len(audio_buffer) < required_samples:
            return None  # Not enough samples yet
        
        # Extract first chunk: segment + lookahead
        chunk_samples = []
        for _ in range(required_samples):
            if audio_buffer:
                chunk_samples.append(audio_buffer.popleft())
        
        # Cache the lookahead samples for next iteration
        audio_lookahead_cache.clear()
        audio_lookahead_cache.extend(chunk_samples[segment_samples:])  # Cache lookahead samples
        
        # Create chunk array
        chunk_audio = np.array(chunk_samples, dtype=np.float32)
        audio_chunk_position += segment_samples
        
        return chunk_audio
    
    else:
        # For subsequent chunks, we need segment samples (lookahead comes from cache + new samples)
        if len(audio_buffer) < segment_samples:
            return None  # Not enough new samples yet
        
        # Build chunk: cached lookahead + new segment samples + new lookahead
        chunk_samples = []
        
        # 1. Add cached lookahead samples from previous chunk
        chunk_samples.extend(list(audio_lookahead_cache))
        
        # 2. Add new segment samples
        new_segment_samples = []
        for _ in range(segment_samples):
            if audio_buffer:
                sample = audio_buffer.popleft()
                new_segment_samples.append(sample)
                chunk_samples.append(sample)
        
        # 3. Add new lookahead samples (if available)
        new_lookahead_samples = []
        lookahead_needed = lookahead_samples
        for i in range(min(lookahead_needed, len(audio_buffer))):
            if i < len(audio_buffer):
                sample = audio_buffer[i]  # Peek without removing
                new_lookahead_samples.append(sample)
                chunk_samples.append(sample)
        
        # 4. Pad if we don't have enough lookahead samples
        if len(new_lookahead_samples) < lookahead_needed:
            pad_needed = lookahead_needed - len(new_lookahead_samples)
            if len(new_segment_samples) > 0:
                last_sample = new_segment_samples[-1]
                for _ in range(pad_needed):
                    chunk_samples.append(last_sample)  # Repeat last sample
        
        # 5. Update lookahead cache for next iteration
        audio_lookahead_cache.clear()
        # Cache the lookahead portion (samples from current audio_buffer)
        for i in range(min(lookahead_samples, len(audio_buffer))):
            if i < len(audio_buffer):
                audio_lookahead_cache.append(audio_buffer[i])
        
        # If we don't have enough samples in buffer, pad the cache
        if len(audio_lookahead_cache) < lookahead_samples and len(new_segment_samples) > 0:
            last_sample = new_segment_samples[-1]
            while len(audio_lookahead_cache) < lookahead_samples:
                audio_lookahead_cache.append(last_sample)
        
        # Create chunk array
        expected_length = segment_samples + lookahead_samples
        if len(chunk_samples) != expected_length:
            # Ensure exact length
            if len(chunk_samples) > expected_length:
                chunk_samples = chunk_samples[:expected_length]
            else:
                # Pad to exact length
                while len(chunk_samples) < expected_length:
                    chunk_samples.append(chunk_samples[-1] if chunk_samples else 0.0)
        
        chunk_audio = np.array(chunk_samples, dtype=np.float32)
        audio_chunk_position += segment_samples
        
        return chunk_audio

def prepare_streaming_chunk() -> Optional[torch.Tensor]:
    """
    Prepare a streaming chunk with proper segment and lookahead management.
    
    Returns:
        torch.Tensor: Prepared chunk [1, segment+lookahead, 80] or None if not ready
    """
    global current_mel_buffer, lookahead_cache, chunk_position, segment_frames, lookahead_frames
    
    if segment_frames == 0 or lookahead_frames == 0:
        return None
    
    # For the first chunk, we need segment + lookahead frames
    if chunk_position == 0:
        required_frames = segment_frames + lookahead_frames
        if len(current_mel_buffer) < required_frames:
            return None  # Not enough frames yet
        
        # Extract first chunk: segment + lookahead
        chunk_frames = []
        for _ in range(required_frames):
            if current_mel_buffer:
                chunk_frames.append(current_mel_buffer.popleft())
        
        # Cache the lookahead frames for next iteration
        lookahead_cache.clear()
        lookahead_cache.extend(chunk_frames[segment_frames:])  # Cache lookahead frames
        
        # Create chunk tensor
        chunk_tensor = torch.stack(chunk_frames).unsqueeze(0)  # [1, segment+lookahead, 80]
        chunk_position += segment_frames
        
        return chunk_tensor
    
    else:
        # For subsequent chunks, we need segment frames (lookahead comes from cache + new frames)
        if len(current_mel_buffer) < segment_frames:
            return None  # Not enough new frames yet
        
        # Build chunk: cached lookahead + new segment frames + new lookahead
        chunk_frames = []
        
        # 1. Add cached lookahead frames from previous chunk
        chunk_frames.extend(list(lookahead_cache))
        
        # 2. Add new segment frames
        new_segment_frames = []
        for _ in range(segment_frames):
            if current_mel_buffer:
                frame = current_mel_buffer.popleft()
                new_segment_frames.append(frame)
                chunk_frames.append(frame)
        
        # 3. Add new lookahead frames (if available)
        new_lookahead_frames = []
        lookahead_needed = lookahead_frames
        for _ in range(min(lookahead_needed, len(current_mel_buffer))):
            if current_mel_buffer:
                frame = current_mel_buffer[_]  # Peek without removing
                new_lookahead_frames.append(frame)
                chunk_frames.append(frame)
        
        # 4. Pad if we don't have enough lookahead frames
        if len(new_lookahead_frames) < lookahead_needed:
            pad_needed = lookahead_needed - len(new_lookahead_frames)
            if len(chunk_frames) > 0:
                last_frame = chunk_frames[-1]
                for _ in range(pad_needed):
                    chunk_frames.append(last_frame)  # Repeat last frame
        
        # 5. Update lookahead cache for next iteration
        lookahead_cache.clear()
        # Cache the lookahead portion (segment frames from current_mel_buffer)
        cache_start = 0
        for i in range(min(lookahead_frames, len(current_mel_buffer))):
            if i < len(current_mel_buffer):
                lookahead_cache.append(current_mel_buffer[i])
        
        # If we don't have enough frames in buffer, pad the cache
        if len(lookahead_cache) < lookahead_frames and len(new_segment_frames) > 0:
            last_frame = new_segment_frames[-1]
            while len(lookahead_cache) < lookahead_frames:
                lookahead_cache.append(last_frame)
        
        # Create chunk tensor
        expected_length = segment_frames + lookahead_frames
        if len(chunk_frames) != expected_length:
            # Ensure exact length
            if len(chunk_frames) > expected_length:
                chunk_frames = chunk_frames[:expected_length]
            else:
                # Pad to exact length
                while len(chunk_frames) < expected_length:
                    chunk_frames.append(chunk_frames[-1] if chunk_frames else torch.zeros(80))
        
        chunk_tensor = torch.stack(chunk_frames).unsqueeze(0)  # [1, segment+lookahead, 80]
        chunk_position += segment_frames
        
        return chunk_tensor

def voice_conversion_algorithm(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Apply voice conversion using the RealtimeVoiceConversion model.
    This function now receives pre-prepared audio chunks with segment + lookahead from the worker.
    """
    global voice_converter, emformer_state, total_frames_processed, segment_samples
    
    if voice_converter is None or not voice_converter.is_reference_set:
        # Return original audio if not ready (only segment part, not lookahead)
        return audio_data[:segment_samples] if segment_samples > 0 else audio_data
    
    try:
        # Convert to 16kHz mono if needed
        if sample_rate != SAMPLE_RATE:
            import librosa
            audio_data = librosa.resample(audio_data.astype(np.float32), 
                                        orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        
        # Ensure mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Convert audio to mel spectrogram
        temp_src_path = os.path.join(temp_dir, f"src_chunk_{time.time()}.wav")
        save_wav(audio_data, temp_src_path, SAMPLE_RATE)
        
        # Get mel spectrogram
        src_mel_np = voice_converter._wav_to_mel(temp_src_path)
        src_mel = torch.from_numpy(src_mel_np).unsqueeze(0).to(voice_converter.device)
        
        # Process the pre-prepared chunk directly with the RealtimeVoiceConversion system
        try:
            wav_chunk, emformer_state = voice_converter.infer_chunk(
                src_mel, 
                emformer_state=emformer_state
            )
            
            total_frames_processed += segment_frames
            
            # Clean up temporary file
            os.remove(temp_src_path)
            
            # Return only the segment portion of the processed audio (not lookahead)
            if len(wav_chunk) > 0:
                # Calculate how many samples correspond to the segment frames
                segment_audio_length = segment_samples if segment_samples > 0 else len(wav_chunk)
                return wav_chunk[:segment_audio_length].astype(np.float32)
            else:
                return audio_data[:segment_samples] if segment_samples > 0 else audio_data
                
        except Exception as e:
            print(f"Error in chunk processing: {e}")
            # Clean up temporary file on error
            if os.path.exists(temp_src_path):
                os.remove(temp_src_path)
            return audio_data[:segment_samples] if segment_samples > 0 else audio_data
        
    except Exception as e:
        print(f"Voice conversion error: {e}")
        # Return original audio on error (only segment part)
        return audio_data[:segment_samples] if segment_samples > 0 else audio_data

def audio_input_callback(audio_data):
    """Process incoming audio stream."""
    # Handle gradio version differences
    if isinstance(audio_data, (list, tuple)) and len(audio_data) == 2 and isinstance(audio_data[1], np.ndarray):
        sr, arr = audio_data
        print(f"Audio input callback: {sr}, {arr}")
    else:
        arr = np.asarray(audio_data, dtype=np.float32)

    # Add to input queue (non-blocking)
    try:
        input_queue.put(arr, block=False)
    except queue.Full:
        pass  # Drop audio if queue is full

def audio_processing_worker():
    """
    Background worker thread that processes audio chunks continuously using segment + lookahead logic
    """
    global audio_buffer, output_audio_buffer
    
    while not stop_processing.is_set():
        try:
            # Get audio data from input queue
            audio_chunk = input_queue.get(timeout=0.1)
            
            # Add to buffer
            audio_buffer.extend(audio_chunk)
            
            # Process using streaming chunk logic (segment + lookahead)
            while True:
                # Try to prepare a streaming audio chunk
                chunk_data = prepare_audio_streaming_chunk()
                if chunk_data is None:
                    break  # Not enough samples for next chunk
                
                # Apply voice conversion to the prepared chunk
                if voice_converter is not None and voice_converter.is_reference_set:
                    try:
                        processed_audio = voice_conversion_algorithm(chunk_data, sampling_rate)
                        
                        # Put processed audio into output audio buffer (not queue)
                        output_audio_buffer.extend(processed_audio)
                            
                    except Exception as e:
                        print(f"Error in voice conversion: {e}")
                        # Put original audio as fallback (only the segment part, not lookahead)
                        segment_only = chunk_data[:segment_samples] if segment_samples > 0 else chunk_data
                        output_audio_buffer.extend(segment_only)
            
            input_queue.task_done()
            
        except queue.Empty:
            # No input audio, continue waiting
            continue
        except Exception as e:
            print(f"Error in audio processing worker: {e}")
            continue

# 1) Make start_streaming a generator function
def start_streaming():
    """Start real-time audio streaming and processing (STREAMING GENERATOR)"""
    global processing_active, stop_processing, audio_buffer, current_mel_buffer, emformer_state, total_frames_processed
    global output_audio_buffer, output_chunk_size
    
    if voice_converter is None:
        # Must yield both outputs each time: (status, audio)
        yield "❌ Model not initialized", None
        return
    
    if not voice_converter.is_reference_set:
        yield "❌ No reference voice set", None
        return
    
    # Initialize streaming parameters
    initialize_streaming_params()
    
    # Clear queues and reset state
    while not input_queue.empty():
        try:
            input_queue.get_nowait()
            input_queue.task_done()
        except queue.Empty:
            break
    
    while not output_queue.empty():
        try:
            output_queue.get_nowait()
        except queue.Empty:
            break
    
    audio_buffer.clear()
    
    # Reset streaming state in voice converter and chunk management
    voice_converter.reset_streaming_state()
    reset_streaming_chunk_state()
    
    # Start background processing
    processing_active.set()
    stop_processing.clear()
    processing_thread = threading.Thread(target=audio_processing_worker, daemon=True)
    processing_thread.start()
    
    # Prime UI with a tiny silence chunk so the player starts
    prime = np.zeros(int(sampling_rate * 0.1), dtype=np.float32)
    yield "✅ Real-time conversion started - speak into your microphone!", (sampling_rate, prime)
    
    # Stream out 40ms chunks until stopped
    while processing_active.is_set() or len(output_audio_buffer) > 0:
        try:
            # Check if we have enough audio for a 40ms chunk
            if len(output_audio_buffer) >= output_chunk_size:
                # Extract 40ms chunk from output buffer
                chunk_40ms = []
                for _ in range(output_chunk_size):
                    if output_audio_buffer:
                        chunk_40ms.append(output_audio_buffer.popleft())
                
                processed_audio = np.array(chunk_40ms, dtype=np.float32)
            else:
                # No audio available right now -> send a short silence to keep the stream alive
                processed_audio = np.zeros(int(sampling_rate * 0.04), dtype=np.float32)  # 40ms silence
            
            yield "✅ Converting…", (sampling_rate, processed_audio)
            
        except Exception as e:
            print(f"Error in output streaming: {e}")
            # Send silence on error
            silence = np.zeros(int(sampling_rate * 0.04), dtype=np.float32)
            yield "✅ Converting…", (sampling_rate, silence)
    
    # When loop ends, send one final update (optional)
    yield "⏹️ Real-time conversion stopped", None


# 2) Stop handler: stop threads and clear audio
def stop_streaming():
    """Stop real-time audio streaming and processing"""
    global processing_active, stop_processing, output_audio_buffer
    processing_active.clear()
    stop_processing.set()
    
    # Clear the output audio buffer
    output_audio_buffer.clear()
    
    # Clean up any remaining queue items
    while not output_queue.empty():
        try:
            output_queue.get_nowait()
            output_queue.task_done()
        except queue.Empty:
            break
    
    # Return status + clear audio output
    return "⏹️ Real-time conversion stopped", gr.update(value=None)

# Create Gradio Interface
with gr.Blocks(title="Real-Time Voice Conversion") as demo:
    gr.Markdown("# 🎙️ Real-Time Voice Conversion Demo")
    gr.Markdown("Convert your voice to sound like the reference speaker in real-time!")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## 🔧 Setup")
            
            # Model initialization
            init_btn = gr.Button("Initialize Model", variant="primary")
            init_status = gr.Textbox(label="Model Status", interactive=False)
            
            # Reference voice upload
            reference_audio = gr.Audio(
                label="Upload Reference Voice",
                type="numpy"
            )
            ref_status = gr.Textbox(label="Reference Status", interactive=False)
        
        with gr.Column():
            gr.Markdown("## 🎤 Real-time Audio Streaming")
            
            # Audio input component with streaming enabled
            audio_input = gr.Audio(
                sources=["microphone"],
                streaming=True,
                type="numpy",
                label="Microphone Input",
                interactive=True,
            )
            
            # Audio output component
            audio_output = gr.Audio(
                label="Converted Voice Output",
                streaming=True,
                autoplay=True
            )
            
            # Status display
            status_display = gr.Textbox(
                label="Conversion Status",
                interactive=False
            )
            
            # Start and stop buttons
            with gr.Row():
                start_btn = gr.Button("Start Conversion", variant="secondary")
                stop_btn = gr.Button("Stop Conversion", variant="stop")
    
    # Event handlers
    init_btn.click(
        fn=initialize_voice_converter,
        inputs=[],
        outputs=init_status
    )
    
    reference_audio.change(
        fn=set_reference_voice,
        inputs=reference_audio,
        outputs=ref_status
    )
    
    start_btn.click(
        fn=start_streaming,
        inputs=[],
        outputs=[status_display, audio_output]
    )
    
    stop_btn.click(
        fn=stop_streaming,
        inputs=[],
        outputs=[status_display, audio_output]
    )
    
    # Real-time stream processing
    audio_input.stream(
        fn=audio_input_callback,
        inputs=[audio_input],
        outputs=[]
    )

    
    gr.Markdown("""
    ## 📝 Instructions:
    1. **Initialize Model**: Click to load the voice conversion model
    2. **Upload Reference**: Upload a WAV file of the target voice (3-10 seconds recommended)  
    3. **Start Conversion**: Begin real-time processing
    4. **Speak**: Use your microphone - your voice will be converted in real-time!
    
    ## ⚠️ Notes:
    - Ensure your microphone permissions are enabled
    - For best results, speak clearly and avoid background noise
    - The system uses streaming inference for low-latency conversion
    """)

if __name__ == "__main__":
    # Launch demo
    demo.launch(server_name="0.0.0.0", server_port=7860)
