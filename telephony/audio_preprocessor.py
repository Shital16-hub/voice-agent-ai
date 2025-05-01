"""
Simplified Audio Preprocessor for Voice AI Agent.

This module provides streamlined audio processing for telephony applications
with a focus on essential processing for better speech recognition.
"""
import logging
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Union, Deque
from collections import deque
from scipy import signal

logger = logging.getLogger(__name__)

# Speech state enum for detection state machine
class SpeechState:
    """Speech state for detection state machine"""
    SILENCE = 0
    POTENTIAL_SPEECH = 1
    CONFIRMED_SPEECH = 2
    SPEECH_ENDED = 3

class AudioPreprocessor:
    """
    Streamlined audio preprocessing with telephony optimization for Voice AI Agent applications.
    
    Features:
    - Essential filtering optimized for telephony audio
    - Simple speech activity detection with fewer states
    - Fast barge-in detection with minimal processing overhead
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        enable_barge_in: bool = True,
        barge_in_threshold: float = 0.04,
        min_speech_frames_for_barge_in: int = 6,
        barge_in_cooldown_ms: int = 500,
        speech_timeout_ms: int = 5000,
        enable_debug: bool = False
    ):
        """
        Initialize the AudioPreprocessor.
        
        Args:
            sample_rate: Audio sample rate (default 16kHz)
            enable_barge_in: Whether to enable barge-in functionality
            barge_in_threshold: Energy threshold for barge-in detection
            min_speech_frames_for_barge_in: Minimum frames of speech for barge-in
            barge_in_cooldown_ms: Cooldown period after agent starts speaking
            speech_timeout_ms: Timeout for speech detection
            enable_debug: Enable detailed debug logging
        """
        self.sample_rate = sample_rate
        self.enable_barge_in = enable_barge_in
        self.barge_in_threshold = barge_in_threshold
        self.min_speech_frames_for_barge_in = min_speech_frames_for_barge_in
        self.barge_in_cooldown_ms = barge_in_cooldown_ms
        self.speech_timeout_ms = speech_timeout_ms
        self.enable_debug = enable_debug
        
        # Simplified noise tracking
        self.noise_samples = []
        self.max_samples = 20
        self.ambient_noise_level = 0.012
        self.min_noise_floor = 0.006
        
        # Energy thresholds with hysteresis
        self.low_threshold = self.ambient_noise_level * 2.0  # For detecting potential speech
        self.high_threshold = self.ambient_noise_level * 3.5
        
        # Speech frequency band energy tracking (reduced history)
        self.speech_band_energies: Deque[float] = deque(maxlen=5)
        
        # Simplified speech detection state
        self.speech_state = SpeechState.SILENCE
        self.potential_speech_frames = 0
        self.confirmed_speech_frames = 0
        self.silence_frames = 0
        
        # Maintain minimal audio buffer for analysis
        self.recent_audio_buffer: Deque[np.ndarray] = deque(maxlen=5)
        
        # Barge-in state tracking
        self.agent_speaking = False
        self.agent_speaking_start_time = 0.0
        self.barge_in_detected = False
        self.last_barge_in_time = 0.0
        
        # Reduced warmup period
        self.vad_warmup_frames = 8
        self.frame_count = 0
        
        # Log initialization
        logger.info(f"Initialized simplified AudioPreprocessor with barge-in: "
                    f"threshold={barge_in_threshold}, cooldown={barge_in_cooldown_ms}ms")
    
    def _update_ambient_noise_level(self, audio_data: np.ndarray) -> None:
        """
        Update ambient noise level using adaptive statistics.
        
        Args:
            audio_data: Audio data as numpy array
        """
        # Calculate energy of the audio
        energy = np.mean(np.abs(audio_data))
        
        # If audio is silence (very low energy), use it to update noise floor
        if energy < 0.02:  # Very quiet audio
            self.noise_samples.append(energy)
            # Keep only recent samples
            if len(self.noise_samples) > self.max_samples:
                self.noise_samples.pop(0)
            
            # Update ambient noise level (with safety floor)
            if self.noise_samples:
                # Use 80th percentile for better sensitivity
                new_noise_level = max(
                    self.min_noise_floor,  
                    np.percentile(self.noise_samples, 80) * 2.0
                )
                
                # Use exponential moving average
                alpha = 0.4  # Fast adaptation (40% weight to new value)
                self.ambient_noise_level = (alpha * new_noise_level + 
                                           (1 - alpha) * self.ambient_noise_level)
                
                # Update derived thresholds
                self.low_threshold = self.ambient_noise_level * 2.0
                self.high_threshold = self.ambient_noise_level * 3.5
    
    def _calculate_speech_band_energy(self, audio_data: np.ndarray) -> Dict[str, float]:
        """
        Calculate energy in the primary speech frequency band.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Dictionary with speech band energy ratio
        """
        if len(audio_data) < 256:  # Need enough samples for FFT
            return {
                "speech_ratio": 0.0
            }
            
        try:
            # Calculate FFT
            fft_data = np.abs(np.fft.rfft(audio_data))
            freqs = np.fft.rfftfreq(len(audio_data), 1/self.sample_rate)
            
            # Focus only on telephony speech range (300-3400 Hz)
            speech_band_idx = (freqs >= 300) & (freqs <= 3400)
            
            # Calculate speech energy vs total energy
            speech_energy = np.sum(fft_data[speech_band_idx]**2)
            total_energy = np.sum(fft_data**2) + 1e-10
            
            # Calculate ratio
            speech_ratio = speech_energy / total_energy
            
            # Store speech band energy for tracking
            self.speech_band_energies.append(speech_ratio)
            
            return {
                "speech_ratio": speech_ratio
            }
        except Exception as e:
            logger.error(f"Error calculating frequency bands: {e}")
            return {
                "speech_ratio": 0.0
            }
    
    def process_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply simplified audio processing optimized for telephony applications.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Processed audio data
        """
        try:
            # Ensure audio_data is float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                
            # Track frame count
            self.frame_count += 1
            
            # Add to recent audio buffer
            self.recent_audio_buffer.append(audio_data.copy())
            
            # 0. Update noise floor
            self._update_ambient_noise_level(audio_data)
            
            # 1. Apply high-pass filter to remove low-frequency noise
            b_hp, a_hp = signal.butter(4, 120/(self.sample_rate/2), 'highpass')
            audio_hp = signal.filtfilt(b_hp, a_hp, audio_data)
            
            # 2. Apply band-pass filter for telephony frequency range
            b_bp, a_bp = signal.butter(4, [300/(self.sample_rate/2), 3400/(self.sample_rate/2)], 'band')
            audio_filtered = signal.filtfilt(b_bp, a_bp, audio_hp)
            
            # 3. Apply noise gate
            threshold = max(0.01, self.ambient_noise_level * 2.5)
            audio_filtered = np.where(np.abs(audio_filtered) < threshold, 0, audio_filtered)
            
            # 4. Apply pre-emphasis
            pre_emphasis = 0.97
            audio_emphasized = np.append(audio_filtered[0], audio_filtered[1:] - pre_emphasis * audio_filtered[:-1])
            
            # 5. Normalize
            max_val = np.max(np.abs(audio_emphasized))
            if max_val > 0:
                audio_normalized = audio_emphasized * (0.95 / max_val)
            else:
                audio_normalized = audio_emphasized
            
            return audio_normalized
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return audio_data  # Return original if processing fails
    
    def contains_speech(self, audio_data: np.ndarray) -> bool:
        """
        Simplified speech detection with faster response.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            True if the audio contains speech
        """
        if len(audio_data) < 500:  # Need enough samples
            return False
        
        # Ensure audio_data is float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            
        try:
            # Calculate RMS energy
            rms_energy = np.sqrt(np.mean(np.square(audio_data)))
            
            # Calculate zero-crossing rate
            zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_data)))) / len(audio_data)
            
            # Calculate speech band energy
            band_energies = self._calculate_speech_band_energy(audio_data)
            speech_ratio = band_energies["speech_ratio"]
            
            # Skip decisions during warmup period
            if self.frame_count < self.vad_warmup_frames:
                return False
            
            # Simplified state machine for faster speech detection
            if self.speech_state == SpeechState.SILENCE:
                # Check if potential speech detected
                if rms_energy > self.low_threshold and speech_ratio > 0.5:
                    self.speech_state = SpeechState.POTENTIAL_SPEECH
                    self.potential_speech_frames = 1
                    return False  # Not confirmed yet
                return False  # Still silence
                
            elif self.speech_state == SpeechState.POTENTIAL_SPEECH:
                # Check if still potential speech
                if rms_energy > self.low_threshold and speech_ratio > 0.5:
                    self.potential_speech_frames += 1
                    # Confirm speech with fewer frames
                    if self.potential_speech_frames >= 2:
                        if rms_energy > self.high_threshold * 0.9 or speech_ratio > 0.6:
                            self.speech_state = SpeechState.CONFIRMED_SPEECH
                            self.confirmed_speech_frames = 1
                            logger.info(f"Speech confirmed: energy={rms_energy:.4f}, speech_ratio={speech_ratio:.2f}")
                            return True
                    return False  # Not confirmed yet
                else:
                    # Go back to silence
                    self.speech_state = SpeechState.SILENCE
                    self.potential_speech_frames = 0
                    return False
                    
            elif self.speech_state == SpeechState.CONFIRMED_SPEECH:
                # Still speech?
                if rms_energy > self.high_threshold * 0.7 or speech_ratio > 0.4:
                    # Still speech, maintain state
                    self.confirmed_speech_frames += 1
                    self.silence_frames = 0
                    return True
                else:
                    # Potential end of speech
                    self.speech_state = SpeechState.SPEECH_ENDED
                    self.silence_frames = 1
                    return True  # Still report as speech
                    
            elif self.speech_state == SpeechState.SPEECH_ENDED:
                # Faster confirmation of silence
                if rms_energy < self.high_threshold * 0.7 or speech_ratio < 0.4:
                    self.silence_frames += 1
                    # Fewer frames to confirm end
                    if self.silence_frames >= 3:
                        self.speech_state = SpeechState.SILENCE
                        self.silence_frames = 0
                        return False
                    return True  # Still report as speech until confirmed end
                else:
                    # Speech resumed
                    self.speech_state = SpeechState.CONFIRMED_SPEECH
                    self.confirmed_speech_frames += 1
                    self.silence_frames = 0
                    return True
                    
            # Simplified fallback
            return rms_energy > self.high_threshold and zero_crossings > 0.01 and zero_crossings < 0.15
            
        except Exception as e:
            logger.error(f"Error in speech detection: {e}")
            # Fall back to simple energy threshold
            energy = np.mean(np.abs(audio_data))
            return energy > max(0.025, self.ambient_noise_level * 4.0)
    
    def check_for_barge_in(self, audio_data: np.ndarray) -> bool:
        """
        Fast barge-in detection with minimal processing.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            True if barge-in is detected
        """
        if not self.enable_barge_in or not self.agent_speaking:
            return False
        
        try:
            # Check cooldown period
            current_time = time.time()
            time_since_agent_started = (current_time - self.agent_speaking_start_time) * 1000  # ms
            if time_since_agent_started < self.barge_in_cooldown_ms:
                return False
            
            # Check last barge-in time
            time_since_last_barge_in = (current_time - self.last_barge_in_time) * 1000  # ms
            if time_since_last_barge_in < 350:  # Prevent rapid repeated triggers
                return False
            
            # Simplified energy check for faster response
            rms_energy = np.sqrt(np.mean(np.square(audio_data)))
            
            # Simple approach: trigger on energy spike
            barge_in_detected = rms_energy > self.barge_in_threshold * 0.7
            
            # Additional speech ratio check if energy is borderline
            if not barge_in_detected and rms_energy > self.barge_in_threshold * 0.5:
                # Check speech band energy
                band_energies = self._calculate_speech_band_energy(audio_data)
                speech_ratio = band_energies["speech_ratio"]
                
                # Trigger if good speech quality
                if speech_ratio > 0.6:
                    barge_in_detected = True
            
            # Detect barge-in
            if barge_in_detected or (self.speech_state == SpeechState.CONFIRMED_SPEECH and 
                                    self.confirmed_speech_frames >= self.min_speech_frames_for_barge_in * 0.5):
                # Update barge-in state
                self.barge_in_detected = True
                self.last_barge_in_time = current_time
                
                logger.info(f"Barge-in detected! Energy: {rms_energy:.4f}, Threshold: {self.barge_in_threshold*0.7:.4f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking for barge-in: {e}")
            return False
    
    def set_agent_speaking(self, is_speaking: bool) -> None:
        """
        Set the agent speaking state.
        
        Args:
            is_speaking: Whether the agent is speaking
        """
        # Only update if state changes
        if is_speaking != self.agent_speaking:
            self.agent_speaking = is_speaking
            
            if is_speaking:
                # Agent started speaking
                self.agent_speaking_start_time = time.time()
                self.barge_in_detected = False
                # Minimal reset for cleaner barge-in detection
                self.speech_state = SpeechState.SILENCE
                logger.info("Agent speaking started, barge-in detection will be enabled after cooldown")
            else:
                # Agent stopped speaking
                logger.info("Agent speaking stopped")
    
    def reset(self) -> None:
        """Reset preprocessor state."""
        # Reset speech detection state
        self.speech_state = SpeechState.SILENCE
        self.potential_speech_frames = 0
        self.confirmed_speech_frames = 0
        self.silence_frames = 0
        
        # Reset barge-in tracking
        self.agent_speaking = False
        self.barge_in_detected = False
        self.agent_speaking_start_time = 0.0
        self.last_barge_in_time = 0.0
        
        # Reset buffers
        self.recent_audio_buffer.clear()
        self.speech_band_energies.clear()
        self.noise_samples = []
        
        # Reset threshold values to defaults
        self.ambient_noise_level = 0.012
        self.low_threshold = self.ambient_noise_level * 2.0
        self.high_threshold = self.ambient_noise_level * 3.5
        
        # Reset frame counter
        self.frame_count = 0
        
        logger.info("Audio preprocessor state reset")