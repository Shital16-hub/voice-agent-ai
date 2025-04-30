"""
Enhanced Audio Preprocessor for Voice AI Agent.

This module provides advanced audio processing capabilities specifically
optimized for telephony applications, with robust speech/noise discrimination
and improved barge-in detection.
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
    Advanced audio preprocessing with telephony optimization and superior
    speech/noise discrimination for Voice AI Agent applications.
    
    Features:
    - Multi-stage noise filtering optimized for telephone audio
    - Robust speech activity detection with state machine
    - Adaptive noise floor tracking
    - Speech band energy analysis
    - Improved barge-in detection with cooldown periods
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        enable_barge_in: bool = True,
        barge_in_threshold: float = 0.055,  # Increased from 0.045
        min_speech_frames_for_barge_in: int = 12,  # Increased from 10
        barge_in_cooldown_ms: int = 2000,  # 2 second cooldown after agent speaks
        speech_timeout_ms: int = 5000,  # 5 second speech timeout
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
        
        # Enhanced adaptive noise tracking
        self.noise_samples = []
        self.max_samples = 50  # Increased for better statistics
        self.ambient_noise_level = 0.01555  # Increased from 0.01 for better noise rejection
        self.min_noise_floor = 0.008  # Increased from 0.005
        
        # Enhanced energy thresholds with hysteresis
        self.low_threshold = self.ambient_noise_level * 2.0  # For detecting potential speech
        self.high_threshold = self.ambient_noise_level * 4.5  # Increased from 3.5 for better noise rejection
        
        # Speech frequency band energy tracking
        self.speech_band_energies: Deque[float] = deque(maxlen=30)
        
        # Speech detection state machine
        self.speech_state = SpeechState.SILENCE
        self.potential_speech_frames = 0  # Count of consecutive potential speech frames
        self.confirmed_speech_frames = 0  # Count of consecutive confirmed speech frames
        self.silence_frames = 0  # Count of consecutive silence frames
        
        # Maintain enhanced audio buffer for improved detection
        self.recent_audio_buffer: Deque[np.ndarray] = deque(maxlen=20)  # Buffer for analysis
        
        # Barge-in state tracking
        self.agent_speaking = False
        self.agent_speaking_start_time = 0.0
        self.barge_in_detected = False
        self.last_barge_in_time = 0.0
        
        # Enable VAD warmup period - don't make decisions until we have enough data
        self.vad_warmup_frames = 15
        self.frame_count = 0
        
        # Spectral history for better noise estimation
        self.spectral_history: List[np.ndarray] = []
        self.max_spectral_history = 10
        
        # Time-domain filters states - no longer used with simplified approach
        self.hp_filter_state = None  # For high-pass filter
        self.bp_filter_state = None  # For band-pass filter
        
        # Log initialization
        logger.info(f"AudioPreprocessor initialized with barge_in_threshold={barge_in_threshold}, "
                    f"min_frames={min_speech_frames_for_barge_in}, cooldown={barge_in_cooldown_ms}ms")
    
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
                # Use 90th percentile to avoid outliers
                new_noise_level = max(
                    self.min_noise_floor,  # Minimum threshold
                    np.percentile(self.noise_samples, 90) * 2.0  # Set threshold just above noise
                )
                
                # Use exponential moving average to avoid abrupt changes
                alpha = 0.3  # Weight for new value (0.3 = 30% new, 70% old)
                self.ambient_noise_level = (alpha * new_noise_level + 
                                           (1 - alpha) * self.ambient_noise_level)
                
                if self.enable_debug:
                    logger.debug(f"Updated ambient noise level to {self.ambient_noise_level:.6f}")
                
                # Update derived thresholds
                self.low_threshold = self.ambient_noise_level * 2.0
                self.high_threshold = self.ambient_noise_level * 4.5  # Increased from 3.5
    
    def _calculate_speech_band_energy(self, audio_data: np.ndarray) -> Dict[str, float]:
        """
        Calculate energy in specific frequency bands relevant to speech.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Dictionary with energy in different frequency bands
        """
        if len(audio_data) < 256:  # Need enough samples for FFT
            return {
                "low_band": 0.0,
                "speech_band": 0.0,
                "high_band": 0.0,
                "speech_ratio": 0.0
            }
            
        try:
            # Calculate FFT
            fft_data = np.abs(np.fft.rfft(audio_data))
            freqs = np.fft.rfftfreq(len(audio_data), 1/self.sample_rate)
            
            # Define frequency bands - focus on telephony speech range
            low_band_idx = (freqs < 300)
            speech_band_idx = (freqs >= 300) & (freqs <= 3400)  # Primary speech frequencies
            high_band_idx = (freqs > 3400)
            
            # Calculate energy in each band
            low_energy = np.sum(fft_data[low_band_idx]**2)  # Using squared magnitude for energy
            speech_energy = np.sum(fft_data[speech_band_idx]**2)
            high_energy = np.sum(fft_data[high_band_idx]**2)
            
            total_energy = low_energy + speech_energy + high_energy + 1e-10
            
            # Calculate ratios
            speech_ratio = speech_energy / total_energy
            
            result = {
                "low_band": low_energy / total_energy,
                "speech_band": speech_ratio,
                "high_band": high_energy / total_energy,
                "speech_ratio": speech_ratio
            }
            
            # Store speech band energy for tracking
            self.speech_band_energies.append(speech_ratio)
            
            return result
        except Exception as e:
            logger.error(f"Error calculating frequency bands: {e}")
            return {
                "low_band": 0.0,
                "speech_band": 0.0,
                "high_band": 0.0,
                "speech_ratio": 0.0
            }
    
    def _apply_spectral_subtraction(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply spectral subtraction to reduce background noise.
        Simplified implementation that avoids shape mismatch issues.
        
        Args:
            audio_data: Audio data as numpy array
                
        Returns:
            Enhanced audio data
        """
        if len(audio_data) < 512:  # Need enough samples for a good FFT
            return audio_data
                
        try:
            # Compute STFT
            f, t, Zxx = signal.stft(audio_data, fs=self.sample_rate, nperseg=512, noverlap=384)
            
            # Compute magnitude spectrum
            magnitude = np.abs(Zxx)
            phase = np.angle(Zxx)
            
            # Simplified noise estimation that doesn't depend on history
            # Estimate noise as the minimum magnitude across time for each frequency
            # This avoids any shape mismatch issues completely
            noise_spectrum = np.min(magnitude, axis=1, keepdims=True) * 1.5
            
            # Create frequency weights for speech bands
            speech_weight = np.ones((len(f), 1))  # Shape is (freq, 1) for broadcasting
            speech_band = (f >= 300) & (f <= 3400)
            speech_weight[speech_band, 0] = 0.7  # Reduce subtraction in speech bands
            
            # Apply simple spectral subtraction with floor
            oversubtraction_factor = 2.0
            spectral_floor = 0.02
            
            # This broadcasted operation is safe because everything has compatible dimensions
            magnitude_enhanced = np.maximum(
                magnitude - oversubtraction_factor * speech_weight * noise_spectrum,
                spectral_floor * magnitude
            )
            
            # Reconstruct signal
            Zxx_enhanced = magnitude_enhanced * np.exp(1j * phase)
            _, audio_enhanced = signal.istft(Zxx_enhanced, fs=self.sample_rate, nperseg=512, noverlap=384)
            
            # Ensure same length as original
            if len(audio_enhanced) < len(audio_data):
                audio_enhanced = np.pad(audio_enhanced, (0, len(audio_data) - len(audio_enhanced)))
            elif len(audio_enhanced) > len(audio_data):
                audio_enhanced = audio_enhanced[:len(audio_data)]
            
            return audio_enhanced
                
        except Exception as e:
            logger.error(f"Error in spectral subtraction: {e}")
            return audio_data  # Return original if processing fails
    
    def _apply_time_domain_filters(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply time-domain filters with a simpler approach that avoids state issues.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Filtered audio data
        """
        try:
            # Apply high-pass filter without state preservation - higher cutoff for telephony
            b_hp, a_hp = signal.butter(4, 120/(self.sample_rate/2), 'highpass')  # Increased from 100Hz
            audio_hp = signal.filtfilt(b_hp, a_hp, audio_data)
            
            # Apply band-pass filter without state preservation - focused on speech frequencies
            b_bp, a_bp = signal.butter(4, [300/(self.sample_rate/2), 3400/(self.sample_rate/2)], 'band')
            audio_filtered = signal.filtfilt(b_bp, a_bp, audio_hp)
            
            return audio_filtered
            
        except Exception as e:
            logger.error(f"Error applying time-domain filters: {e}")
            return audio_data  # Return original if filtering fails
    
    def _apply_dynamic_compression(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply dynamic range compression to make speech more consistent.
        Optimized for telephony applications.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Compressed audio data
        """
        try:
            # Parameters for compression - optimized for telephony
            threshold = 0.2      # Lower threshold for speech
            ratio = 0.3          # More aggressive 3:1 compression ratio
            attack = 0.002       # Fast 2ms attack time
            release = 0.05       # 50ms release time
            makeup_gain = 2.0    # Higher makeup gain
            
            # Convert attack/release to samples
            attack_samples = int(attack * self.sample_rate)
            release_samples = int(release * self.sample_rate)
            
            # Initialize gain tracker
            envelope = np.zeros_like(audio_data)
            gain = np.ones_like(audio_data)
            
            # Simple envelope follower
            for i in range(1, len(audio_data)):
                # Current sample absolute value
                current_level = abs(audio_data[i])
                
                # Attack/release envelope detection
                if current_level > envelope[i-1]:
                    # Attack phase - fast rise
                    envelope[i] = envelope[i-1] + (current_level - envelope[i-1]) / attack_samples
                else:
                    # Release phase - slow fall
                    envelope[i] = envelope[i-1] + (current_level - envelope[i-1]) / release_samples
            
            # Apply compression curve
            for i in range(len(audio_data)):
                if envelope[i] > threshold:
                    # Above threshold, apply compression
                    above_threshold = envelope[i] - threshold
                    compressed = threshold + above_threshold * ratio
                    gain[i] = compressed / envelope[i] if envelope[i] > 0 else 1.0
                else:
                    # Below threshold, no compression
                    gain[i] = 1.0
            
            # Apply gain smoothing (additional pass)
            smoothed_gain = np.zeros_like(gain)
            smoothed_gain[0] = gain[0]
            
            # Simple lowpass filter on gain to reduce artifacts
            alpha = 0.9  # Smoothing factor
            for i in range(1, len(gain)):
                smoothed_gain[i] = alpha * smoothed_gain[i-1] + (1-alpha) * gain[i]
            
            # Apply compressed gain with makeup gain
            compressed_audio = audio_data * smoothed_gain * makeup_gain
            
            # Apply hard limiting to prevent clipping
            compressed_audio = np.clip(compressed_audio, -0.98, 0.98)
            
            return compressed_audio
            
        except Exception as e:
            logger.error(f"Error applying compression: {e}")
            return audio_data  # Return original if compression fails
    
    def _apply_speech_enhancement(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply speech enhancement techniques to improve speech clarity.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Enhanced audio data
        """
        try:
            # 1. Apply pre-emphasis to boost higher frequencies
            # This improves speech clarity by emphasizing consonants
            pre_emphasis = 0.97
            emphasized_audio = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])
            
            # 2. Apply speech onset enhancement - boost sudden changes in energy
            if len(emphasized_audio) > 320:  # At least 20ms
                # Calculate energy envelope
                frame_size = 160  # 10ms
                frame_count = (len(emphasized_audio) - frame_size) // frame_size
                energy_envelope = np.zeros(frame_count)
                
                for i in range(frame_count):
                    start = i * frame_size
                    end = start + frame_size
                    energy_envelope[i] = np.mean(np.square(emphasized_audio[start:end]))
                
                # Compute derivative of energy
                energy_derivative = np.diff(energy_envelope, prepend=energy_envelope[0])
                
                # Find regions of rising energy (speech onset)
                rising_energy = energy_derivative > 0
                
                # Expand to full audio length
                rising_energy_full = np.repeat(rising_energy, frame_size)
                if len(rising_energy_full) < len(emphasized_audio):
                    rising_energy_full = np.pad(
                        rising_energy_full,
                        (0, len(emphasized_audio) - len(rising_energy_full)),
                        'edge'
                    )
                
                # Apply gentle boost to regions of rising energy
                boost_factor = np.ones_like(emphasized_audio)
                boost_factor[rising_energy_full] = 1.3  # 30% boost for speech onsets (increased)
                emphasized_audio = emphasized_audio * boost_factor
            
            return emphasized_audio
            
        except Exception as e:
            logger.error(f"Error applying speech enhancement: {e}")
            return audio_data  # Return original if enhancement fails
    
    def process_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply comprehensive audio processing optimized for telephony applications.
        
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
            
            # Add to recent audio buffer for context
            self.recent_audio_buffer.append(audio_data.copy())
            
            # 0. Update noise floor from incoming audio 
            self._update_ambient_noise_level(audio_data)
            
            # 1. Apply time-domain filters with state preservation
            audio_filtered = self._apply_time_domain_filters(audio_data)
            
            # 2. Apply spectral subtraction for noise reduction
            if len(audio_filtered) >= 512:  # Need enough samples for STFT
                audio_filtered = self._apply_spectral_subtraction(audio_filtered)
            
            # 3. Apply adaptive noise gate with threshold based on ambient noise
            threshold = max(0.01, self.ambient_noise_level * 2.5)
            audio_filtered = np.where(np.abs(audio_filtered) < threshold, 0, audio_filtered)
            
            # 4. Apply speech enhancement techniques
            audio_enhanced = self._apply_speech_enhancement(audio_filtered)
            
            # 5. Apply dynamic compression for consistent levels
            audio_compressed = self._apply_dynamic_compression(audio_enhanced)
            
            # 6. Normalize audio level for consistency
            max_val = np.max(np.abs(audio_compressed))
            if max_val > 0:
                audio_normalized = audio_compressed * (0.95 / max_val)
            else:
                audio_normalized = audio_compressed
            
            return audio_normalized
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return audio_data  # Return original if processing fails
    
    def contains_speech(self, audio_data: np.ndarray) -> bool:
        """
        Enhanced speech detection with state machine for better noise discrimination.
        Specifically optimized for telephony applications.
        
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
            
            # Calculate zero-crossing rate (helps distinguish speech from noise)
            zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_data)))) / len(audio_data)
            
            # Calculate frequency band energies
            band_energies = self._calculate_speech_band_energy(audio_data)
            speech_ratio = band_energies["speech_ratio"]
            
            # Calculate spectral flux - measure of how quickly spectrum changes
            if len(audio_data) >= 512:
                half_point = len(audio_data) // 2
                first_half = audio_data[:half_point]
                second_half = audio_data[half_point:2*half_point]
                
                # Calculate spectra
                first_spectrum = np.abs(np.fft.rfft(first_half))
                second_spectrum = np.abs(np.fft.rfft(second_half))
                
                # Calculate normalized spectral flux
                if np.sum(first_spectrum) > 0 and np.sum(second_spectrum) > 0:
                    # Normalize spectra
                    first_spectrum = first_spectrum / np.sum(first_spectrum)
                    second_spectrum = second_spectrum / np.sum(second_spectrum)
                    
                    # Calculate flux (sum of squared differences)
                    spectral_flux = np.sum(np.square(second_spectrum - first_spectrum))
                else:
                    spectral_flux = 0.0
            else:
                spectral_flux = 0.0
            
            # Calculate the average speech band ratio from history
            avg_speech_ratio = 0.4  # Default
            if self.speech_band_energies:
                avg_speech_ratio = np.mean(self.speech_band_energies)
            
            # Skip decisions during warmup period, default to false
            if self.frame_count < self.vad_warmup_frames:
                if self.enable_debug:
                    logger.debug(f"In warmup period, frame {self.frame_count}/{self.vad_warmup_frames}")
                return False
            
            # Log detailed VAD metrics for debugging
            if self.enable_debug:
                logger.debug(f"VAD metrics - State: {self.speech_state}, Energy: {rms_energy:.4f}, "
                           f"ZCR: {zero_crossings:.4f}, Speech ratio: {speech_ratio:.2f}, "
                           f"Flux: {spectral_flux:.4f}, Low threshold: {self.low_threshold:.4f}, "
                           f"High threshold: {self.high_threshold:.4f}")
            
            # State machine for more robust speech detection
            if self.speech_state == SpeechState.SILENCE:
                # Check if potential speech detected - more strict criteria
                if rms_energy > self.low_threshold and speech_ratio > 0.6:  # Increased from 0.5
                    self.speech_state = SpeechState.POTENTIAL_SPEECH
                    self.potential_speech_frames = 1
                    if self.enable_debug:
                        logger.debug(f"Potential speech detected: energy={rms_energy:.4f}, speech_ratio={speech_ratio:.2f}")
                    return False  # Not confirmed yet
                return False  # Still silence
                
            elif self.speech_state == SpeechState.POTENTIAL_SPEECH:
                # Check if still potential speech
                if rms_energy > self.low_threshold and speech_ratio > 0.6:  # Increased from 0.5
                    self.potential_speech_frames += 1
                    # Check if we have enough frames to move to confirmed speech - more frames required
                    if self.potential_speech_frames >= 4:  # Increased from 3 for better confidence
                        # Check against higher threshold for confirmation - stricter criteria
                        if rms_energy > self.high_threshold and speech_ratio > 0.65:  # Increased from 0.6
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
                # Check if still speech
                if rms_energy > self.high_threshold * 0.7 or speech_ratio > 0.5:
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
                # Check if silence is confirmed
                if rms_energy < self.high_threshold * 0.7 or speech_ratio < 0.4:
                    self.silence_frames += 1
                    # Check if we have enough frames to confirm end of speech
                    if self.silence_frames >= 6:  # Increased from 5 for more robust speech end detection
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
                    
            # Default fallback - use direct detection
            is_speech = (
                (rms_energy > self.high_threshold) and 
                (zero_crossings > 0.01 and zero_crossings < 0.15) and
                ((speech_ratio > 0.6 and speech_ratio > avg_speech_ratio * 0.8) or spectral_flux > 0.2)  # Increased threshold
            )
            
            if is_speech and self.enable_debug:
                logger.debug("Speech detected through direct conditions")
            
            return is_speech
            
        except Exception as e:
            logger.error(f"Error in speech detection: {e}")
            # Fall back to simple energy threshold
            energy = np.mean(np.abs(audio_data))
            return energy > max(0.025, self.ambient_noise_level * 4.0)  # Higher threshold
    
    def check_for_barge_in(self, audio_data: np.ndarray) -> bool:
        """
        Enhanced barge-in detection with state machine and cooldown period.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            True if barge-in is detected
        """
        if not self.enable_barge_in or not self.agent_speaking:
            return False
        
        # Ensure audio_data is float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            
        try:
            # Check cooldown period - don't detect barge-in immediately after agent starts speaking
            current_time = time.time()
            time_since_agent_started = (current_time - self.agent_speaking_start_time) * 1000  # ms
            if time_since_agent_started < self.barge_in_cooldown_ms:
                if self.enable_debug:
                    logger.debug(f"In barge-in cooldown period: {time_since_agent_started:.0f}/{self.barge_in_cooldown_ms}ms")
                return False
            
            # Check last barge-in time - don't trigger multiple barge-ins too quickly
            time_since_last_barge_in = (current_time - self.last_barge_in_time) * 1000  # ms
            if time_since_last_barge_in < 1000:  # Increased from 500ms - at least 1 second between barge-ins
                return False
            
            # Ensure we have speech using the state machine
            contains_speech = self.contains_speech(audio_data)
            
            # Only consider barge-in if we have confirmed speech for enough frames - higher requirement
            if (contains_speech and self.speech_state == SpeechState.CONFIRMED_SPEECH and 
                self.confirmed_speech_frames >= self.min_speech_frames_for_barge_in):
                
                # Calculate additional metrics for higher confidence
                rms_energy = np.sqrt(np.mean(np.square(audio_data)))
                
                # Check additional energy threshold specifically for barge-in - higher threshold
                if rms_energy > self.barge_in_threshold:
                    logger.info(f"Barge-in detected! Speech frames: {self.confirmed_speech_frames}, "
                               f"Energy: {rms_energy:.4f}, Threshold: {self.barge_in_threshold:.4f}")
                    
                    # Update barge-in state
                    self.barge_in_detected = True
                    self.last_barge_in_time = current_time
                    
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
                # Reset speech state to ensure we start fresh for barge-in detection
                self.speech_state = SpeechState.SILENCE
                self.potential_speech_frames = 0
                self.confirmed_speech_frames = 0
                self.silence_frames = 0
                logger.info("Agent speaking started, barge-in detection enabled after cooldown")
            else:
                # Agent stopped speaking
                logger.info("Agent speaking stopped")
    
    def reset(self) -> None:
        """Reset preprocessor state completely, avoiding any state persistence issues."""
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
        
        # Reset buffers and history collections
        self.spectral_history = []
        self.recent_audio_buffer.clear()
        self.speech_band_energies.clear()
        self.noise_samples = []
        
        # Reset threshold values to defaults
        self.ambient_noise_level = 0.01555  # Increased from 0.01
        self.low_threshold = self.ambient_noise_level * 2.0
        self.high_threshold = self.ambient_noise_level * 4.5  # Increased from 3.5
        
        # Reset frame counter
        self.frame_count = 0
        
        # Remove filter states completely - no longer used with new implementation
        self.hp_filter_state = None
        self.bp_filter_state = None
        
        logger.info("Audio preprocessor state reset")