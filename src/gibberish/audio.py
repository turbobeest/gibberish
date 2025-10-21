"""
Audio I/O operations for acoustic data transmission.

This module handles all audio input/output operations using ggwave for
encoding/decoding data into acoustic signals, including cable detection
via spectrogram analysis and bandwidth measurement.
"""

from typing import Optional, Tuple, Dict, List
import numpy as np
import time
import logging
from dataclasses import dataclass
from enum import Enum
import struct
import zlib

try:
    import ggwave
    GGWAVE_AVAILABLE = True
except ImportError:
    GGWAVE_AVAILABLE = False
    logging.warning("ggwave not available, using mock implementation")

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    logging.warning("sounddevice not available, audio I/O disabled")

from scipy import signal
from scipy.fft import fft, fftfreq


class TransmissionMode(Enum):
    """Audio transmission modes"""
    AUDIBLE = 1      # 1-3kHz for air transmission
    ULTRASONIC = 2   # 18-22kHz for cable transmission


class ConnectionType(Enum):
    """Connection type detection results"""
    AIR = "air"
    CABLE = "cable"
    UNKNOWN = "unknown"


@dataclass
class AudioConfig:
    """Configuration for audio transmission"""
    sample_rate: int = 48000
    protocol_id: int = 1
    mode: TransmissionMode = TransmissionMode.AUDIBLE
    # Bandwidth targets (bps)
    cable_bps: int = 350
    air_bps: int = 212
    # Detection thresholds
    cable_snr_threshold: float = 40.0  # dB
    # Frame settings
    max_payload_size: int = 240  # bytes per frame


class AudioManager:
    """Manages audio I/O operations for acoustic data transmission"""

    def __init__(self, config: Optional[AudioConfig] = None):
        """
        Initialize the AudioManager

        Args:
            config: Audio configuration (uses defaults if not provided)
        """
        self.config = config or AudioConfig()
        self.sample_rate = self.config.sample_rate
        self.protocol_id = self.config.protocol_id

        # Connection state
        self._is_cable_connected = False
        self._connection_type = ConnectionType.UNKNOWN
        self._last_snr = 0.0

        # Bandwidth monitoring
        self._bandwidth_samples: List[float] = []
        self._last_tx_time = 0.0
        self._last_tx_bytes = 0

        # Initialize ggwave instance if available
        self._ggwave_available = GGWAVE_AVAILABLE
        self._ggwave_instance = None
        if self._ggwave_available:
            try:
                self._ggwave_instance = ggwave.init()
                logging.info(f"ggwave instance initialized: {self._ggwave_instance}")
            except Exception as e:
                logging.warning(f"Failed to initialize ggwave: {e}")
                self._ggwave_available = False

        # Audio stream
        self._stream = None
        self._recording_buffer = []
        self._is_recording = False

        logging.info(f"AudioManager initialized: mode={self.config.mode}, "
                    f"sample_rate={self.sample_rate}, ggwave={'available' if self._ggwave_available else 'mock'}")

    def __del__(self):
        """Cleanup ggwave instance on destruction"""
        if hasattr(self, '_ggwave_instance') and self._ggwave_instance is not None:
            try:
                ggwave.free(self._ggwave_instance)
                logging.debug(f"Freed ggwave instance {self._ggwave_instance}")
            except Exception as e:
                logging.debug(f"Error freeing ggwave instance: {e}")

    def encode(self, data: bytes) -> np.ndarray:
        """
        Encode data into audio samples using ggwave

        Args:
            data: Raw bytes to encode

        Returns:
            Audio samples as numpy array
        """
        if self._ggwave_available:
            try:
                # Use ggwave to encode
                # Note: ggwave.encode() takes just the payload string, returns int16 samples as bytes
                payload_str = data.decode('utf-8') if isinstance(data, bytes) else data
                waveform_bytes = ggwave.encode(payload_str)
                # Convert int16 bytes to float32 numpy array normalized to [-1, 1]
                waveform_int16 = np.frombuffer(waveform_bytes, dtype=np.int16)
                waveform_float = waveform_int16.astype(np.float32) / 32768.0
                logging.info(f"ggwave encoded {len(data)} bytes -> {len(waveform_float)} audio samples")
                return waveform_float
            except Exception as e:
                logging.error(f"ggwave encoding failed: {e}")
                return self._mock_encode(data)
        else:
            return self._mock_encode(data)

    def _mock_encode(self, data: bytes) -> np.ndarray:
        """
        Mock encoder for testing when ggwave unavailable

        Creates a simple FSK-like signal
        """
        # Simple mock: encode each byte as a tone
        duration = 0.05  # 50ms per byte
        samples_per_byte = int(self.sample_rate * duration)
        result = []

        for byte_val in data:
            # Map byte to frequency (1000-3000 Hz for audible mode)
            if self.config.mode == TransmissionMode.AUDIBLE:
                freq = 1000 + (byte_val / 255.0) * 2000
            else:
                freq = 18000 + (byte_val / 255.0) * 4000

            t = np.linspace(0, duration, samples_per_byte, False)
            tone = np.sin(2 * np.pi * freq * t)
            result.extend(tone)

        return np.array(result, dtype=np.float32)

    def decode(self, audio_samples: np.ndarray) -> Optional[bytes]:
        """
        Decode audio samples into data using ggwave

        Args:
            audio_samples: Audio samples to decode

        Returns:
            Decoded bytes or None if decoding failed
        """
        if self._ggwave_available and self._ggwave_instance is not None:
            try:
                # Convert float32 samples to int16 bytes for ggwave
                waveform_int16 = (audio_samples * 32768.0).astype(np.int16)
                waveform_bytes = waveform_int16.tobytes()
                # ggwave.decode(instance_id, waveform_bytes) returns string like "b'DATA'" or None
                decoded_result = ggwave.decode(self._ggwave_instance, waveform_bytes)
                if decoded_result:
                    # Result is a string representation like "b'HELLO'"
                    # Strip the b' prefix and ' suffix, then encode to bytes
                    if decoded_result.startswith("b'") and decoded_result.endswith("'"):
                        decoded_str = decoded_result[2:-1]
                        return decoded_str.encode('latin-1')  # Use latin-1 to preserve byte values
                    return decoded_result.encode('utf-8')
                return None
            except Exception as e:
                logging.error(f"ggwave decoding failed: {e}")
                return None
        else:
            # Mock decode - just return None as we can't reliably decode mock signals
            logging.warning("Mock decode called - returning None")
            return None

    def encode_frame(self, payload: bytes, sequence_num: int = 0) -> bytes:
        """
        Encode a frame with header and CRC32 checksum

        Frame format:
        - Magic bytes (2 bytes): 0xAA 0x55
        - Version (1 byte): 0x01
        - Sequence number (2 bytes): uint16
        - Payload length (2 bytes): uint16
        - Payload (variable)
        - CRC32 (4 bytes): checksum of entire frame except CRC

        Args:
            payload: Data to encode in frame
            sequence_num: Sequence number for this frame

        Returns:
            Encoded frame as bytes
        """
        # Validate payload size
        if len(payload) > self.config.max_payload_size:
            raise ValueError(f"Payload size {len(payload)} exceeds maximum {self.config.max_payload_size}")

        # Build frame header
        magic = b'\xaa\x55'
        version = b'\x01'
        seq_bytes = struct.pack('>H', sequence_num & 0xFFFF)  # Big-endian uint16
        length_bytes = struct.pack('>H', len(payload))  # Big-endian uint16

        # Combine header and payload
        frame_without_crc = magic + version + seq_bytes + length_bytes + payload

        # Calculate CRC32
        crc = zlib.crc32(frame_without_crc) & 0xFFFFFFFF
        crc_bytes = struct.pack('>I', crc)  # Big-endian uint32

        # Complete frame
        frame = frame_without_crc + crc_bytes

        logging.debug(f"Encoded frame: seq={sequence_num}, len={len(payload)}, crc={crc:08x}")
        return frame

    def decode_frame(self, frame_data: bytes) -> Optional[Tuple[int, bytes]]:
        """
        Decode a frame and verify CRC32

        Args:
            frame_data: Raw frame bytes

        Returns:
            Tuple of (sequence_number, payload) or None if invalid
        """
        # Minimum frame size: 2 (magic) + 1 (version) + 2 (seq) + 2 (len) + 4 (crc) = 11 bytes
        if len(frame_data) < 11:
            logging.error(f"Frame too short: {len(frame_data)} bytes")
            return None

        # Check magic bytes
        if frame_data[0:2] != b'\xaa\x55':
            logging.error(f"Invalid magic bytes: {frame_data[0:2].hex()}")
            return None

        # Check version
        version = frame_data[2]
        if version != 0x01:
            logging.error(f"Unsupported version: {version}")
            return None

        # Extract header fields
        seq_num = struct.unpack('>H', frame_data[3:5])[0]
        payload_len = struct.unpack('>H', frame_data[5:7])[0]

        # Validate frame length
        expected_len = 7 + payload_len + 4  # header + payload + crc
        if len(frame_data) != expected_len:
            logging.error(f"Frame length mismatch: got {len(frame_data)}, expected {expected_len}")
            return None

        # Extract payload and CRC
        payload = frame_data[7:7+payload_len]
        received_crc = struct.unpack('>I', frame_data[7+payload_len:7+payload_len+4])[0]

        # Verify CRC
        frame_without_crc = frame_data[:7+payload_len]
        calculated_crc = zlib.crc32(frame_without_crc) & 0xFFFFFFFF

        if received_crc != calculated_crc:
            logging.error(f"CRC mismatch: received={received_crc:08x}, calculated={calculated_crc:08x}")
            return None

        logging.debug(f"Decoded frame: seq={seq_num}, len={payload_len}, crc={received_crc:08x}")
        return seq_num, payload

    def split_into_frames(self, data: bytes, max_payload_size: Optional[int] = None) -> List[bytes]:
        """
        Split data into multiple frames if necessary

        Args:
            data: Data to split
            max_payload_size: Maximum payload per frame (uses config default if None)

        Returns:
            List of encoded frames
        """
        max_size = max_payload_size or self.config.max_payload_size
        frames = []

        for i in range(0, len(data), max_size):
            chunk = data[i:i+max_size]
            seq_num = i // max_size
            frame = self.encode_frame(chunk, seq_num)
            frames.append(frame)

        logging.debug(f"Split {len(data)} bytes into {len(frames)} frames")
        return frames

    def reassemble_frames(self, frames: List[Tuple[int, bytes]]) -> bytes:
        """
        Reassemble data from multiple frames

        Args:
            frames: List of (sequence_number, payload) tuples

        Returns:
            Reassembled data
        """
        # Sort by sequence number
        sorted_frames = sorted(frames, key=lambda x: x[0])

        # Check for missing frames
        expected_seq = 0
        for seq_num, _ in sorted_frames:
            if seq_num != expected_seq:
                logging.warning(f"Missing frame: expected seq {expected_seq}, got {seq_num}")
            expected_seq = seq_num + 1

        # Concatenate payloads
        data = b''.join(payload for _, payload in sorted_frames)
        logging.debug(f"Reassembled {len(sorted_frames)} frames into {len(data)} bytes")
        return data

    def detect_cable(self, test_duration: float = 2.0) -> Tuple[bool, float]:
        """
        Detect if aux cable is connected using spectrogram analysis

        Performs a loopback test by transmitting a test signal and analyzing
        the received signal's SNR, frequency response, and amplitude stability.
        Cable connections typically show SNR >40dB, while air transmission
        shows much lower SNR due to ambient noise and distance.

        Args:
            test_duration: Duration of test signal in seconds

        Returns:
            Tuple of (is_connected, snr_db)
        """
        if not SOUNDDEVICE_AVAILABLE:
            logging.warning("sounddevice not available, cannot detect cable")
            return False, 0.0

        try:
            # Generate test signal: multi-frequency chirp
            test_signal = self._generate_test_signal(test_duration)

            # Record while playing (loopback test)
            logging.info("Running cable detection test...")

            # Start recording
            frames = int(test_duration * self.sample_rate * 1.5)  # Extra buffer
            recording = sd.playrec(test_signal.reshape(-1, 1),
                                   samplerate=self.sample_rate,
                                   channels=1,
                                   dtype='float32')
            sd.wait()

            # Analyze received signal
            received = recording.flatten()
            snr_db = self._calculate_snr(received, test_signal)

            # Additional cable indicators
            freq_response_quality = self._analyze_frequency_response(received)
            amplitude_stability = self._analyze_amplitude_stability(received)

            # Decision criteria
            is_cable = (
                snr_db > self.config.cable_snr_threshold and
                freq_response_quality > 0.7 and
                amplitude_stability > 0.8
            )

            self._last_snr = snr_db
            self._is_cable_connected = is_cable
            self._connection_type = ConnectionType.CABLE if is_cable else ConnectionType.AIR

            logging.info(f"Cable detection: {'CABLE' if is_cable else 'AIR'} "
                        f"(SNR={snr_db:.1f}dB, freq_quality={freq_response_quality:.2f}, "
                        f"amp_stability={amplitude_stability:.2f})")

            return is_cable, snr_db

        except Exception as e:
            logging.error(f"Cable detection failed: {e}")
            return False, 0.0

    def _generate_test_signal(self, duration: float) -> np.ndarray:
        """
        Generate a multi-frequency test signal for cable detection

        Uses a chirp signal that sweeps through the transmission frequency range

        Args:
            duration: Signal duration in seconds

        Returns:
            Test signal as numpy array
        """
        t = np.linspace(0, duration, int(duration * self.sample_rate), False)

        # Generate chirp based on transmission mode
        if self.config.mode == TransmissionMode.AUDIBLE:
            f0, f1 = 1000, 3000  # Audible range
        else:
            f0, f1 = 18000, 22000  # Ultrasonic range

        # Linear chirp
        chirp = signal.chirp(t, f0, duration, f1, method='linear')

        # Add some harmonics for better detection
        harmonics = 0.3 * np.sin(2 * np.pi * (f0 + f1) / 2 * t)

        test_signal = 0.7 * chirp + 0.3 * harmonics

        # Normalize
        test_signal = test_signal / np.max(np.abs(test_signal)) * 0.5

        return test_signal.astype(np.float32)

    def _calculate_snr(self, received: np.ndarray, reference: np.ndarray) -> float:
        """
        Calculate Signal-to-Noise Ratio in dB

        Args:
            received: Received signal
            reference: Reference (transmitted) signal

        Returns:
            SNR in dB
        """
        # Align signals (handle latency)
        if len(received) > len(reference):
            received = received[:len(reference)]
        elif len(received) < len(reference):
            reference = reference[:len(received)]

        # Cross-correlation to find alignment
        correlation = np.correlate(received, reference, mode='valid')
        if len(correlation) == 0:
            return 0.0

        # Find best alignment
        max_corr_idx = np.argmax(np.abs(correlation))

        # Extract aligned portion
        if max_corr_idx + len(reference) <= len(received):
            aligned_received = received[max_corr_idx:max_corr_idx + len(reference)]
        else:
            aligned_received = received

        # Calculate signal power (from reference)
        signal_power = np.mean(reference ** 2)

        # Calculate noise power (difference between received and reference)
        if len(aligned_received) == len(reference):
            noise = aligned_received - reference
        else:
            # Fallback: use high-frequency components as noise estimate
            noise = aligned_received - np.mean(aligned_received)

        noise_power = np.mean(noise ** 2)

        # Avoid division by zero
        if noise_power < 1e-10:
            return 60.0  # Very high SNR

        snr = 10 * np.log10(signal_power / noise_power)
        return float(snr)

    def _analyze_frequency_response(self, signal_data: np.ndarray) -> float:
        """
        Analyze frequency response quality

        Cable connections should preserve frequency content better than air

        Args:
            signal_data: Signal to analyze

        Returns:
            Quality metric (0-1, higher is better)
        """
        # Compute FFT
        fft_vals = fft(signal_data)
        freqs = fftfreq(len(signal_data), 1/self.sample_rate)

        # Get positive frequencies only
        positive_freqs = freqs[:len(freqs)//2]
        magnitude = np.abs(fft_vals[:len(fft_vals)//2])

        # Define expected frequency range
        if self.config.mode == TransmissionMode.AUDIBLE:
            f_min, f_max = 1000, 3000
        else:
            f_min, f_max = 18000, 22000

        # Find indices in expected range
        freq_mask = (positive_freqs >= f_min) & (positive_freqs <= f_max)

        if not np.any(freq_mask):
            return 0.0

        # Calculate power in expected range vs total
        signal_power = np.sum(magnitude[freq_mask] ** 2)
        total_power = np.sum(magnitude ** 2)

        if total_power < 1e-10:
            return 0.0

        quality = signal_power / total_power
        return float(np.clip(quality, 0, 1))

    def _analyze_amplitude_stability(self, signal_data: np.ndarray) -> float:
        """
        Analyze amplitude stability over time

        Cable connections should have more stable amplitude

        Args:
            signal_data: Signal to analyze

        Returns:
            Stability metric (0-1, higher is more stable)
        """
        # Divide into chunks
        chunk_size = int(0.1 * self.sample_rate)  # 100ms chunks
        n_chunks = len(signal_data) // chunk_size

        if n_chunks < 2:
            return 0.5

        chunk_rms = []
        for i in range(n_chunks):
            chunk = signal_data[i*chunk_size:(i+1)*chunk_size]
            rms = np.sqrt(np.mean(chunk ** 2))
            chunk_rms.append(rms)

        chunk_rms = np.array(chunk_rms)

        # Calculate coefficient of variation (lower is more stable)
        mean_rms = np.mean(chunk_rms)
        if mean_rms < 1e-10:
            return 0.0

        std_rms = np.std(chunk_rms)
        cv = std_rms / mean_rms

        # Convert to stability metric (inverse of CV, normalized)
        # CV of 0.1 or less indicates very stable (score 1.0)
        # CV of 0.5 or more indicates unstable (score 0.0)
        stability = 1.0 - np.clip(cv / 0.5, 0, 1)

        return float(stability)

    def measure_bandwidth(self) -> float:
        """
        Measure current transmission bandwidth

        Returns average bandwidth over recent transmissions

        Returns:
            Bandwidth in bytes per second
        """
        if not self._bandwidth_samples:
            return 0.0

        # Return average of recent samples
        return float(np.mean(self._bandwidth_samples))

    def get_bandwidth_stats(self) -> Dict[str, float]:
        """
        Get detailed bandwidth statistics

        Returns:
            Dictionary with bandwidth metrics:
            - mean: Average bandwidth (B/s)
            - median: Median bandwidth (B/s)
            - min: Minimum bandwidth (B/s)
            - max: Maximum bandwidth (B/s)
            - std: Standard deviation (B/s)
            - last: Last measured bandwidth (B/s)
        """
        if not self._bandwidth_samples:
            return {
                'mean': 0.0,
                'median': 0.0,
                'min': 0.0,
                'max': 0.0,
                'std': 0.0,
                'last': 0.0
            }

        samples = np.array(self._bandwidth_samples)

        return {
            'mean': float(np.mean(samples)),
            'median': float(np.median(samples)),
            'min': float(np.min(samples)),
            'max': float(np.max(samples)),
            'std': float(np.std(samples)),
            'last': float(samples[-1]) if len(samples) > 0 else 0.0
        }

    def get_connection_info(self) -> Dict[str, any]:
        """
        Get comprehensive connection information

        Returns:
            Dictionary with connection details:
            - type: ConnectionType enum (CABLE/AIR/UNKNOWN)
            - snr_db: Last measured SNR in dB
            - bandwidth: Current bandwidth stats
            - mode: TransmissionMode enum (AUDIBLE/ULTRASONIC)
            - sample_rate: Audio sample rate
        """
        return {
            'type': self._connection_type,
            'snr_db': self._last_snr,
            'bandwidth': self.get_bandwidth_stats(),
            'mode': self.config.mode,
            'sample_rate': self.sample_rate,
            'is_cable': self._is_cable_connected
        }

    def reset_bandwidth_stats(self):
        """Reset bandwidth statistics"""
        self._bandwidth_samples = []
        logging.debug("Bandwidth statistics reset")

    def estimate_transfer_time(self, data_size_bytes: int) -> float:
        """
        Estimate time to transfer given amount of data

        Args:
            data_size_bytes: Amount of data to transfer in bytes

        Returns:
            Estimated transfer time in seconds
        """
        current_bw = self.measure_bandwidth()

        if current_bw <= 0:
            # Use default estimates based on connection type
            if self._connection_type == ConnectionType.CABLE:
                current_bw = self.config.cable_bps
            elif self._connection_type == ConnectionType.AIR:
                current_bw = self.config.air_bps
            else:
                # Conservative estimate
                current_bw = self.config.air_bps

        # Add 20% overhead for framing and retransmissions
        overhead_factor = 1.2
        estimated_time = (data_size_bytes * overhead_factor) / current_bw

        return estimated_time

    def monitor_transmission_quality(self, sample_window: int = 10) -> Dict[str, float]:
        """
        Monitor real-time transmission quality metrics

        Args:
            sample_window: Number of recent samples to analyze

        Returns:
            Dictionary with quality metrics:
            - stability: Bandwidth stability (0-1, higher is better)
            - trend: Bandwidth trend (-1 to 1, positive means improving)
            - consistency: Consistency score (0-1, higher is better)
        """
        if len(self._bandwidth_samples) < 2:
            return {
                'stability': 0.0,
                'trend': 0.0,
                'consistency': 0.0
            }

        # Get recent samples
        recent = self._bandwidth_samples[-sample_window:]
        samples = np.array(recent)

        # Calculate stability (inverse of coefficient of variation)
        mean_bw = np.mean(samples)
        if mean_bw > 0:
            cv = np.std(samples) / mean_bw
            stability = max(0.0, 1.0 - cv)
        else:
            stability = 0.0

        # Calculate trend (slope of linear regression)
        if len(samples) >= 3:
            x = np.arange(len(samples))
            coeffs = np.polyfit(x, samples, 1)
            slope = coeffs[0]
            # Normalize trend to -1 to 1 range
            trend = np.clip(slope / (mean_bw / len(samples)), -1, 1)
        else:
            trend = 0.0

        # Calculate consistency (what % of samples are within 20% of mean)
        if len(samples) > 0:
            within_range = np.sum(np.abs(samples - mean_bw) < 0.2 * mean_bw)
            consistency = within_range / len(samples)
        else:
            consistency = 0.0

        return {
            'stability': float(stability),
            'trend': float(trend),
            'consistency': float(consistency)
        }

    def set_transmission_mode(self, mode: TransmissionMode):
        """
        Set transmission mode (audible or ultrasonic)

        Args:
            mode: TransmissionMode to use

        Note:
            When changing to ULTRASONIC mode, ensure the connection is cable-based
            as ultrasonic frequencies don't propagate well through air
        """
        old_mode = self.config.mode
        self.config.mode = mode

        # Update protocol ID for ggwave
        # ggwave protocol IDs:
        # 1: DT_NORMAL (audible, 1-3kHz)
        # 2: DT_FAST (audible, higher speed)
        # 3: DT_FASTEST (audible, highest speed)
        # 9: DT_ULTRASONIC (18-22kHz)
        # 10: DT_ULTRASONIC_FAST (18-22kHz, faster)
        # 11: DT_ULTRASONIC_FASTEST (18-22kHz, fastest)

        if mode == TransmissionMode.ULTRASONIC:
            self.protocol_id = 9  # Base ultrasonic mode
            if self._ggwave_instance:
                logging.info("Switched to ULTRASONIC mode (18-22kHz)")
        else:
            self.protocol_id = 1  # Base audible mode
            if self._ggwave_instance:
                logging.info("Switched to AUDIBLE mode (1-3kHz)")

        logging.info(f"Transmission mode changed: {old_mode} -> {mode} (protocol_id={self.protocol_id})")

    def get_transmission_mode(self) -> TransmissionMode:
        """
        Get current transmission mode

        Returns:
            Current TransmissionMode
        """
        return self.config.mode

    def auto_select_mode(self) -> TransmissionMode:
        """
        Automatically select best transmission mode based on connection type

        Returns:
            Selected TransmissionMode

        Note:
            - Cable connections use ULTRASONIC for higher bandwidth
            - Air connections use AUDIBLE for better propagation
        """
        if self._connection_type == ConnectionType.CABLE:
            recommended_mode = TransmissionMode.ULTRASONIC
            logging.info("Cable detected - recommending ULTRASONIC mode")
        else:
            recommended_mode = TransmissionMode.AUDIBLE
            logging.info("Air connection detected - recommending AUDIBLE mode")

        self.set_transmission_mode(recommended_mode)
        return recommended_mode

    def get_supported_protocols(self) -> Dict[str, List[int]]:
        """
        Get list of supported ggwave protocol IDs by mode

        Returns:
            Dictionary mapping mode names to protocol ID lists
        """
        return {
            'audible': [1, 2, 3],  # NORMAL, FAST, FASTEST
            'ultrasonic': [9, 10, 11],  # ULTRASONIC, ULTRASONIC_FAST, ULTRASONIC_FASTEST
            'current': [self.protocol_id]
        }

    def set_protocol_speed(self, speed: str = 'normal'):
        """
        Set protocol speed variant within current mode

        Args:
            speed: One of 'normal', 'fast', 'fastest'

        Raises:
            ValueError: If speed is invalid
        """
        speed_map = {
            'normal': 0,
            'fast': 1,
            'fastest': 2
        }

        if speed not in speed_map:
            raise ValueError(f"Invalid speed '{speed}'. Must be one of: {list(speed_map.keys())}")

        offset = speed_map[speed]

        if self.config.mode == TransmissionMode.ULTRASONIC:
            self.protocol_id = 9 + offset  # 9, 10, or 11
        else:
            self.protocol_id = 1 + offset  # 1, 2, or 3

        logging.info(f"Protocol speed set to {speed} (protocol_id={self.protocol_id})")

    def optimize_transmission_parameters(self):
        """
        Automatically optimize transmission parameters based on connection type and quality

        Adjusts:
        - Transmission mode (audible vs ultrasonic)
        - Protocol speed (normal/fast/fastest)
        - Frame payload size
        - Target bandwidth

        Returns best configuration based on detected connection characteristics
        """
        # First, ensure we have recent connection info
        if self._connection_type == ConnectionType.UNKNOWN:
            logging.warning("Connection type unknown, running detection...")
            self.detect_cable()

        # Select mode based on connection
        if self._connection_type == ConnectionType.CABLE:
            # Cable: use ultrasonic for higher bandwidth
            self.set_transmission_mode(TransmissionMode.ULTRASONIC)

            # High SNR allows fastest protocol
            if self._last_snr > 50.0:
                self.set_protocol_speed('fastest')
                self.config.max_payload_size = 240  # Maximum
                target_bps = 400
            elif self._last_snr > 45.0:
                self.set_protocol_speed('fast')
                self.config.max_payload_size = 220
                target_bps = 350
            else:
                self.set_protocol_speed('normal')
                self.config.max_payload_size = 200
                target_bps = 300

            logging.info(f"Cable optimized: speed={self.protocol_id}, "
                        f"payload={self.config.max_payload_size}, target={target_bps}bps")

        else:
            # Air: use audible for better propagation
            self.set_transmission_mode(TransmissionMode.AUDIBLE)

            # Conservative settings for air transmission
            if self._last_snr > 15.0:
                self.set_protocol_speed('fast')
                self.config.max_payload_size = 180
                target_bps = 225
            else:
                self.set_protocol_speed('normal')
                self.config.max_payload_size = 160
                target_bps = 200

            logging.info(f"Air optimized: speed={self.protocol_id}, "
                        f"payload={self.config.max_payload_size}, target={target_bps}bps")

        # Update config bandwidth targets
        if self._connection_type == ConnectionType.CABLE:
            self.config.cable_bps = target_bps
        else:
            self.config.air_bps = target_bps

    def adapt_to_quality(self, error_rate: float = 0.0):
        """
        Adapt transmission parameters based on observed quality metrics

        Args:
            error_rate: Observed error rate (0.0 to 1.0)

        Automatically reduces transmission speed if errors are high,
        or increases speed if transmission is very reliable
        """
        quality_metrics = self.monitor_transmission_quality()

        # High error rate or low stability: reduce speed
        if error_rate > 0.1 or quality_metrics['stability'] < 0.5:
            current_speed = (self.protocol_id - 1) % 3 if self.config.mode == TransmissionMode.AUDIBLE else (self.protocol_id - 9) % 3

            if current_speed > 0:
                # Downgrade speed
                new_speed = ['normal', 'fast', 'fastest'][max(0, current_speed - 1)]
                self.set_protocol_speed(new_speed)
                self.config.max_payload_size = max(160, self.config.max_payload_size - 20)
                logging.warning(f"Reduced transmission speed due to quality issues (error_rate={error_rate:.2%}, stability={quality_metrics['stability']:.2f})")

        # Very low error rate and high stability: try increasing speed
        elif error_rate < 0.01 and quality_metrics['stability'] > 0.9:
            current_speed = (self.protocol_id - 1) % 3 if self.config.mode == TransmissionMode.AUDIBLE else (self.protocol_id - 9) % 3

            if current_speed < 2:
                # Upgrade speed
                new_speed = ['normal', 'fast', 'fastest'][min(2, current_speed + 1)]
                self.set_protocol_speed(new_speed)
                self.config.max_payload_size = min(240, self.config.max_payload_size + 20)
                logging.info(f"Increased transmission speed due to excellent quality (error_rate={error_rate:.2%}, stability={quality_metrics['stability']:.2f})")

    def get_adaptive_frame_size(self) -> int:
        """
        Get adaptive frame size based on current conditions

        Returns:
            Recommended frame payload size in bytes
        """
        base_size = self.config.max_payload_size

        # Adjust based on bandwidth variability
        quality_metrics = self.monitor_transmission_quality()

        if quality_metrics['stability'] < 0.5:
            # High variability: use smaller frames
            return max(100, int(base_size * 0.7))
        elif quality_metrics['stability'] > 0.9:
            # Very stable: can use larger frames
            return min(240, int(base_size * 1.1))
        else:
            return base_size

    def get_adaptive_timeout(self, base_timeout: float = 5.0) -> float:
        """
        Calculate adaptive timeout based on connection quality and bandwidth

        Args:
            base_timeout: Base timeout value in seconds

        Returns:
            Adaptive timeout in seconds
        """
        # Adjust based on connection type
        if self._connection_type == ConnectionType.CABLE:
            multiplier = 1.0  # Cable is fast and reliable
        elif self._connection_type == ConnectionType.AIR:
            multiplier = 1.5  # Air needs more time
        else:
            multiplier = 2.0  # Unknown connection: be conservative

        # Adjust based on recent transmission quality
        quality_metrics = self.monitor_transmission_quality()

        if quality_metrics['stability'] < 0.5:
            multiplier *= 1.5  # Add more time for unstable connection
        elif quality_metrics['trend'] < -0.3:
            multiplier *= 1.3  # Connection degrading, add buffer

        return base_timeout * multiplier

    def get_transmission_config(self) -> Dict[str, any]:
        """
        Get complete transmission configuration

        Returns:
            Dictionary with all transmission parameters:
            - mode: TransmissionMode
            - protocol_id: ggwave protocol ID
            - max_payload_size: Maximum bytes per frame
            - target_bandwidth: Target bandwidth in bps
            - connection_type: Detected connection type
            - snr_db: Last measured SNR
            - adaptive_frame_size: Current recommended frame size
            - adaptive_timeout: Current recommended timeout
        """
        return {
            'mode': self.config.mode,
            'protocol_id': self.protocol_id,
            'max_payload_size': self.config.max_payload_size,
            'target_bandwidth': self.config.cable_bps if self._connection_type == ConnectionType.CABLE else self.config.air_bps,
            'connection_type': self._connection_type,
            'snr_db': self._last_snr,
            'adaptive_frame_size': self.get_adaptive_frame_size(),
            'adaptive_timeout': self.get_adaptive_timeout()
        }

    def transmit(self, data: bytes) -> bool:
        """
        Transmit data via audio

        Args:
            data: Data to transmit

        Returns:
            True if transmission successful
        """
        try:
            start_time = time.time()

            # Encode data to audio samples
            audio_samples = self.encode(data)
            if len(audio_samples) == 0:
                logging.error("Failed to encode data")
                return False

            # Play audio if sounddevice available
            if SOUNDDEVICE_AVAILABLE:
                sd.play(audio_samples, self.sample_rate)
                sd.wait()
            else:
                logging.warning("sounddevice not available, simulating transmission")
                # Simulate transmission time
                time.sleep(len(audio_samples) / self.sample_rate)

            # Update bandwidth tracking
            elapsed = time.time() - start_time
            self._last_tx_time = elapsed
            self._last_tx_bytes = len(data)

            if elapsed > 0:
                bps = len(data) / elapsed
                self._bandwidth_samples.append(bps)
                # Keep last 100 samples
                if len(self._bandwidth_samples) > 100:
                    self._bandwidth_samples.pop(0)

            logging.debug(f"Transmitted {len(data)} bytes in {elapsed:.2f}s ({len(data)/elapsed:.1f} B/s)")
            return True

        except Exception as e:
            logging.error(f"Transmission failed: {e}")
            return False

    def receive(self, timeout: float = 30.0) -> Optional[bytes]:
        """
        Receive data via audio

        Args:
            timeout: Timeout in seconds

        Returns:
            Received data or None if timeout
        """
        if not SOUNDDEVICE_AVAILABLE:
            logging.error("sounddevice not available, cannot receive audio")
            return None

        try:
            # Calculate buffer size (use timeout duration)
            frames = int(timeout * self.sample_rate)

            logging.info(f"Recording for up to {timeout}s...")
            recording = sd.rec(frames, samplerate=self.sample_rate, channels=1, dtype='float32')
            sd.wait()

            # Decode the recording
            audio_samples = recording.flatten()
            decoded = self.decode(audio_samples)

            if decoded:
                logging.info(f"Received {len(decoded)} bytes")
            else:
                logging.warning("No data decoded from audio")

            return decoded

        except Exception as e:
            logging.error(f"Reception failed: {e}")
            return None

    def start_recording(self):
        """Start continuous audio recording for reception"""
        if not SOUNDDEVICE_AVAILABLE:
            logging.error("sounddevice not available")
            return

        self._is_recording = True
        self._recording_buffer = []

        def callback(indata, frames, time_info, status):
            if status:
                logging.warning(f"Recording status: {status}")
            if self._is_recording:
                self._recording_buffer.append(indata.copy())

        self._stream = sd.InputStream(
            callback=callback,
            channels=1,
            samplerate=self.sample_rate,
            dtype='float32'
        )
        self._stream.start()
        logging.info("Recording started")

    def stop_recording(self) -> Optional[bytes]:
        """Stop recording and decode accumulated buffer"""
        if not self._is_recording:
            return None

        self._is_recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._recording_buffer:
            return None

        # Concatenate all recorded chunks
        audio_data = np.concatenate(self._recording_buffer)
        self._recording_buffer = []

        # Decode
        return self.decode(audio_data.flatten())

    def __del__(self):
        """Cleanup resources"""
        if self._stream:
            self._stream.stop()
            self._stream.close()
