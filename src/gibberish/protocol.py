"""
Communication protocol handling for acoustic data transmission.

This module implements the TCP-style error correction protocol with
ACK/NACK handling, frame management, and retransmission logic.
"""

from typing import Optional, Tuple, Dict, List
from enum import Enum
from dataclasses import dataclass, field
import struct
import uuid
import time
import logging
import zlib


class HandshakeState(Enum):
    """States in the handshake state machine"""
    IDLE = 0
    INIT_SENT = 1
    ACK_SENT = 2
    READY_SENT = 3
    ESTABLISHED = 4
    FAILED = 5


class FrameType(Enum):
    """Types of protocol frames"""
    DATA = 0x01
    ACK = 0x02
    NACK = 0x03
    HANDSHAKE_INIT = 0x04
    HANDSHAKE_ACK = 0x05
    HANDSHAKE_READY = 0x06
    TREE_HASH = 0x07
    SYNC_COMPLETE = 0x08


@dataclass
class Frame:
    """
    Represents a protocol frame with error correction

    Frame structure (total: 265 bytes max):
    - Frame type (1 byte): Type of frame (DATA, ACK, NACK, etc.)
    - Sequence number (2 bytes): Frame sequence number (0-65535)
    - Total frames (2 bytes): Total number of frames in transmission
    - File ID (16 bytes): UUID identifying the file/transmission
    - CRC32 checksum (4 bytes): CRC32 of frame type + seq + total + file_id + payload
    - Payload length (1 byte): Actual payload size (0-256)
    - Payload (up to 256 bytes): Actual data

    Total header: 26 bytes + payload (0-256 bytes)
    """
    frame_type: FrameType
    sequence_number: int
    total_frames: int
    file_id: str  # UUID string
    payload: bytes
    crc32: int = 0

    # Maximum payload size
    MAX_PAYLOAD_SIZE = 256

    def calculate_crc32(self) -> int:
        """
        Calculate CRC32 checksum for frame integrity

        Returns:
            CRC32 checksum value
        """
        # Build data to checksum (everything except the CRC field itself)
        data = struct.pack('B', self.frame_type.value)  # 1 byte
        data += struct.pack('!H', self.sequence_number)  # 2 bytes (big-endian)
        data += struct.pack('!H', self.total_frames)  # 2 bytes (big-endian)

        # Convert UUID string to bytes
        file_uuid = uuid.UUID(self.file_id)
        data += file_uuid.bytes  # 16 bytes

        # Add payload length and payload
        data += struct.pack('B', len(self.payload))  # 1 byte
        data += self.payload

        # Calculate CRC32
        return zlib.crc32(data) & 0xffffffff

    def pack(self) -> bytes:
        """
        Pack frame into bytes for transmission

        Returns:
            Packed frame bytes
        """
        # Validate payload size
        if len(self.payload) > self.MAX_PAYLOAD_SIZE:
            raise ValueError(f"Payload size {len(self.payload)} exceeds maximum {self.MAX_PAYLOAD_SIZE}")

        # Calculate CRC32 if not already set
        if self.crc32 == 0:
            self.crc32 = self.calculate_crc32()

        # Pack frame components
        data = struct.pack('B', self.frame_type.value)  # 1 byte
        data += struct.pack('!H', self.sequence_number)  # 2 bytes
        data += struct.pack('!H', self.total_frames)  # 2 bytes

        # Pack file ID (UUID)
        file_uuid = uuid.UUID(self.file_id)
        data += file_uuid.bytes  # 16 bytes

        # Pack CRC32
        data += struct.pack('!I', self.crc32)  # 4 bytes

        # Pack payload length and payload
        data += struct.pack('B', len(self.payload))  # 1 byte
        data += self.payload  # 0-256 bytes

        return data

    @staticmethod
    def unpack(data: bytes) -> Optional['Frame']:
        """
        Unpack bytes into frame

        Args:
            data: Raw frame bytes

        Returns:
            Unpacked Frame object or None if invalid
        """
        try:
            # Minimum frame size: 26 bytes header + 0 bytes payload
            if len(data) < 26:
                logging.error(f"Frame too short: {len(data)} bytes")
                return None

            offset = 0

            # Unpack frame type
            frame_type_value = struct.unpack('B', data[offset:offset+1])[0]
            offset += 1
            try:
                frame_type = FrameType(frame_type_value)
            except ValueError:
                logging.error(f"Invalid frame type: {frame_type_value}")
                return None

            # Unpack sequence number and total frames
            sequence_number = struct.unpack('!H', data[offset:offset+2])[0]
            offset += 2
            total_frames = struct.unpack('!H', data[offset:offset+2])[0]
            offset += 2

            # Unpack file ID (UUID)
            file_uuid_bytes = data[offset:offset+16]
            offset += 16
            file_id = str(uuid.UUID(bytes=file_uuid_bytes))

            # Unpack CRC32
            crc32_received = struct.unpack('!I', data[offset:offset+4])[0]
            offset += 4

            # Unpack payload length
            payload_length = struct.unpack('B', data[offset:offset+1])[0]
            offset += 1

            # Validate payload length
            if payload_length > Frame.MAX_PAYLOAD_SIZE:
                logging.error(f"Invalid payload length: {payload_length}")
                return None

            # Check we have enough data for payload
            if len(data) < offset + payload_length:
                logging.error(f"Incomplete payload: expected {payload_length}, got {len(data) - offset}")
                return None

            # Unpack payload
            payload = data[offset:offset+payload_length]

            # Create frame
            frame = Frame(
                frame_type=frame_type,
                sequence_number=sequence_number,
                total_frames=total_frames,
                file_id=file_id,
                payload=payload,
                crc32=crc32_received
            )

            return frame

        except Exception as e:
            logging.error(f"Failed to unpack frame: {e}")
            return None

    def verify_checksum(self) -> bool:
        """
        Verify frame integrity using CRC32 checksum

        Returns:
            True if CRC32 matches, False otherwise
        """
        calculated_crc = self.calculate_crc32()
        is_valid = calculated_crc == self.crc32

        if not is_valid:
            logging.warning(f"CRC mismatch: calculated={calculated_crc:08x}, received={self.crc32:08x}")

        return is_valid


@dataclass
class HandshakeSession:
    """Tracks handshake session state"""
    session_id: str
    protocol_version: int
    state: HandshakeState
    role: str  # 'initiator' or 'responder'
    peer_session_id: Optional[str] = None
    peer_protocol_version: Optional[int] = None
    start_time: float = 0.0
    last_activity: float = 0.0

    def __post_init__(self):
        if self.start_time == 0.0:
            self.start_time = time.time()
        if self.last_activity == 0.0:
            self.last_activity = time.time()

    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()

    def elapsed_time(self) -> float:
        """Get elapsed time since handshake start"""
        return time.time() - self.start_time

    def time_since_activity(self) -> float:
        """Get time since last activity"""
        return time.time() - self.last_activity


@dataclass
class TransmissionState:
    """
    Tracks state of an ongoing transmission session
    """
    file_id: str  # UUID of the transmission
    total_frames: int
    current_sequence: int = 0
    frames_sent: int = 0
    frames_acked: int = 0
    frames_nacked: int = 0
    retransmissions: int = 0
    start_time: float = field(default_factory=time.time)
    last_frame_time: float = field(default_factory=time.time)

    def update_frame_time(self):
        """Update timestamp of last frame activity"""
        self.last_frame_time = time.time()

    def elapsed_time(self) -> float:
        """Get elapsed time since transmission start"""
        return time.time() - self.start_time

    def get_progress(self) -> float:
        """
        Get transmission progress as percentage

        Returns:
            Progress percentage (0-100)
        """
        if self.total_frames == 0:
            return 0.0
        return (self.frames_acked / self.total_frames) * 100.0

    def get_retransmission_rate(self) -> float:
        """
        Get retransmission rate as percentage

        Returns:
            Retransmission rate (0-100)
        """
        if self.frames_sent == 0:
            return 0.0
        return (self.retransmissions / self.frames_sent) * 100.0


class ProtocolHandler:
    """Handles protocol-level operations"""

    PROTOCOL_VERSION = 1
    HANDSHAKE_TIMEOUT = 30.0  # 30 seconds
    INIT_BEACON_INTERVAL = 2.0  # 2 seconds

    def __init__(self, max_retries: int = 3, frame_size: int = 256):
        """
        Initialize the protocol handler

        Args:
            max_retries: Maximum number of retransmission attempts
            frame_size: Maximum payload size per frame
        """
        self.max_retries = max_retries
        self.frame_size = frame_size
        self._sequence_number = 0
        self._rtt_samples = []

        # Handshake state
        self._handshake_session: Optional[HandshakeSession] = None
        self._connection_established = False

        # Transmission state
        self._transmission_state: Optional[TransmissionState] = None

        # RTT tracking for adaptive timeout
        self._rtt_avg = 2.0  # Initial RTT estimate (2 seconds)
        self._rtt_var = 0.5  # RTT variance estimate

        # Reassembly buffers for receiving
        self._reassembly_buffers: Dict[str, ReassemblyBuffer] = {}  # file_id -> buffer

    def create_frame(self, frame_type: FrameType, file_id: str, total_frames: int,
                     payload: bytes = b"") -> Frame:
        """
        Create a protocol frame

        Args:
            frame_type: Type of frame
            file_id: UUID identifying the transmission
            total_frames: Total number of frames in transmission
            payload: Frame payload

        Returns:
            Created frame
        """
        frame = Frame(
            frame_type=frame_type,
            sequence_number=self._sequence_number,
            total_frames=total_frames,
            file_id=file_id,
            payload=payload
        )
        self._sequence_number = (self._sequence_number + 1) % 65536
        return frame

    def send_with_ack(self, audio_manager, frame: Frame, timeout: Optional[float] = None) -> bool:
        """
        Send frame and wait for ACK (stop-and-wait protocol)

        Implements stop-and-wait ARQ:
        1. Send frame
        2. Wait for ACK/NACK with timeout
        3. Return True if ACK received, False if NACK or timeout

        Args:
            audio_manager: AudioManager instance for transmission
            frame: Frame to send
            timeout: Timeout in seconds (uses adaptive timeout if None)

        Returns:
            True if ACK received, False otherwise
        """
        if timeout is None:
            timeout = self.calculate_timeout()

        try:
            # Pack and send frame
            frame_data = frame.pack()
            logging.info(f"Sending frame seq={frame.sequence_number}, size={len(frame_data)} bytes")

            # Record send time for RTT measurement
            send_time = time.time()

            # Transmit via audio manager
            success = audio_manager.transmit(frame_data)
            if not success:
                logging.error("Failed to transmit frame")
                return False

            # Update transmission state
            if self._transmission_state:
                self._transmission_state.frames_sent += 1
                self._transmission_state.update_frame_time()

            # Wait for ACK/NACK
            logging.info(f"Waiting for ACK (timeout={timeout:.2f}s)...")
            response = self._wait_for_ack_nack(audio_manager, frame.sequence_number, timeout)

            if response is None:
                logging.warning(f"Timeout waiting for ACK/NACK (seq={frame.sequence_number})")
                return False

            response_type, response_seq = response

            # Verify sequence number matches
            if response_seq != frame.sequence_number:
                logging.warning(f"Sequence mismatch: sent={frame.sequence_number}, ack={response_seq}")
                return False

            # Handle ACK
            if response_type == FrameType.ACK:
                # Measure RTT
                rtt = time.time() - send_time
                self._update_rtt(rtt)

                logging.info(f"ACK received for seq={frame.sequence_number} (RTT={rtt:.3f}s)")

                # Update transmission state
                if self._transmission_state:
                    self._transmission_state.frames_acked += 1

                return True

            # Handle NACK
            elif response_type == FrameType.NACK:
                logging.warning(f"NACK received for seq={frame.sequence_number}")

                # Update transmission state
                if self._transmission_state:
                    self._transmission_state.frames_nacked += 1

                return False

            else:
                logging.error(f"Unexpected response type: {response_type}")
                return False

        except Exception as e:
            logging.error(f"Error in send_with_ack: {e}")
            return False

    def _wait_for_ack_nack(self, audio_manager, expected_seq: int,
                          timeout: float) -> Optional[Tuple[FrameType, int]]:
        """
        Wait for ACK or NACK response

        Args:
            audio_manager: AudioManager instance
            expected_seq: Expected sequence number
            timeout: Timeout in seconds

        Returns:
            Tuple of (frame_type, sequence_number) or None if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Receive with remaining timeout
                remaining = timeout - (time.time() - start_time)
                received_data = audio_manager.receive(timeout=min(1.0, remaining))

                if not received_data or len(received_data) < 26:
                    continue

                # Unpack frame
                frame = Frame.unpack(received_data)
                if not frame:
                    continue

                # Verify checksum
                if not frame.verify_checksum():
                    logging.warning(f"Received frame with invalid checksum")
                    continue

                # Check if it's ACK or NACK
                if frame.frame_type in [FrameType.ACK, FrameType.NACK]:
                    return frame.frame_type, frame.sequence_number

            except Exception as e:
                logging.debug(f"Error waiting for ACK/NACK: {e}")
                continue

        return None

    def handle_retransmission(self, audio_manager, frame: Frame) -> bool:
        """
        Handle frame retransmission on NACK with automatic retry

        Implements automatic retransmission logic:
        1. Try up to max_retries times
        2. Use exponential backoff between attempts
        3. Track retransmission statistics
        4. Return True only if ACK received

        Args:
            audio_manager: AudioManager instance for transmission
            frame: Frame to retransmit

        Returns:
            True if successfully retransmitted and ACKed
        """
        logging.info(f"Starting retransmission for frame seq={frame.sequence_number}")

        for attempt in range(self.max_retries):
            logging.info(f"Retransmission attempt {attempt + 1}/{self.max_retries} for seq={frame.sequence_number}")

            # Update retransmission counter
            if self._transmission_state:
                self._transmission_state.retransmissions += 1

            # Exponential backoff: wait before retrying
            if attempt > 0:
                backoff_delay = min(0.5 * (2 ** attempt), 5.0)  # Cap at 5 seconds
                logging.info(f"Backing off for {backoff_delay:.2f}s before retry...")
                time.sleep(backoff_delay)

            # Attempt to send with ACK
            success = self.send_with_ack(audio_manager, frame)

            if success:
                logging.info(f"Retransmission successful for seq={frame.sequence_number} on attempt {attempt + 1}")
                return True
            else:
                logging.warning(f"Retransmission attempt {attempt + 1} failed for seq={frame.sequence_number}")

        # All retries exhausted
        logging.error(f"All {self.max_retries} retransmission attempts failed for seq={frame.sequence_number}")
        return False

    def send_frame_reliable(self, audio_manager, frame: Frame) -> bool:
        """
        Send frame with automatic retransmission on failure

        Combines send_with_ack and handle_retransmission into a single
        reliable send operation that handles both initial transmission
        and retries transparently.

        Args:
            audio_manager: AudioManager instance
            frame: Frame to send

        Returns:
            True if frame was successfully delivered (ACK received)
        """
        # Try initial transmission
        success = self.send_with_ack(audio_manager, frame)

        if success:
            return True

        # Initial transmission failed or NACK received - retry
        logging.warning(f"Initial transmission failed for seq={frame.sequence_number}, retrying...")
        return self.handle_retransmission(audio_manager, frame)

    def calculate_timeout(self) -> float:
        """
        Calculate adaptive timeout based on measured RTT

        Uses TCP-style adaptive timeout calculation:
        timeout = RTT_avg + 4 * RTT_var

        This provides a balance between responsiveness and avoiding
        premature timeouts due to RTT variance.

        Returns:
            Timeout in seconds (minimum 1.0s, maximum 10.0s)
        """
        # Calculate timeout using TCP formula
        timeout = self._rtt_avg + 4 * self._rtt_var

        # Clamp to reasonable bounds
        timeout = max(1.0, min(timeout, 10.0))

        logging.debug(f"Adaptive timeout: {timeout:.2f}s (avg={self._rtt_avg:.2f}s, var={self._rtt_var:.2f}s)")

        return timeout

    def _update_rtt(self, measured_rtt: float):
        """
        Update RTT estimates using exponential weighted moving average

        Uses TCP's RTT estimation algorithm:
        - RTT_avg = (1 - alpha) * RTT_avg + alpha * measured_RTT
        - RTT_var = (1 - beta) * RTT_var + beta * |measured_RTT - RTT_avg|

        Where alpha = 0.125 and beta = 0.25 (standard TCP values)

        Args:
            measured_rtt: Newly measured round-trip time in seconds
        """
        alpha = 0.125  # Smoothing factor for average
        beta = 0.25    # Smoothing factor for variance

        # Update variance first (using old average)
        rtt_diff = abs(measured_rtt - self._rtt_avg)
        self._rtt_var = (1 - beta) * self._rtt_var + beta * rtt_diff

        # Update average
        self._rtt_avg = (1 - alpha) * self._rtt_avg + alpha * measured_rtt

        # Add to samples list for statistics
        self._rtt_samples.append(measured_rtt)

        # Keep only recent samples (last 100)
        if len(self._rtt_samples) > 100:
            self._rtt_samples.pop(0)

        logging.debug(f"RTT updated: measured={measured_rtt:.3f}s, avg={self._rtt_avg:.3f}s, var={self._rtt_var:.3f}s")

    def get_rtt_statistics(self) -> Dict[str, float]:
        """
        Get RTT statistics

        Returns:
            Dictionary with RTT metrics:
            - avg: Current average RTT
            - var: Current RTT variance
            - min: Minimum observed RTT
            - max: Maximum observed RTT
            - samples: Number of RTT samples collected
        """
        stats = {
            'avg': self._rtt_avg,
            'var': self._rtt_var,
            'samples': len(self._rtt_samples)
        }

        if self._rtt_samples:
            stats['min'] = min(self._rtt_samples)
            stats['max'] = max(self._rtt_samples)
        else:
            stats['min'] = 0.0
            stats['max'] = 0.0

        return stats

    def get_transmission_statistics(self) -> Dict[str, any]:
        """
        Get comprehensive transmission statistics and quality metrics

        Returns:
            Dictionary with transmission metrics:
            - frames_sent: Total frames sent
            - frames_acked: Frames successfully acknowledged
            - frames_nacked: Frames negatively acknowledged
            - retransmissions: Total retransmission attempts
            - retransmission_rate: Percentage of frames requiring retransmission
            - success_rate: Percentage of frames successfully delivered
            - progress: Transmission progress percentage
            - elapsed_time: Time since transmission started
            - throughput_bps: Estimated throughput in bits per second
            - rtt_avg: Average round-trip time
            - rtt_var: RTT variance
            - quality_score: Overall quality score (0-100)
        """
        if not self._transmission_state:
            return {
                'frames_sent': 0,
                'frames_acked': 0,
                'frames_nacked': 0,
                'retransmissions': 0,
                'retransmission_rate': 0.0,
                'success_rate': 0.0,
                'progress': 0.0,
                'elapsed_time': 0.0,
                'throughput_bps': 0.0,
                'rtt_avg': self._rtt_avg,
                'rtt_var': self._rtt_var,
                'quality_score': 0.0
            }

        state = self._transmission_state

        # Calculate rates
        retrans_rate = state.get_retransmission_rate()
        success_rate = (state.frames_acked / state.frames_sent * 100.0) if state.frames_sent > 0 else 0.0
        progress = state.get_progress()

        # Calculate throughput
        elapsed = state.elapsed_time()
        if elapsed > 0:
            # Estimate based on ACKed frames
            bytes_acked = state.frames_acked * self.frame_size
            throughput_bps = (bytes_acked * 8) / elapsed
        else:
            throughput_bps = 0.0

        # Calculate quality score (0-100)
        # Based on: success rate, retransmission rate, RTT
        quality_score = self._calculate_quality_score(success_rate, retrans_rate)

        return {
            'frames_sent': state.frames_sent,
            'frames_acked': state.frames_acked,
            'frames_nacked': state.frames_nacked,
            'retransmissions': state.retransmissions,
            'retransmission_rate': retrans_rate,
            'success_rate': success_rate,
            'progress': progress,
            'elapsed_time': elapsed,
            'throughput_bps': throughput_bps,
            'rtt_avg': self._rtt_avg,
            'rtt_var': self._rtt_var,
            'quality_score': quality_score
        }

    def _calculate_quality_score(self, success_rate: float, retrans_rate: float) -> float:
        """
        Calculate overall connection quality score

        Args:
            success_rate: Frame success rate (0-100)
            retrans_rate: Retransmission rate (0-100)

        Returns:
            Quality score (0-100)
        """
        # Weight success rate heavily (70%)
        score = success_rate * 0.7

        # Penalize retransmissions (30%)
        retrans_penalty = (100 - retrans_rate) * 0.3
        score += retrans_penalty

        # RTT penalty: higher RTT = lower score
        if self._rtt_avg > 5.0:
            rtt_penalty = min((self._rtt_avg - 5.0) * 5, 20)  # Max 20 point penalty
            score -= rtt_penalty

        # Clamp to 0-100
        return max(0.0, min(100.0, score))

    def display_transmission_stats(self):
        """
        Display transmission statistics in a formatted way
        """
        stats = self.get_transmission_statistics()

        print("\n" + "=" * 70)
        print("  TRANSMISSION STATISTICS")
        print("=" * 70)

        print("\n[FRAMES]")
        print(f"  Sent: {stats['frames_sent']}")
        print(f"  ACKed: {stats['frames_acked']}")
        print(f"  NACKed: {stats['frames_nacked']}")
        print(f"  Retransmissions: {stats['retransmissions']}")

        print("\n[RATES]")
        print(f"  Success Rate: {stats['success_rate']:.1f}%")
        print(f"  Retransmission Rate: {stats['retransmission_rate']:.1f}%")
        print(f"  Progress: {stats['progress']:.1f}%")

        print("\n[PERFORMANCE]")
        print(f"  Elapsed Time: {stats['elapsed_time']:.2f}s")
        print(f"  Throughput: {stats['throughput_bps']:.0f} bps")
        print(f"  RTT Average: {stats['rtt_avg']:.3f}s")
        print(f"  RTT Variance: {stats['rtt_var']:.3f}s")

        print("\n[QUALITY]")
        score = stats['quality_score']
        if score >= 90:
            quality_bar = "█████ EXCELLENT"
        elif score >= 75:
            quality_bar = "████░ GOOD"
        elif score >= 50:
            quality_bar = "███░░ FAIR"
        elif score >= 25:
            quality_bar = "██░░░ POOR"
        else:
            quality_bar = "█░░░░ VERY POOR"

        print(f"  Quality Score: {quality_bar} ({score:.0f}/100)")

        print("\n" + "=" * 70 + "\n")

    def start_transmission(self, file_id: str, total_frames: int):
        """
        Initialize a new transmission session

        Args:
            file_id: UUID for the transmission
            total_frames: Total number of frames to send
        """
        self._transmission_state = TransmissionState(
            file_id=file_id,
            total_frames=total_frames
        )
        logging.info(f"Started transmission: file_id={file_id[:8]}..., total_frames={total_frames}")

    def get_transmission_state(self) -> Optional[TransmissionState]:
        """
        Get current transmission state

        Returns:
            TransmissionState object or None
        """
        return self._transmission_state

    def handle_out_of_order_frame(self, frame: Frame, buffer: 'ReassemblyBuffer') -> bool:
        """
        Handle out-of-order frame delivery

        Frames may arrive out of order due to varying transmission times.
        This method ensures frames are buffered correctly regardless of
        arrival order.

        Args:
            frame: Received frame
            buffer: Reassembly buffer for this transmission

        Returns:
            True if frame was successfully handled
        """
        seq = frame.sequence_number

        # Check if frame is within valid range
        if seq >= buffer.total_frames:
            logging.error(f"Frame sequence {seq} exceeds total frames {buffer.total_frames}")
            return False

        # Check if already received (duplicate)
        if seq in buffer.received_frames:
            logging.info(f"Duplicate frame {seq} (out-of-order), already buffered")
            return True  # Not an error, just redundant

        # Add to buffer - ReassemblyBuffer handles out-of-order automatically
        logging.info(f"Buffering out-of-order frame {seq}")
        buffer.add_frame(seq, frame.payload)

        return True

    def handle_missing_frames(self, audio_manager, buffer: 'ReassemblyBuffer',
                             timeout_per_frame: float = 10.0) -> bool:
        """
        Handle missing frames by requesting retransmission

        When some frames are missing after initial transmission, this
        method can request specific frames to be retransmitted.

        Args:
            audio_manager: AudioManager instance
            buffer: Reassembly buffer with missing frames
            timeout_per_frame: Timeout for each missing frame

        Returns:
            True if all missing frames were recovered
        """
        missing = buffer.get_missing_sequences()

        if not missing:
            return True

        logging.warning(f"Missing {len(missing)} frames: {missing[:20]}...")

        # In a full implementation, we would send retransmission requests
        # For now, we wait for sender to detect timeout and retransmit
        start_time = time.time()
        timeout = timeout_per_frame * len(missing)

        while buffer.get_missing_sequences() and time.time() - start_time < timeout:
            # Receive any incoming frames
            frame = self.receive_frame(audio_manager)

            if frame is None:
                time.sleep(0.1)
                continue

            # Verify it's for this transmission
            if frame.file_id != buffer.file_id:
                continue

            # Handle the frame
            if frame.frame_type == FrameType.DATA:
                self.handle_out_of_order_frame(frame, buffer)
                self.send_ack(audio_manager, frame.sequence_number, buffer.file_id)

        # Check if we recovered all frames
        missing_after = buffer.get_missing_sequences()
        if missing_after:
            logging.error(f"Still missing {len(missing_after)} frames after recovery attempt")
            return False

        logging.info("Successfully recovered all missing frames")
        return True

    def handle_transmission_error(self, error_type: str, context: Dict[str, any]) -> bool:
        """
        Handle various transmission error scenarios

        Args:
            error_type: Type of error ('timeout', 'corruption', 'sequence_error', etc.)
            context: Context information about the error

        Returns:
            True if error was handled and recovery is possible
        """
        logging.error(f"Transmission error: {error_type}")

        if error_type == 'timeout':
            # Timeout waiting for ACK
            seq = context.get('sequence_number', -1)
            logging.error(f"Timeout on frame {seq}, will retry")
            return True  # Retry is possible

        elif error_type == 'corruption':
            # CRC32 mismatch
            seq = context.get('sequence_number', -1)
            logging.error(f"Corruption detected on frame {seq}")
            return True  # Send NACK and retry

        elif error_type == 'sequence_error':
            # Unexpected sequence number
            expected = context.get('expected', -1)
            received = context.get('received', -1)
            logging.error(f"Sequence error: expected {expected}, got {received}")
            return False  # Protocol violation, abort

        elif error_type == 'max_retries':
            # Exhausted all retries
            seq = context.get('sequence_number', -1)
            logging.error(f"Max retries exceeded for frame {seq}")
            return False  # Cannot recover

        elif error_type == 'buffer_overflow':
            # Too many pending frames
            logging.error("Reassembly buffer overflow")
            return False  # Protocol error

        else:
            logging.error(f"Unknown error type: {error_type}")
            return False

    def recover_from_error(self, audio_manager, error_type: str,
                          frame: Optional[Frame] = None) -> bool:
        """
        Attempt to recover from transmission error

        Args:
            audio_manager: AudioManager instance
            error_type: Type of error to recover from
            frame: Frame that caused the error (if applicable)

        Returns:
            True if recovery was successful
        """
        logging.info(f"Attempting recovery from {error_type}")

        if error_type == 'corruption' and frame:
            # Send NACK to request retransmission
            success = self.send_nack(audio_manager, frame.sequence_number, frame.file_id)
            if success:
                logging.info(f"Sent NACK for corrupted frame {frame.sequence_number}")
                return True
            return False

        elif error_type == 'timeout' and frame:
            # Wait a bit longer, then retry
            time.sleep(1.0)
            return True

        elif error_type == 'out_of_order':
            # Out-of-order is handled automatically by ReassemblyBuffer
            logging.info("Out-of-order delivery handled by reassembly buffer")
            return True

        else:
            logging.warning(f"No recovery strategy for {error_type}")
            return False

    def validate_transmission_integrity(self, buffer: 'ReassemblyBuffer') -> bool:
        """
        Validate integrity of complete transmission

        Args:
            buffer: Completed reassembly buffer

        Returns:
            True if transmission is complete and valid
        """
        # Check completeness
        if not buffer.is_complete():
            missing = buffer.get_missing_sequences()
            logging.error(f"Transmission incomplete: {len(missing)} frames missing")
            return False

        # Check all frames are present
        for seq in range(buffer.total_frames):
            if seq not in buffer.received_frames:
                logging.error(f"Frame {seq} missing from buffer")
                return False

        # All checks passed
        logging.info(f"Transmission integrity validated: {buffer.total_frames} frames complete")
        return True

    def verify_frame(self, frame: Frame) -> bool:
        """
        Verify frame integrity using CRC32

        Args:
            frame: Frame to verify

        Returns:
            True if CRC matches
        """
        return frame.verify_checksum()

    def receive_frame(self, audio_manager) -> Optional[Frame]:
        """
        Receive and verify a data frame

        Args:
            audio_manager: AudioManager instance

        Returns:
            Verified Frame object or None
        """
        try:
            # Receive data
            received_data = audio_manager.receive(timeout=5.0)

            if not received_data or len(received_data) < 26:
                return None

            # Unpack frame
            frame = Frame.unpack(received_data)
            if not frame:
                logging.warning("Failed to unpack received frame")
                return None

            # Verify integrity
            if not self.verify_frame(frame):
                logging.warning(f"Frame {frame.sequence_number} failed integrity check")
                return None

            logging.info(f"Received valid frame seq={frame.sequence_number}")
            return frame

        except Exception as e:
            logging.error(f"Error receiving frame: {e}")
            return None

    def send_ack(self, audio_manager, sequence_number: int, file_id: str) -> bool:
        """
        Send ACK for received frame

        Args:
            audio_manager: AudioManager instance
            sequence_number: Sequence number to acknowledge
            file_id: File ID of the transmission

        Returns:
            True if ACK sent successfully
        """
        try:
            # Create ACK frame (empty payload)
            ack_frame = Frame(
                frame_type=FrameType.ACK,
                sequence_number=sequence_number,
                total_frames=0,
                file_id=file_id,
                payload=b""
            )

            # Pack and send
            ack_data = ack_frame.pack()
            success = audio_manager.transmit(ack_data)

            if success:
                logging.info(f"Sent ACK for seq={sequence_number}")
            else:
                logging.warning(f"Failed to send ACK for seq={sequence_number}")

            return success

        except Exception as e:
            logging.error(f"Error sending ACK: {e}")
            return False

    def send_nack(self, audio_manager, sequence_number: int, file_id: str) -> bool:
        """
        Send NACK for corrupted/failed frame

        Args:
            audio_manager: AudioManager instance
            sequence_number: Sequence number to NACK
            file_id: File ID of the transmission

        Returns:
            True if NACK sent successfully
        """
        try:
            # Create NACK frame (empty payload)
            nack_frame = Frame(
                frame_type=FrameType.NACK,
                sequence_number=sequence_number,
                total_frames=0,
                file_id=file_id,
                payload=b""
            )

            # Pack and send
            nack_data = nack_frame.pack()
            success = audio_manager.transmit(nack_data)

            if success:
                logging.info(f"Sent NACK for seq={sequence_number}")
            else:
                logging.warning(f"Failed to send NACK for seq={sequence_number}")

            return success

        except Exception as e:
            logging.error(f"Error sending NACK: {e}")
            return False

    def receive_and_reassemble(self, audio_manager, file_id: str,
                               total_frames: int) -> Optional[bytes]:
        """
        Receive frames and reassemble into complete data

        Handles frame reception, ACK/NACK sending, and reassembly with
        integrity verification.

        Args:
            audio_manager: AudioManager instance
            file_id: Expected file ID for the transmission
            total_frames: Expected total number of frames

        Returns:
            Complete reassembled data or None on failure
        """
        # Create or get reassembly buffer
        if file_id not in self._reassembly_buffers:
            self._reassembly_buffers[file_id] = ReassemblyBuffer(
                file_id=file_id,
                total_frames=total_frames
            )

        buffer = self._reassembly_buffers[file_id]

        logging.info(f"Starting reassembly for {total_frames} frames (file {file_id[:8]}...)")

        # Receive frames until complete or timeout
        timeout = 60.0  # Overall timeout
        start_time = time.time()

        while not buffer.is_complete():
            if time.time() - start_time > timeout:
                logging.error(f"Reassembly timeout after {timeout}s")
                missing = buffer.get_missing_sequences()
                logging.error(f"Missing {len(missing)} frames: {missing[:10]}...")
                return None

            # Receive frame
            frame = self.receive_frame(audio_manager)

            if frame is None:
                continue

            # Verify it's for this transmission
            if frame.file_id != file_id:
                logging.warning(f"Received frame for different file: {frame.file_id[:8]}...")
                continue

            # Only process DATA frames
            if frame.frame_type != FrameType.DATA:
                logging.debug(f"Ignoring non-DATA frame: {frame.frame_type}")
                continue

            # Check if already received (duplicate)
            if frame.sequence_number in buffer.received_frames:
                logging.info(f"Duplicate frame seq={frame.sequence_number}, sending ACK anyway")
                self.send_ack(audio_manager, frame.sequence_number, file_id)
                continue

            # Add to buffer
            logging.info(f"Adding frame seq={frame.sequence_number} to buffer ({buffer.get_progress():.1f}% complete)")
            is_complete = buffer.add_frame(frame.sequence_number, frame.payload)

            # Send ACK
            self.send_ack(audio_manager, frame.sequence_number, file_id)

            if is_complete:
                logging.info("All frames received!")
                break

        # Reassemble
        data = buffer.reassemble()

        if data:
            logging.info(f"Successfully reassembled {len(data)} bytes")
            # Clean up buffer
            del self._reassembly_buffers[file_id]
            return data
        else:
            logging.error("Failed to reassemble data")
            return None

    def _create_handshake_payload(self, session_id: str, protocol_version: int) -> bytes:
        """
        Create handshake message payload

        Format:
        - Session ID (16 bytes UUID)
        - Protocol version (1 byte)

        Args:
            session_id: Session UUID as string
            protocol_version: Protocol version number

        Returns:
            Encoded payload
        """
        # Convert UUID string to bytes
        session_uuid = uuid.UUID(session_id)
        session_bytes = session_uuid.bytes

        # Add protocol version
        version_byte = struct.pack('B', protocol_version)

        return session_bytes + version_byte

    def _parse_handshake_payload(self, payload: bytes) -> Optional[Tuple[str, int]]:
        """
        Parse handshake message payload

        Args:
            payload: Raw payload bytes

        Returns:
            Tuple of (session_id, protocol_version) or None if invalid
        """
        if len(payload) < 17:  # 16 bytes UUID + 1 byte version
            logging.error(f"Invalid handshake payload length: {len(payload)}")
            return None

        try:
            # Extract session ID
            session_bytes = payload[:16]
            session_uuid = uuid.UUID(bytes=session_bytes)
            session_id = str(session_uuid)

            # Extract protocol version
            protocol_version = struct.unpack('B', payload[16:17])[0]

            return session_id, protocol_version

        except Exception as e:
            logging.error(f"Failed to parse handshake payload: {e}")
            return None

    def _transition_state(self, new_state: HandshakeState) -> bool:
        """
        Transition handshake state machine to new state

        Args:
            new_state: Target state

        Returns:
            True if transition is valid
        """
        if not self._handshake_session:
            logging.error("No active handshake session")
            return False

        old_state = self._handshake_session.state

        # Define valid state transitions
        valid_transitions = {
            HandshakeState.IDLE: [HandshakeState.INIT_SENT, HandshakeState.ACK_SENT],
            HandshakeState.INIT_SENT: [HandshakeState.ACK_SENT, HandshakeState.ESTABLISHED, HandshakeState.FAILED],
            HandshakeState.ACK_SENT: [HandshakeState.READY_SENT, HandshakeState.ESTABLISHED, HandshakeState.FAILED],
            HandshakeState.READY_SENT: [HandshakeState.ESTABLISHED, HandshakeState.FAILED],
            HandshakeState.ESTABLISHED: [HandshakeState.FAILED],
            HandshakeState.FAILED: [HandshakeState.IDLE],
        }

        if new_state not in valid_transitions.get(old_state, []):
            logging.warning(f"Invalid state transition: {old_state} -> {new_state}")
            return False

        self._handshake_session.state = new_state
        self._handshake_session.update_activity()

        logging.info(f"Handshake state: {old_state.name} -> {new_state.name}")
        return True

    def _check_handshake_timeout(self) -> bool:
        """
        Check if handshake has timed out

        Returns:
            True if timed out
        """
        if not self._handshake_session:
            return False

        if self._handshake_session.state == HandshakeState.ESTABLISHED:
            return False

        elapsed = self._handshake_session.elapsed_time()
        if elapsed > self.HANDSHAKE_TIMEOUT:
            logging.error(f"Handshake timeout after {elapsed:.1f}s")
            self._transition_state(HandshakeState.FAILED)
            return True

        return False

    def _generate_session_id(self) -> str:
        """
        Generate a unique session ID

        Returns:
            UUID string for session identification
        """
        session_uuid = uuid.uuid4()
        session_id = str(session_uuid)
        logging.info(f"Generated session ID: {session_id}")
        return session_id

    def _check_protocol_version(self, peer_version: int) -> bool:
        """
        Check if peer protocol version is compatible

        Args:
            peer_version: Peer's protocol version number

        Returns:
            True if compatible, False otherwise
        """
        # For now, require exact version match
        # Future: could support backwards compatibility ranges
        compatible = peer_version == self.PROTOCOL_VERSION

        if compatible:
            logging.info(f"Protocol version check passed: peer={peer_version}, local={self.PROTOCOL_VERSION}")
        else:
            logging.error(f"Protocol version mismatch: peer={peer_version}, local={self.PROTOCOL_VERSION}")

        return compatible

    def _init_handshake_session(self, role: str) -> HandshakeSession:
        """
        Initialize a new handshake session

        Args:
            role: 'initiator' or 'responder'

        Returns:
            New HandshakeSession object
        """
        session_id = self._generate_session_id()
        session = HandshakeSession(
            session_id=session_id,
            protocol_version=self.PROTOCOL_VERSION,
            state=HandshakeState.IDLE,
            role=role
        )

        logging.info(f"Initialized handshake session: role={role}, session_id={session_id}")
        return session

    def _send_handshake_message(self, audio_manager, frame_type: FrameType,
                                session_id: str, protocol_version: int,
                                retry_count: int = 3, retry_delay: float = 1.0) -> bool:
        """
        Send a handshake message with retry logic

        Args:
            audio_manager: AudioManager instance for transmission
            frame_type: Type of handshake frame (INIT/ACK/READY)
            session_id: Session ID to send
            protocol_version: Protocol version to send
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds

        Returns:
            True if sent successfully
        """
        payload = self._create_handshake_payload(session_id, protocol_version)

        for attempt in range(retry_count):
            try:
                # Encode the handshake frame
                frame_data = audio_manager.encode_frame(payload, sequence_num=0)

                # Prefix with frame type marker
                frame_type_byte = struct.pack('B', frame_type.value)
                message = frame_type_byte + frame_data

                # Transmit via audio
                success = audio_manager.transmit(message)

                if success:
                    logging.info(f"Sent {frame_type.name} handshake message (attempt {attempt + 1}/{retry_count})")
                    return True
                else:
                    logging.warning(f"Failed to send {frame_type.name} (attempt {attempt + 1}/{retry_count})")

            except Exception as e:
                logging.error(f"Error sending {frame_type.name}: {e}")

            # Wait before retry (except on last attempt)
            if attempt < retry_count - 1:
                time.sleep(retry_delay)

        logging.error(f"Failed to send {frame_type.name} after {retry_count} attempts")
        return False

    def _receive_handshake_message(self, audio_manager, timeout: float = 30.0) -> Optional[Tuple[FrameType, str, int]]:
        """
        Receive a handshake message with timeout

        Args:
            audio_manager: AudioManager instance for reception
            timeout: Timeout in seconds

        Returns:
            Tuple of (frame_type, session_id, protocol_version) or None if timeout/error
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Receive audio data
                received_data = audio_manager.receive(timeout=min(5.0, timeout - (time.time() - start_time)))

                if not received_data or len(received_data) < 2:
                    continue

                # Extract frame type
                frame_type_value = struct.unpack('B', received_data[0:1])[0]
                try:
                    frame_type = FrameType(frame_type_value)
                except ValueError:
                    logging.warning(f"Unknown frame type: {frame_type_value}")
                    continue

                # Only process handshake frames
                if frame_type not in [FrameType.HANDSHAKE_INIT, FrameType.HANDSHAKE_ACK, FrameType.HANDSHAKE_READY]:
                    logging.warning(f"Received non-handshake frame: {frame_type.name}")
                    continue

                # Decode frame
                frame_result = audio_manager.decode_frame(received_data[1:])
                if not frame_result:
                    logging.warning("Failed to decode handshake frame")
                    continue

                seq_num, payload = frame_result

                # Parse handshake payload
                parsed = self._parse_handshake_payload(payload)
                if not parsed:
                    logging.warning("Failed to parse handshake payload")
                    continue

                session_id, protocol_version = parsed

                logging.info(f"Received {frame_type.name}: session_id={session_id}, version={protocol_version}")
                return frame_type, session_id, protocol_version

            except Exception as e:
                logging.error(f"Error receiving handshake message: {e}")
                continue

        logging.warning(f"Handshake receive timeout after {timeout}s")
        return None

    def _wait_with_timeout(self, audio_manager, expected_frame_type: FrameType,
                          timeout: float = 30.0) -> Optional[Tuple[str, int]]:
        """
        Wait for a specific handshake message type with timeout

        Args:
            audio_manager: AudioManager instance
            expected_frame_type: Expected frame type to receive
            timeout: Timeout in seconds

        Returns:
            Tuple of (session_id, protocol_version) or None if timeout/error
        """
        result = self._receive_handshake_message(audio_manager, timeout)

        if not result:
            return None

        frame_type, session_id, protocol_version = result

        if frame_type != expected_frame_type:
            logging.warning(f"Expected {expected_frame_type.name}, got {frame_type.name}")
            return None

        return session_id, protocol_version

    def perform_handshake(self, audio_manager, is_initiator: Optional[bool] = None) -> Tuple[bool, str]:
        """
        Perform acoustic handshake with auto-detection support

        Implements three-way handshake protocol:
        - INIT: Initiator sends GIBBERLAN_INIT beacon
        - ACK: Responder sends GIBBERLAN_ACK response
        - READY: Initiator sends GIBBERLAN_READY confirmation

        Args:
            audio_manager: AudioManager instance for audio I/O
            is_initiator: Role - True=initiator, False=responder, None=auto-detect

        Returns:
            Tuple of (success, session_id)
        """
        from gibberish.audio import AudioManager

        if not isinstance(audio_manager, AudioManager):
            logging.error("Invalid audio_manager instance")
            return False, ""

        # Auto-detection mode
        if is_initiator is None:
            logging.info("Auto-detecting role...")
            is_initiator = self._auto_detect_role(audio_manager)
            logging.info(f"Auto-detected role: {'initiator' if is_initiator else 'responder'}")

        # Perform handshake based on role
        if is_initiator:
            return self._perform_initiator_handshake(audio_manager)
        else:
            return self._perform_responder_handshake(audio_manager)

    def _auto_detect_role(self, audio_manager, detection_timeout: float = 5.0) -> bool:
        """
        Auto-detect whether to be initiator or responder

        Listens for INIT beacons for a short period. If received, become responder.
        Otherwise, become initiator.

        Args:
            audio_manager: AudioManager instance
            detection_timeout: Time to listen for INIT beacons

        Returns:
            True if should be initiator, False if should be responder
        """
        logging.info(f"Listening for INIT beacons for {detection_timeout}s...")

        result = self._receive_handshake_message(audio_manager, timeout=detection_timeout)

        if result:
            frame_type, session_id, protocol_version = result
            if frame_type == FrameType.HANDSHAKE_INIT:
                logging.info("Detected INIT beacon - becoming responder")
                return False

        logging.info("No INIT beacon detected - becoming initiator")
        return True

    def _perform_initiator_handshake(self, audio_manager) -> Tuple[bool, str]:
        """
        Perform handshake as initiator

        Protocol flow:
        1. Send GIBBERLAN_INIT beacon every 2 seconds
        2. Wait for GIBBERLAN_ACK response (30s timeout)
        3. Send GIBBERLAN_READY confirmation
        4. Connection established

        Args:
            audio_manager: AudioManager instance

        Returns:
            Tuple of (success, session_id)
        """
        # Initialize session
        self._handshake_session = self._init_handshake_session('initiator')
        session_id = self._handshake_session.session_id

        logging.info("Starting handshake as INITIATOR")

        try:
            # Phase 1: Send INIT beacons and wait for ACK
            self._transition_state(HandshakeState.INIT_SENT)

            start_time = time.time()
            ack_received = False

            while not ack_received and time.time() - start_time < self.HANDSHAKE_TIMEOUT:
                # Send INIT beacon
                logging.info("Broadcasting GIBBERLAN_INIT beacon...")
                success = self._send_handshake_message(
                    audio_manager,
                    FrameType.HANDSHAKE_INIT,
                    session_id,
                    self.PROTOCOL_VERSION,
                    retry_count=1  # Single attempt per beacon
                )

                if not success:
                    logging.warning("Failed to send INIT beacon")

                # Wait for ACK with short timeout
                result = self._wait_with_timeout(
                    audio_manager,
                    FrameType.HANDSHAKE_ACK,
                    timeout=self.INIT_BEACON_INTERVAL
                )

                if result:
                    peer_session_id, peer_version = result

                    # Verify protocol version
                    if not self._check_protocol_version(peer_version):
                        logging.error("Protocol version incompatible")
                        self._transition_state(HandshakeState.FAILED)
                        return False, ""

                    # Store peer info
                    self._handshake_session.peer_session_id = peer_session_id
                    self._handshake_session.peer_protocol_version = peer_version

                    logging.info(f"Received ACK from peer: {peer_session_id}")
                    ack_received = True
                    self._transition_state(HandshakeState.ACK_SENT)
                    break

            if not ack_received:
                logging.error("Timeout waiting for ACK")
                self._transition_state(HandshakeState.FAILED)
                return False, ""

            # Phase 2: Send READY confirmation
            logging.info("Sending GIBBERLAN_READY confirmation...")
            success = self._send_handshake_message(
                audio_manager,
                FrameType.HANDSHAKE_READY,
                session_id,
                self.PROTOCOL_VERSION,
                retry_count=3
            )

            if not success:
                logging.error("Failed to send READY")
                self._transition_state(HandshakeState.FAILED)
                return False, ""

            # Handshake complete
            self._transition_state(HandshakeState.ESTABLISHED)
            self._connection_established = True

            elapsed = time.time() - start_time
            logging.info(f"Handshake ESTABLISHED as initiator (took {elapsed:.2f}s)")
            logging.info(f"Session ID: {session_id}")
            logging.info(f"Peer Session ID: {self._handshake_session.peer_session_id}")

            return True, session_id

        except Exception as e:
            logging.error(f"Initiator handshake failed: {e}")
            self._transition_state(HandshakeState.FAILED)
            return False, ""

    def _perform_responder_handshake(self, audio_manager) -> Tuple[bool, str]:
        """
        Perform handshake as responder

        Protocol flow:
        1. Listen for GIBBERLAN_INIT beacon (30s timeout)
        2. Send GIBBERLAN_ACK response
        3. Wait for GIBBERLAN_READY confirmation (30s timeout)
        4. Connection established

        Args:
            audio_manager: AudioManager instance

        Returns:
            Tuple of (success, session_id)
        """
        # Initialize session
        self._handshake_session = self._init_handshake_session('responder')
        session_id = self._handshake_session.session_id

        logging.info("Starting handshake as RESPONDER")

        try:
            # Phase 1: Wait for INIT beacon
            logging.info("Listening for GIBBERLAN_INIT beacon...")

            result = self._wait_with_timeout(
                audio_manager,
                FrameType.HANDSHAKE_INIT,
                timeout=self.HANDSHAKE_TIMEOUT
            )

            if not result:
                logging.error("Timeout waiting for INIT beacon")
                self._transition_state(HandshakeState.FAILED)
                return False, ""

            peer_session_id, peer_version = result

            # Verify protocol version
            if not self._check_protocol_version(peer_version):
                logging.error("Protocol version incompatible")
                self._transition_state(HandshakeState.FAILED)
                return False, ""

            # Store peer info
            self._handshake_session.peer_session_id = peer_session_id
            self._handshake_session.peer_protocol_version = peer_version

            logging.info(f"Received INIT from peer: {peer_session_id}")

            # Phase 2: Send ACK response
            logging.info("Sending GIBBERLAN_ACK response...")
            success = self._send_handshake_message(
                audio_manager,
                FrameType.HANDSHAKE_ACK,
                session_id,
                self.PROTOCOL_VERSION,
                retry_count=3
            )

            if not success:
                logging.error("Failed to send ACK")
                self._transition_state(HandshakeState.FAILED)
                return False, ""

            self._transition_state(HandshakeState.ACK_SENT)

            # Phase 3: Wait for READY confirmation
            logging.info("Waiting for GIBBERLAN_READY confirmation...")

            result = self._wait_with_timeout(
                audio_manager,
                FrameType.HANDSHAKE_READY,
                timeout=self.HANDSHAKE_TIMEOUT
            )

            if not result:
                logging.error("Timeout waiting for READY confirmation")
                self._transition_state(HandshakeState.FAILED)
                return False, ""

            ready_session_id, ready_version = result

            # Verify it's from the same peer
            if ready_session_id != peer_session_id:
                logging.error(f"Session ID mismatch: expected {peer_session_id}, got {ready_session_id}")
                self._transition_state(HandshakeState.FAILED)
                return False, ""

            # Handshake complete
            self._transition_state(HandshakeState.ESTABLISHED)
            self._connection_established = True

            elapsed = self._handshake_session.elapsed_time()
            logging.info(f"Handshake ESTABLISHED as responder (took {elapsed:.2f}s)")
            logging.info(f"Session ID: {session_id}")
            logging.info(f"Peer Session ID: {self._handshake_session.peer_session_id}")

            return True, session_id

        except Exception as e:
            logging.error(f"Responder handshake failed: {e}")
            self._transition_state(HandshakeState.FAILED)
            return False, ""

    def get_handshake_status(self) -> Dict[str, any]:
        """
        Get current handshake status and information

        Returns:
            Dictionary with handshake details:
            - established: Whether connection is established
            - state: Current HandshakeState
            - role: 'initiator' or 'responder' or None
            - session_id: Local session ID
            - peer_session_id: Peer's session ID
            - protocol_version: Local protocol version
            - peer_protocol_version: Peer's protocol version
            - elapsed_time: Time since handshake start
        """
        if not self._handshake_session:
            return {
                'established': False,
                'state': None,
                'role': None,
                'session_id': None,
                'peer_session_id': None,
                'protocol_version': self.PROTOCOL_VERSION,
                'peer_protocol_version': None,
                'elapsed_time': 0.0
            }

        return {
            'established': self._connection_established,
            'state': self._handshake_session.state,
            'role': self._handshake_session.role,
            'session_id': self._handshake_session.session_id,
            'peer_session_id': self._handshake_session.peer_session_id,
            'protocol_version': self._handshake_session.protocol_version,
            'peer_protocol_version': self._handshake_session.peer_protocol_version,
            'elapsed_time': self._handshake_session.elapsed_time()
        }

    def assess_connection_quality(self, audio_manager) -> Dict[str, any]:
        """
        Assess connection quality during/after handshake

        Args:
            audio_manager: AudioManager instance

        Returns:
            Dictionary with quality metrics:
            - connection_type: 'cable', 'air', or 'unknown'
            - snr_db: Signal-to-noise ratio in dB
            - quality_score: Overall quality score (0-100)
            - bandwidth_estimate: Estimated bandwidth in bps
            - recommendation: Suggested transmission mode
        """
        from gibberish.audio import ConnectionType, TransmissionMode

        # Get connection info from AudioManager
        conn_info = audio_manager.get_connection_info()

        connection_type = conn_info.get('type', ConnectionType.UNKNOWN)
        snr_db = conn_info.get('snr_db', 0.0)

        # Calculate quality score based on SNR and connection type
        if connection_type == ConnectionType.CABLE:
            # Cable: expect SNR > 40dB
            if snr_db > 50:
                quality_score = 100
            elif snr_db > 45:
                quality_score = 90
            elif snr_db > 40:
                quality_score = 80
            else:
                quality_score = max(0, int((snr_db / 40) * 70))
        elif connection_type == ConnectionType.AIR:
            # Air: expect SNR 10-30dB
            if snr_db > 25:
                quality_score = 100
            elif snr_db > 20:
                quality_score = 85
            elif snr_db > 15:
                quality_score = 70
            elif snr_db > 10:
                quality_score = 50
            else:
                quality_score = max(0, int((snr_db / 10) * 40))
        else:
            # Unknown connection
            quality_score = 50

        # Estimate bandwidth based on connection type
        if connection_type == ConnectionType.CABLE:
            bandwidth_estimate = 350  # bps
            recommendation = "Use ULTRASONIC mode for best performance"
        elif connection_type == ConnectionType.AIR:
            bandwidth_estimate = 212  # bps
            recommendation = "Use AUDIBLE mode for reliable transmission"
        else:
            bandwidth_estimate = 200  # Conservative estimate
            recommendation = "Perform cable detection for optimization"

        return {
            'connection_type': connection_type.value if hasattr(connection_type, 'value') else str(connection_type),
            'snr_db': snr_db,
            'quality_score': quality_score,
            'bandwidth_estimate': bandwidth_estimate,
            'recommendation': recommendation
        }

    def display_connection_status(self, audio_manager):
        """
        Display clear connection status in terminal

        Args:
            audio_manager: AudioManager instance
        """
        from gibberish.audio import ConnectionType

        # Get handshake status
        handshake_status = self.get_handshake_status()

        # Get connection quality
        quality = self.assess_connection_quality(audio_manager)

        # Header
        print("\n" + "=" * 70)
        print("  GIBBERLAN CONNECTION STATUS")
        print("=" * 70)

        # Handshake Status
        print("\n[HANDSHAKE]")
        if handshake_status['established']:
            print(f"  Status: ✓ ESTABLISHED")
            print(f"  Role: {handshake_status['role'].upper()}")
            print(f"  Session ID: {handshake_status['session_id'][:8]}...")
            print(f"  Peer Session: {handshake_status['peer_session_id'][:8] if handshake_status['peer_session_id'] else 'N/A'}...")
            print(f"  Duration: {handshake_status['elapsed_time']:.2f}s")
        else:
            state = handshake_status['state']
            print(f"  Status: ✗ NOT ESTABLISHED")
            print(f"  State: {state.name if state else 'IDLE'}")

        # Connection Info
        print("\n[CONNECTION]")
        conn_type = quality['connection_type']
        if conn_type == 'cable':
            conn_symbol = "🔌"
            conn_name = "CABLE (AUX)"
        elif conn_type == 'air':
            conn_symbol = "📡"
            conn_name = "AIR (WIRELESS)"
        else:
            conn_symbol = "❓"
            conn_name = "UNKNOWN"

        print(f"  Type: {conn_symbol} {conn_name}")
        print(f"  SNR: {quality['snr_db']:.1f} dB")

        # Quality Score
        score = quality['quality_score']
        if score >= 90:
            quality_bar = "█████ EXCELLENT"
        elif score >= 75:
            quality_bar = "████░ GOOD"
        elif score >= 50:
            quality_bar = "███░░ FAIR"
        elif score >= 25:
            quality_bar = "██░░░ POOR"
        else:
            quality_bar = "█░░░░ VERY POOR"

        print(f"  Quality: {quality_bar} ({score}/100)")

        # Bandwidth
        print(f"  Est. Bandwidth: {quality['bandwidth_estimate']} bps")

        # Transmission Mode
        print("\n[TRANSMISSION]")
        current_mode = audio_manager.get_transmission_mode()
        print(f"  Mode: {current_mode.name}")
        print(f"  Protocol: v{self.PROTOCOL_VERSION}")

        # Recommendation
        print(f"\n[RECOMMENDATION]")
        print(f"  {quality['recommendation']}")

        print("\n" + "=" * 70 + "\n")

    def get_connection_summary(self, audio_manager) -> str:
        """
        Get a concise connection summary string

        Args:
            audio_manager: AudioManager instance

        Returns:
            One-line connection summary
        """
        from gibberish.audio import ConnectionType

        handshake = self.get_handshake_status()
        quality = self.assess_connection_quality(audio_manager)

        if not handshake['established']:
            return "Connection: NOT ESTABLISHED"

        conn_type = quality['connection_type']
        role = handshake['role']
        score = quality['quality_score']

        type_emoji = "🔌" if conn_type == 'cable' else "📡" if conn_type == 'air' else "❓"

        return f"Connection: {type_emoji} {conn_type.upper()} | Role: {role.upper()} | Quality: {score}/100"

    def send_tree_hash(self, audio_manager, tree_hash: str) -> bool:
        """
        Send tree hash to peer for synchronization verification

        Args:
            audio_manager: AudioManager instance
            tree_hash: SHA256 tree hash as hex string (64 characters)

        Returns:
            True if sent successfully
        """
        try:
            if len(tree_hash) != 64:
                logging.error(f"Invalid tree hash length: {len(tree_hash)} (expected 64)")
                return False

            # Create payload with tree hash
            payload = tree_hash.encode('ascii')

            # Create TREE_HASH frame
            file_id = str(uuid.uuid4())
            frame = self.create_frame(
                frame_type=FrameType.TREE_HASH,
                file_id=file_id,
                total_frames=1,
                payload=payload
            )

            # Pack and send
            frame_data = frame.pack()
            success = audio_manager.transmit(frame_data)

            if success:
                logging.info(f"Sent tree hash: {tree_hash[:16]}...")
            else:
                logging.error("Failed to send tree hash")

            return success

        except Exception as e:
            logging.error(f"Error sending tree hash: {e}")
            return False

    def receive_tree_hash(self, audio_manager, timeout: float = 30.0) -> Optional[str]:
        """
        Receive tree hash from peer

        Args:
            audio_manager: AudioManager instance
            timeout: Timeout in seconds

        Returns:
            Tree hash string or None if timeout/error
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Receive data
                received_data = audio_manager.receive(timeout=min(5.0, timeout - (time.time() - start_time)))

                if not received_data or len(received_data) < 26:
                    continue

                # Unpack frame
                frame = Frame.unpack(received_data)
                if not frame:
                    continue

                # Verify checksum
                if not frame.verify_checksum():
                    logging.warning("Received frame with invalid checksum")
                    continue

                # Check if it's a TREE_HASH frame
                if frame.frame_type == FrameType.TREE_HASH:
                    try:
                        tree_hash = frame.payload.decode('ascii')
                        if len(tree_hash) == 64:
                            logging.info(f"Received tree hash: {tree_hash[:16]}...")
                            return tree_hash
                        else:
                            logging.error(f"Invalid tree hash length: {len(tree_hash)}")
                    except UnicodeDecodeError:
                        logging.error("Failed to decode tree hash")
                        continue

            except Exception as e:
                logging.debug(f"Error receiving tree hash: {e}")
                continue

        logging.warning(f"Timeout waiting for tree hash after {timeout}s")
        return None

    def verify_sync_completion(self, audio_manager, local_tree_hash: str,
                              timeout: float = 30.0) -> Dict[str, any]:
        """
        Exchange tree hashes and verify synchronization completion

        This implements the final verification step after all patches are applied.
        Both peers exchange their tree hashes and compare them to ensure sync success.

        Args:
            audio_manager: AudioManager instance
            local_tree_hash: Local tree hash after applying patches
            timeout: Timeout for hash exchange

        Returns:
            Dictionary with verification results:
            {
                'success': bool,
                'local_hash': str,
                'peer_hash': str or None,
                'hashes_match': bool,
                'error': str or None,
                'diagnostics': dict with mismatch details if applicable
            }
        """
        result = {
            'success': False,
            'local_hash': local_tree_hash,
            'peer_hash': None,
            'hashes_match': False,
            'error': None,
            'diagnostics': {}
        }

        try:
            logging.info("Starting tree hash exchange for sync verification...")

            # Send local tree hash to peer
            send_success = self.send_tree_hash(audio_manager, local_tree_hash)
            if not send_success:
                result['error'] = "Failed to send local tree hash"
                return result

            # Receive peer's tree hash
            peer_hash = self.receive_tree_hash(audio_manager, timeout=timeout)
            if not peer_hash:
                result['error'] = "Failed to receive peer tree hash (timeout)"
                return result

            result['peer_hash'] = peer_hash

            # Compare hashes
            if local_tree_hash == peer_hash:
                result['hashes_match'] = True
                result['success'] = True
                logging.info("Sync verification SUCCESS: tree hashes match")
            else:
                result['hashes_match'] = False
                result['error'] = "Tree hash mismatch - synchronization incomplete"
                result['diagnostics'] = {
                    'local_hash': local_tree_hash,
                    'peer_hash': peer_hash,
                    'message': 'File states differ between sender and receiver'
                }
                logging.error(f"Sync verification FAILED: hash mismatch")
                logging.error(f"  Local:  {local_tree_hash}")
                logging.error(f"  Peer:   {peer_hash}")

            # Send SYNC_COMPLETE acknowledgment if hashes match
            if result['hashes_match']:
                self._send_sync_complete(audio_manager, success=True)
            else:
                self._send_sync_complete(audio_manager, success=False)

        except Exception as e:
            result['error'] = f"Unexpected error during sync verification: {e}"
            logging.error(result['error'])

        return result

    def _send_sync_complete(self, audio_manager, success: bool) -> bool:
        """
        Send SYNC_COMPLETE frame to indicate sync status

        Args:
            audio_manager: AudioManager instance
            success: Whether sync was successful

        Returns:
            True if sent successfully
        """
        try:
            # Create payload: 1 byte status (0=failure, 1=success)
            payload = struct.pack('B', 1 if success else 0)

            # Create SYNC_COMPLETE frame
            file_id = str(uuid.uuid4())
            frame = self.create_frame(
                frame_type=FrameType.SYNC_COMPLETE,
                file_id=file_id,
                total_frames=1,
                payload=payload
            )

            # Pack and send
            frame_data = frame.pack()
            result = audio_manager.transmit(frame_data)

            if result:
                logging.info(f"Sent SYNC_COMPLETE: {'SUCCESS' if success else 'FAILURE'}")

            return result

        except Exception as e:
            logging.error(f"Error sending SYNC_COMPLETE: {e}")
            return False


@dataclass
class ReassemblyBuffer:
    """
    Buffer for reassembling frames into complete data

    Tracks received frames and handles out-of-order delivery
    """
    file_id: str
    total_frames: int
    received_frames: Dict[int, bytes] = field(default_factory=dict)  # seq -> payload
    complete: bool = False
    start_time: float = field(default_factory=time.time)

    def add_frame(self, sequence_number: int, payload: bytes) -> bool:
        """
        Add a received frame to the buffer

        Args:
            sequence_number: Frame sequence number
            payload: Frame payload data

        Returns:
            True if this completes the reassembly
        """
        # Store frame payload
        self.received_frames[sequence_number] = payload

        # Check if all frames received
        if len(self.received_frames) == self.total_frames:
            self.complete = True
            logging.info(f"All {self.total_frames} frames received for file {self.file_id[:8]}...")
            return True

        return False

    def is_complete(self) -> bool:
        """Check if all frames have been received"""
        return self.complete

    def get_missing_sequences(self) -> List[int]:
        """
        Get list of missing frame sequence numbers

        Returns:
            List of missing sequence numbers
        """
        received_seqs = set(self.received_frames.keys())
        all_seqs = set(range(self.total_frames))
        missing = sorted(all_seqs - received_seqs)
        return missing

    def get_progress(self) -> float:
        """
        Get reassembly progress as percentage

        Returns:
            Progress percentage (0-100)
        """
        if self.total_frames == 0:
            return 0.0
        return (len(self.received_frames) / self.total_frames) * 100.0

    def reassemble(self) -> Optional[bytes]:
        """
        Reassemble frames into complete data

        Returns:
            Complete reassembled data or None if incomplete
        """
        if not self.complete:
            missing = self.get_missing_sequences()
            logging.error(f"Cannot reassemble: missing {len(missing)} frames: {missing[:10]}...")
            return None

        # Reassemble in sequence order
        data = b""
        for seq in range(self.total_frames):
            if seq not in self.received_frames:
                logging.error(f"Missing frame {seq} during reassembly")
                return None
            data += self.received_frames[seq]

        logging.info(f"Reassembled {len(data)} bytes from {self.total_frames} frames")
        return data

    def elapsed_time(self) -> float:
        """Get time since reassembly started"""
        return time.time() - self.start_time
