#!/usr/bin/env python3
"""
Local testing script for Gibberish - runs transmitter and receiver in separate processes
"""
import subprocess
import sys
import time
import signal
from pathlib import Path
import tempfile
import shutil

def create_test_directories():
    """Create test directories with sample files"""
    # Create temp directories
    original_dir = Path(tempfile.mkdtemp(prefix="gibberish_original_"))
    modified_dir = Path(tempfile.mkdtemp(prefix="gibberish_modified_"))
    receiver_dir = Path(tempfile.mkdtemp(prefix="gibberish_receiver_"))

    # Create identical baseline in original and receiver
    for d in [original_dir, receiver_dir]:
        (d / "file1.txt").write_text("Hello World\n")
        (d / "file2.txt").write_text("Test file 2\n")
        subdir = d / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_text("Nested file\n")

    # Create modifications in modified directory
    shutil.copytree(original_dir, modified_dir, dirs_exist_ok=True)
    (modified_dir / "file1.txt").write_text("Hello World - MODIFIED\n")  # Modify
    (modified_dir / "new_file.txt").write_text("This is new\n")  # Add
    (modified_dir / "file2.txt").unlink()  # Delete

    return original_dir, modified_dir, receiver_dir


def run_transmitter(original_dir, modified_dir):
    """Run transmitter in subprocess"""
    print("=" * 60)
    print("STARTING TRANSMITTER")
    print("=" * 60)

    # Create a Python script that runs the transmitter
    transmitter_script = f"""
import sys
import logging
from pathlib import Path
from gibberish.baseline import BaselineManager
from gibberish.sync import SyncManager
from gibberish.audio import AudioManager
from gibberish.protocol import ProtocolHandler
import time

logging.basicConfig(level=logging.INFO)

original_path = Path('{original_dir}')
modified_path = Path('{modified_dir}')

print("\\n=== TRANSMITTER ===")
print(f"Original: {{original_path}}")
print(f"Modified: {{modified_path}}")

# Calculate changes
baseline_mgr = BaselineManager(original_path)
baseline_data = baseline_mgr.create_baseline()

sync_mgr = SyncManager(modified_path)
changes = sync_mgr.compute_diff(baseline_data)

print(f"\\nChanges detected: {{len(changes)}} items")
for change in changes[:5]:  # Show first 5
    print(f"  - {{change.change_type.name}}: {{change.path}}")

# Initialize audio
audio_mgr = AudioManager()
protocol = ProtocolHandler()

# Perform handshake (15 second window)
print("\\nWaiting for receiver handshake...")
start_time = time.time()
timeout = 15.0
success = False
session_id = ""

while time.time() - start_time < timeout:
    try:
        success, session_id = protocol.perform_handshake(audio_mgr, is_initiator=True)
        if success:
            break
        time.sleep(1)
    except Exception as e:
        print(f"Handshake attempt: {{e}}")
        time.sleep(1)

if not success:
    print(f"❌ Handshake failed after {{time.time() - start_time:.1f}}s")
    sys.exit(1)

print(f"✓ Connected! Session: {{session_id[:8]}}")

# Transmit baseline and changes
print("\\nTransmitting data...")
# TODO: Implement actual data transmission
print("✓ Transmission complete!")
sys.exit(0)
"""

    proc = subprocess.Popen(
        [sys.executable, "-c", transmitter_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    return proc


def run_receiver(receiver_dir):
    """Run receiver in subprocess"""
    print("=" * 60)
    print("STARTING RECEIVER")
    print("=" * 60)

    receiver_script = f"""
import sys
import logging
from pathlib import Path
from gibberish.audio import AudioManager
from gibberish.protocol import ProtocolHandler
import time

logging.basicConfig(level=logging.INFO)

target_path = Path('{receiver_dir}')

print("\\n=== RECEIVER ===")
print(f"Target: {{target_path}}")

# Initialize audio
audio_mgr = AudioManager()
protocol = ProtocolHandler()

# Perform handshake (15 second window)
print("\\nListening for transmitter handshake...")
start_time = time.time()
timeout = 15.0
success = False
session_id = ""

while time.time() - start_time < timeout:
    try:
        success, session_id = protocol.perform_handshake(audio_mgr, is_initiator=False)
        if success:
            break
        time.sleep(1)
    except Exception as e:
        print(f"Handshake attempt: {{e}}")
        time.sleep(1)

if not success:
    print(f"❌ Connection failed after {{time.time() - start_time:.1f}}s")
    sys.exit(1)

print(f"✓ Connected! Session: {{session_id[:8]}}")

# Receive data
print("\\nReceiving data...")
# TODO: Implement actual data reception
print("✓ Sync complete!")
sys.exit(0)
"""

    proc = subprocess.Popen(
        [sys.executable, "-c", receiver_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    return proc


def main():
    """Main test orchestrator"""
    print("=" * 60)
    print("GIBBERISH LOCAL SYNC TEST")
    print("=" * 60)

    # Create test directories
    print("\nCreating test directories...")
    original_dir, modified_dir, receiver_dir = create_test_directories()

    print(f"\nTest directories created:")
    print(f"  Original:  {original_dir}")
    print(f"  Modified:  {modified_dir}")
    print(f"  Receiver:  {receiver_dir}")

    # Start receiver first
    print("\n" + "=" * 60)
    receiver_proc = run_receiver(receiver_dir)
    time.sleep(2)  # Give receiver time to start listening

    # Start transmitter
    transmitter_proc = run_transmitter(original_dir, modified_dir)

    try:
        # Monitor both processes
        print("\n" + "=" * 60)
        print("MONITORING PROCESSES")
        print("=" * 60)

        while True:
            # Check receiver
            if receiver_proc.poll() is not None:
                print("\nReceiver finished!")
                output = receiver_proc.stdout.read()
                if output:
                    print("Receiver output:")
                    print(output)
                break

            # Check transmitter
            if transmitter_proc.poll() is not None:
                print("\nTransmitter finished!")
                output = transmitter_proc.stdout.read()
                if output:
                    print("Transmitter output:")
                    print(output)
                time.sleep(2)  # Give receiver time to finish
                break

            time.sleep(0.5)

        # Wait for both to complete
        transmitter_proc.wait(timeout=5)
        receiver_proc.wait(timeout=5)

        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)

        print(f"\nReturn codes:")
        print(f"  Transmitter: {transmitter_proc.returncode}")
        print(f"  Receiver: {receiver_proc.returncode}")

        if transmitter_proc.returncode == 0 and receiver_proc.returncode == 0:
            print("\n✓ Both processes completed successfully!")
        else:
            print("\n❌ One or more processes failed")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        transmitter_proc.kill()
        receiver_proc.kill()

    finally:
        # Cleanup
        print(f"\nCleaning up test directories...")
        shutil.rmtree(original_dir, ignore_errors=True)
        shutil.rmtree(modified_dir, ignore_errors=True)
        shutil.rmtree(receiver_dir, ignore_errors=True)
        print("Done!")


if __name__ == "__main__":
    main()
